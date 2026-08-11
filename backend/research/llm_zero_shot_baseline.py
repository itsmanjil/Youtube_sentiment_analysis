#!/usr/bin/env python3
"""
Zero-shot LLM baseline on the human-labeled gold set.

Every other model in this thesis (TF-IDF/LogReg/SVM, the CNN-BiLSTM hybrid,
DeBERTa-v3, the fuzzy/ensemble stack) is fine-tuned on the project's own
training data. This script scores the same gold set with a small,
instruction-tuned LLM that has seen none of it, to establish the
zero-shot comparison point examiners expect in 2026.

Approach
--------
Qwen2.5-1.5B-Instruct (Apache-2.0, ungated — no HF token needed) is asked,
per comment, to classify sentiment as Positive/Neutral/Negative. Each of
those three words tokenizes to exactly one token with this tokenizer, so
rather than free-generating and parsing text, this reads the model's next-
token logits directly at the three candidate token ids and softmaxes them
into a proper 3-way probability distribution — the same shape every other
engine in this project produces (see src/sentiment/base.py::SentimentResult),
which lets this reuse the exact same accuracy/macro-F1/ECE/Brier metrics
research/evaluate_gold_set.py computes for every other model.

Usage
-----
    python research/llm_zero_shot_baseline.py \
        --data data/gold_set_human_reconciled.csv \
        --label_column gold_label
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
# `python research/llm_zero_shot_baseline.py` auto-inserts this script's own
# directory (backend/research/) at sys.path[0] before any of this file's code
# runs. backend/research/transformers/ is a real local subpackage (see
# research/transformers/prob_cube_io.py etc.), so an unqualified
# `import transformers` below would resolve to it instead of the pip-
# installed HuggingFace library. Drop that auto-inserted entry now that
# BASE_DIR (which has no top-level transformers/ of its own) is on the path.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

from src.utils import SENTIMENT_LABELS
from research.evaluation.calibration import compute_calibration_metrics, probs_to_matrix

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for YouTube comments. Given a comment, "
    "reply with exactly one word: Positive, Neutral, or Negative. "
    "Positive means the commenter expresses a favorable opinion; Negative "
    "means an unfavorable one; Neutral means neither, or the sentiment is "
    "unclear/mixed."
)


def _load_gold_set(csv_path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    missing = [col for col in (text_column, label_column) if col not in df.columns]
    if missing:
        raise ValueError(f"Gold set is missing required columns: {missing}")
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].astype(bool)]

    # Match research/ci/gold_set_evaluation.py's convention: only evaluate
    # on rows with a resolved, non-disputed gold label. gold_set_human_
    # reconciled.csv carries 9 rows where the two annotators never reached
    # consensus (is_disputed=yes, gold_label blank) — there is no ground
    # truth to score those against.
    if "is_disputed" in df.columns:
        not_disputed = df["is_disputed"].astype(str).str.strip().str.lower() != "yes"
    else:
        not_disputed = True
    has_label = df[label_column].notna() & (df[label_column].astype(str).str.strip() != "")
    n_before = len(df)
    df = df[has_label & not_disputed].reset_index(drop=True)
    n_skipped = n_before - len(df)
    if n_skipped:
        print(f"Skipping {n_skipped} disputed/unresolved row(s) with no gold label.")

    label_map = {label.lower(): label for label in SENTIMENT_LABELS}
    df[label_column] = df[label_column].astype(str).str.strip().str.lower().map(label_map)
    if df[label_column].isna().any():
        bad = df[df[label_column].isna()].index.tolist()[:5]
        raise ValueError(f"Unrecognized labels at rows: {bad}")
    return df


def _score_examples(texts: List[str], batch_size: int) -> List[Dict[str, float]]:
    """Return a per-example {label: probability} dict for each text, via
    single-token next-token-logit scoring (see module docstring)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()

    label_token_ids = []
    for label in SENTIMENT_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Expected '{label}' to tokenize to exactly one token with "
                f"{MODEL_NAME}, got {len(ids)}: {ids}. The single-token "
                "scoring approach in this script assumes one token per "
                "candidate label."
            )
        label_token_ids.append(ids[0])

    prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    results: List[Dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True)
            outputs = model(**encoded)
            # Last real (non-pad) token of each sequence predicts the next
            # token — with left-padding that's always the final position.
            next_token_logits = outputs.logits[:, -1, :]
            candidate_logits = next_token_logits[:, label_token_ids]
            probs = torch.softmax(candidate_logits, dim=-1)
            for row in probs.tolist():
                results.append(dict(zip(SENTIMENT_LABELS, row)))
            print(f"  scored {min(start + batch_size, len(prompts))}/{len(prompts)}")

    return results


def _to_markdown(result: Dict[str, object]) -> str:
    metrics = result["metrics"]
    cal = metrics["calibration"]
    lines = [
        "# LLM Zero-Shot Baseline (Gold Set)",
        "",
        f"- Model: `{result['model']}`",
        f"- Data: `{result['data']}`",
        f"- Samples: {result['n_samples']}",
        f"- Generated at (UTC): {result['generated_at_utc']}",
        "",
        "| Model | Accuracy | Macro-F1 | ECE | Brier |",
        "| --- | --- | --- | --- | --- |",
        "| llm_zero_shot ({m}) | {acc:.4f} | {f1:.4f} | {ece:.6f} | {brier:.6f} |".format(
            m=result["model"],
            acc=metrics["accuracy"],
            f1=metrics["macro_f1"],
            ece=cal["ece"],
            brier=cal["brier"],
        ),
        "",
        "Compare against the fine-tuned models' rows in "
        "`results/master_results_table.md` / `results/gold_set_evaluation.md` "
        "— this baseline has seen none of the project's training data.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a zero-shot LLM baseline on the gold set."
    )
    parser.add_argument("--data", required=True, help="Path to gold-set CSV.")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="gold_label")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--calibration_bins", type=int, default=15)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: backend/results/llm_zero_shot_gold_set.json).",
    )
    parser.add_argument(
        "--summary_md",
        default=None,
        help="Optional markdown summary path. Defaults to JSON path with .md extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Gold set CSV not found: {data_path}")

    gold = _load_gold_set(data_path, args.text_column, args.label_column)
    texts = gold[args.text_column].tolist()
    y_true = gold[args.label_column].tolist()

    probs = _score_examples(texts, batch_size=args.batch_size)
    y_pred = [max(p, key=p.get) for p in probs]
    prob_matrix = probs_to_matrix(probs, labels=SENTIMENT_LABELS)
    ece, brier = compute_calibration_metrics(
        y_true, prob_matrix, labels=SENTIMENT_LABELS, n_bins=args.calibration_bins
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(SENTIMENT_LABELS)
        ).tolist(),
        "calibration": {
            "ece": round(float(ece), 6),
            "brier": round(float(brier), 6),
            "bins": int(args.calibration_bins),
        },
    }

    result = {
        "model": MODEL_NAME,
        "method": "zero_shot_next_token_logit_scoring",
        "data": str(data_path),
        "n_samples": int(len(gold)),
        "label_distribution": gold[args.label_column].value_counts().to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    out_path = (
        Path(args.output) if args.output else BASE_DIR / "results" / "llm_zero_shot_gold_set.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_path = Path(args.summary_md) if args.summary_md else out_path.with_suffix(".md")
    md_path.write_text(_to_markdown(result), encoding="utf-8")

    print(f"Wrote JSON: {out_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
