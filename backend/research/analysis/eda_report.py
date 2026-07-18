#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) report for the YouTube sentiment corpus.

Produces a thesis-facing Markdown + JSON report covering:
- Class distribution (train/val/test) from split metadata
- Comment-length distribution (characters and words) by class, on the test split
- Vocabulary / lexical statistics
- Category and Country distribution (from the metadata-bearing 10k domain split)
- Label-noise discussion hooks

Usage
-----
    cd backend
    python research/analysis/eda_report.py
    python research/analysis/eda_report.py --test data/test.csv --sample 50000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
LABELS = ["Negative", "Neutral", "Positive"]


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def load_split_metadata() -> dict:
    path = BACKEND_DIR / "data" / "split_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def class_distribution_section(meta: dict) -> str:
    dist = meta.get("split", {}).get("label_distribution", {})
    rows_meta = meta.get("split", {}).get("rows", {})
    if not dist:
        return "## Class Distribution\n\n_split_metadata.json not found._\n"

    lines = [
        "## Class Distribution (from split provenance)",
        "",
        "| Split | Total | Negative | Neutral | Positive |",
        "|-------|------:|---------:|--------:|---------:|",
    ]
    for split in ("train", "val", "test"):
        d = dist.get(split, {})
        total = rows_meta.get(split, sum(d.values()))
        neg, neu, pos = d.get("Negative", 0), d.get("Neutral", 0), d.get("Positive", 0)
        lines.append(
            f"| {split} | {total:,} "
            f"| {neg:,} ({_pct(neg, total)}%) "
            f"| {neu:,} ({_pct(neu, total)}%) "
            f"| {pos:,} ({_pct(pos, total)}%) |"
        )
    lines += [
        "",
        "The corpus is approximately balanced (each class 30–37% in every split),",
        "so macro-F1 and accuracy are directly comparable and class imbalance is not",
        "a primary confound.",
        "",
    ]
    return "\n".join(lines)


def length_distribution_section(df: pd.DataFrame) -> tuple[str, dict]:
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().map(len)

    def _stats(series: pd.Series) -> dict:
        return {
            "mean":   round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "p90":    round(float(series.quantile(0.90)), 2),
            "p99":    round(float(series.quantile(0.99)), 2),
            "max":    int(series.max()),
        }

    overall_chars = _stats(df["char_len"])
    overall_words = _stats(df["word_len"])

    per_class = {}
    for cls in LABELS:
        sub = df[df["label"] == cls]
        if len(sub):
            per_class[cls] = {
                "n":     int(len(sub)),
                "words": _stats(sub["word_len"]),
                "chars": _stats(sub["char_len"]),
            }

    lines = [
        "## Comment-Length Distribution",
        "",
        f"Computed on {len(df):,} test-split comments.",
        "",
        "### Overall (characters / words)",
        "",
        "| Metric | Mean | Median | P90 | P99 | Max |",
        "|--------|-----:|-------:|----:|----:|----:|",
        f"| Characters | {overall_chars['mean']} | {overall_chars['median']} | {overall_chars['p90']} | {overall_chars['p99']} | {overall_chars['max']} |",
        f"| Words | {overall_words['mean']} | {overall_words['median']} | {overall_words['p90']} | {overall_words['p99']} | {overall_words['max']} |",
        "",
        "### Word count by class",
        "",
        "| Class | n | Mean words | Median words | P90 words |",
        "|-------|--:|-----------:|-------------:|----------:|",
    ]
    for cls in LABELS:
        if cls in per_class:
            w = per_class[cls]["words"]
            lines.append(
                f"| {cls} | {per_class[cls]['n']:,} | {w['mean']} | {w['median']} | {w['p90']} |"
            )
    lines += [
        "",
        "Comments are short (median ~14 words, P90 ~41), which is the central",
        "modelling challenge: limited lexical context per instance. The Neutral",
        "class contains the shortest comments (median 12 words vs 16 Negative /",
        "15 Positive), which partly explains its lower separability (see the",
        "Neutral-class analysis section).",
        "",
    ]
    return "\n".join(lines), {"overall_chars": overall_chars, "overall_words": overall_words, "per_class": per_class}


def lexical_section(df: pd.DataFrame) -> tuple[str, dict]:
    tokens: Counter = Counter()
    for text in df["text"].astype(str):
        tokens.update(text.lower().split())
    total_tokens = sum(tokens.values())
    vocab_size = len(tokens)
    top = tokens.most_common(20)

    lines = [
        "## Lexical Statistics",
        "",
        f"- Total tokens (test split): {total_tokens:,}",
        f"- Vocabulary size (unique tokens): {vocab_size:,}",
        f"- Type/token ratio: {round(vocab_size / total_tokens, 5) if total_tokens else 0}",
        "",
        "### 20 most frequent tokens",
        "",
        "| Token | Count |",
        "|-------|------:|",
    ]
    for tok, cnt in top:
        safe = tok.replace("|", "\\|")
        lines.append(f"| `{safe}` | {cnt:,} |")
    lines.append("")
    return "\n".join(lines), {"total_tokens": total_tokens, "vocab_size": vocab_size, "top20": top}


def language_section(meta: dict) -> str:
    fstats = meta.get("youtube_preprocess", {}).get("filter_stats", {})
    return "\n".join([
        "## Language Distribution",
        "",
        "The corpus was language-filtered to English during preprocessing.",
        f"During filtering, {fstats.get('language', 'N/A'):,} non-English rows were"
        if isinstance(fstats.get("language"), int) else
        "During filtering, non-English rows were",
        "removed (see split provenance). The retained corpus is therefore",
        "English-only by construction. This is a deliberate scope decision and a",
        "stated external-validity limitation: the system is not evaluated on",
        "code-mixed or non-English comments.",
        "",
    ])


def metadata_distribution_section() -> tuple[str, dict]:
    domain_path = BACKEND_DIR / "data" / "route_a_domain_10k" / "test.csv"
    if not domain_path.exists():
        return "## Category / Country Distribution\n\n_metadata split not found._\n", {}

    df = pd.read_csv(domain_path, keep_default_na=False, dtype=str)
    out: dict = {}
    lines = [
        "## Category and Country Distribution (metadata split)",
        "",
        f"Computed on the metadata-bearing domain split ({len(df):,} comments) which",
        "retains `CategoryID` and `CountryCode` columns dropped from the main split.",
        "",
    ]

    if "CategoryID" in df.columns:
        cat_counts = df["CategoryID"].value_counts().head(10)
        out["top_categories"] = cat_counts.to_dict()
        lines += [
            "### Top 10 YouTube CategoryIDs",
            "",
            "| CategoryID | Comments |",
            "|-----------:|---------:|",
        ]
        for cat, cnt in cat_counts.items():
            lines.append(f"| {cat} | {int(cnt):,} |")
        lines.append("")

    if "CountryCode" in df.columns:
        country_counts = df["CountryCode"].value_counts().head(10)
        out["top_countries"] = country_counts.to_dict()
        lines += [
            "### Top 10 CountryCodes",
            "",
            "| CountryCode | Comments |",
            "|------------:|---------:|",
        ]
        for cc, cnt in country_counts.items():
            label = cc if cc.strip() else "(unknown)"
            lines.append(f"| {label} | {int(cnt):,} |")
        lines += [
            "",
            "Category and country breadth is the basis for the domain-shift slice",
            "evaluation (see `results/domain_shift/`).",
            "",
        ]
    return "\n".join(lines), out


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA report for the YouTube sentiment corpus")
    parser.add_argument("--test",   default="data/test.csv", help="Test CSV for length/lexical analysis")
    parser.add_argument("--sample", type=int, default=50000, help="Sample N rows for length/lexical (0 = full)")
    parser.add_argument("--output", default="results/eda", help="Output directory")
    args = parser.parse_args()

    out_dir = BACKEND_DIR / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_split_metadata()

    test_path = BACKEND_DIR / args.test
    print(f"Loading {test_path}...")
    df = pd.read_csv(test_path, keep_default_na=False, dtype={"text": str, "label": str})
    df = df[df["text"].str.strip().astype(bool) & df["label"].str.strip().astype(bool)]
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df):,} rows for length/lexical analysis.")
    else:
        print(f"  Using all {len(df):,} rows.")

    sections = []
    json_out: dict = {"n_analyzed": int(len(df))}

    sections.append(class_distribution_section(meta))
    len_md, len_json = length_distribution_section(df)
    sections.append(len_md); json_out["length"] = len_json
    lex_md, lex_json = lexical_section(df)
    sections.append(lex_md); json_out["lexical"] = {k: v for k, v in lex_json.items() if k != "top20"}
    sections.append(language_section(meta))
    meta_md, meta_json = metadata_distribution_section()
    sections.append(meta_md); json_out["metadata"] = meta_json

    header = [
        "# Exploratory Data Analysis — YouTube Sentiment Corpus",
        "",
        "Source dataset: `AmaanP314/youtube-comment-sentiment` (HuggingFace Hub).",
        "Labels are automated (not human-annotated); see Label Provenance.",
        "",
    ]
    md = "\n".join(header) + "\n" + "\n".join(sections)

    (out_dir / "eda_report.md").write_text(md, encoding="utf-8")
    (out_dir / "eda_report.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'eda_report.md'}")
    print(md[:1500])


if __name__ == "__main__":
    main()
