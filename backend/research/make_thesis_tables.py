#!/usr/bin/env python3
"""
Generate thesis-ready tables from saved experiment metrics JSON files.

This script is intentionally lightweight (stdlib only) so it can be run in the
same environment as the rest of the backend tools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODELS = ["tfidf", "logreg", "svm", "ensemble", "meta_learner"]


def _backend_dir() -> Path:
    # .../backend/research/make_thesis_tables.py -> parents[1] == backend
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return ""


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _latex_table(
    caption: str,
    label: str,
    headers: List[str],
    rows: List[List[str]],
) -> str:
    cols = "l" + ("c" * (len(headers) - 1))
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join([rf"\textbf{{{h}}}" for h in headers]) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _extract_per_class_f1(report: Dict[str, Any], label: str) -> Optional[float]:
    entry = report.get(label)
    if not isinstance(entry, dict):
        return None
    value = entry.get("f1-score")
    try:
        return float(value)
    except Exception:
        return None


def _load_split_meta(path: Path) -> Dict[str, Any]:
    meta = _read_json(path)
    split = meta.get("split") or {}
    rows = split.get("rows") or {}
    yt = meta.get("youtube_preprocess") or {}
    return {
        "created_at": meta.get("created_at"),
        "rows": {
            "train": int(rows.get("train") or 0),
            "val": int(rows.get("val") or 0),
            "test": int(rows.get("test") or 0),
        },
        "youtube_preprocess": {
            "enabled": bool(yt.get("enabled")) if "enabled" in yt else None,
            "filter_spam": bool(yt.get("filter_spam")) if "filter_spam" in yt else None,
            "filter_language": bool(yt.get("filter_language")) if "filter_language" in yt else None,
            "filter_stats": yt.get("filter_stats") if isinstance(yt.get("filter_stats"), dict) else None,
        },
    }


def build_model_table(metrics: Dict[str, Any], models: List[str]) -> Dict[str, str]:
    headers = ["Model", "Accuracy", "Macro-F1", "ECE", "Brier"]
    rows: List[List[str]] = []
    for model in models:
        if model not in metrics:
            continue
        entry = metrics.get(model) or {}
        calib = entry.get("calibration") or {}
        rows.append(
            [
                model,
                _fmt(entry.get("accuracy"), 4),
                _fmt(entry.get("macro_f1"), 4),
                _fmt(calib.get("ece"), 6),
                _fmt(calib.get("brier"), 6),
            ]
        )

    md = _md_table(headers, rows)
    tex = _latex_table(
        caption="Model performance on the YouTube filtered test set.",
        label="tab:youtube_filtered_model_performance",
        headers=headers,
        rows=rows,
    )
    return {"md": md, "tex": tex}


def build_per_class_f1_table(metrics: Dict[str, Any], models: List[str]) -> Dict[str, str]:
    headers = ["Model", "Negative", "Neutral", "Positive"]
    rows: List[List[str]] = []
    for model in models:
        if model not in metrics:
            continue
        entry = metrics.get(model) or {}
        report = entry.get("report") or {}
        rows.append(
            [
                model,
                _fmt(_extract_per_class_f1(report, "Negative"), 4),
                _fmt(_extract_per_class_f1(report, "Neutral"), 4),
                _fmt(_extract_per_class_f1(report, "Positive"), 4),
            ]
        )

    md = _md_table(headers, rows)
    tex = _latex_table(
        caption="Per-class F1 on the YouTube filtered test set.",
        label="tab:youtube_filtered_per_class_f1",
        headers=headers,
        rows=rows,
    )
    return {"md": md, "tex": tex}


def build_preprocess_ablation_table(
    variants: List[Dict[str, Any]],
    models: List[str],
) -> Dict[str, str]:
    headers = [
        "Dataset",
        "YouTubePre",
        "Spam",
        "Lang",
        "Train",
        "Val",
        "Test",
        "LogReg F1",
        "Meta F1",
    ]
    rows: List[List[str]] = []

    for var in variants:
        name = str(var["name"])
        meta = var["split_meta"]
        yt = meta.get("youtube_preprocess") or {}
        rows_meta = meta.get("rows") or {}
        metrics = var["metrics"]

        def f1(model: str) -> str:
            if model not in metrics:
                return ""
            return _fmt((metrics.get(model) or {}).get("macro_f1"), 4)

        rows.append(
            [
                name,
                str(yt.get("enabled") if yt.get("enabled") is not None else ""),
                str(yt.get("filter_spam") if yt.get("filter_spam") is not None else ""),
                str(yt.get("filter_language") if yt.get("filter_language") is not None else ""),
                str(int(rows_meta.get("train") or 0)),
                str(int(rows_meta.get("val") or 0)),
                str(int(rows_meta.get("test") or 0)),
                f1("logreg"),
                f1("meta_learner"),
            ]
        )

    md = _md_table(headers, rows)
    tex = _latex_table(
        caption="Preprocessing and filtering ablation (macro-F1).",
        label="tab:preprocess_ablation",
        headers=headers,
        rows=rows,
    )
    return {"md": md, "tex": tex}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready markdown/LaTeX tables from metrics JSON files."
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: backend/results).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models to include (default: tfidf,logreg,svm,ensemble,meta_learner).",
    )
    args = parser.parse_args()

    backend_dir = _backend_dir()
    out_dir = Path(args.out_dir) if args.out_dir else (backend_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]

    # Primary (final) dataset for the thesis table.
    filtered_metrics_path = out_dir / "leakfree_youtube_filtered_test_metrics.json"
    if not filtered_metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {filtered_metrics_path}")

    filtered_metrics = _read_json(filtered_metrics_path)

    perf = build_model_table(filtered_metrics, models=models)
    (out_dir / "thesis_model_performance_youtube_filtered.md").write_text(
        perf["md"], encoding="utf-8"
    )
    (out_dir / "thesis_model_performance_youtube_filtered.tex").write_text(
        perf["tex"], encoding="utf-8"
    )

    per_class = build_per_class_f1_table(filtered_metrics, models=models)
    (out_dir / "thesis_per_class_f1_youtube_filtered.md").write_text(
        per_class["md"], encoding="utf-8"
    )
    (out_dir / "thesis_per_class_f1_youtube_filtered.tex").write_text(
        per_class["tex"], encoding="utf-8"
    )

    # Ablation: raw vs youtube_clean vs youtube_filtered (if available).
    variants: List[Dict[str, Any]] = []
    candidate_variants = [
        (
            "raw",
            out_dir / "leakfree_raw_test_metrics.json",
            backend_dir / "data" / "split_metadata_raw.json",
        ),
        (
            "youtube_clean",
            out_dir / "leakfree_youtube_clean_test_metrics.json",
            backend_dir / "data" / "split_metadata_youtube_clean.json",
        ),
        (
            "youtube_filtered",
            out_dir / "leakfree_youtube_filtered_test_metrics.json",
            backend_dir / "data" / "split_metadata_youtube_filtered.json",
        ),
    ]

    for name, metrics_path, meta_path in candidate_variants:
        if metrics_path.exists() and meta_path.exists():
            variants.append(
                {
                    "name": name,
                    "metrics": _read_json(metrics_path),
                    "split_meta": _load_split_meta(meta_path),
                }
            )

    if variants:
        ablation = build_preprocess_ablation_table(variants, models=models)
        (out_dir / "thesis_preprocess_ablation.md").write_text(
            ablation["md"], encoding="utf-8"
        )
        (out_dir / "thesis_preprocess_ablation.tex").write_text(
            ablation["tex"], encoding="utf-8"
        )

    print(f"Wrote tables to: {out_dir}")


if __name__ == "__main__":
    main()

