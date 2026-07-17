#!/usr/bin/env python3
"""
Run a Route A benchmark sweep across multiple encoder presets and summarize the outcomes.

This is the practical entry point for the next thesis milestone:
1. train multiple strong encoders on the same transformer-profile split,
2. run the identical CI workflow for each encoder,
3. compare encoder strength, CI weight usage, and significance results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.model_registry import list_encoder_presets
from src.utils.config import Config


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run(command: List[str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=BACKEND_ROOT, check=True)


def _parse_models(raw_value: str) -> List[str]:
    models = [item.strip().lower().replace("-", "_") for item in str(raw_value).split(",") if item.strip()]
    if not models:
        raise ValueError("At least one encoder preset must be provided.")
    unknown = [model for model in models if model not in list_encoder_presets()]
    if unknown:
        raise ValueError(
            f"Unknown encoder presets: {', '.join(unknown)}. "
            f"Available presets: {', '.join(list_encoder_presets())}"
        )
    return models


def _find_mcnemar_row(rows: List[Dict[str, object]], model_a: str, model_b: str) -> Dict[str, object] | None:
    for row in rows:
        if row.get("model_a") == model_a and row.get("model_b") == model_b:
            return row
        if row.get("model_a") == model_b and row.get("model_b") == model_a:
            swapped = dict(row)
            swapped["model_a"] = model_a
            swapped["model_b"] = model_b
            swapped["n01"] = row.get("n10")
            swapped["n10"] = row.get("n01")
            return swapped
    return None


def _best_classical_model(metrics: Dict[str, Dict[str, float]], candidates: List[str]) -> str | None:
    available = [name for name in candidates if name in metrics]
    if not available:
        return None
    return max(available, key=lambda name: metrics[name].get("macro_f1", float("-inf")))


def _collect_run_summary(run_dir: Path, model_name: str, classical_models: List[str]) -> Dict[str, object]:
    metrics_path = run_dir / f"{model_name}_metrics.json"
    calibration_path = run_dir / f"{model_name}_temperature_scaling.json"
    ci_path = run_dir / "ci" / "multi_objective_ensemble.json"
    fuzzy_path = run_dir / "fuzzy" / "neuro_fuzzy_gate.json"
    significance_path = run_dir / "significance.json"

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
    ci_payload = json.loads(ci_path.read_text(encoding="utf-8"))
    fuzzy_payload = json.loads(fuzzy_path.read_text(encoding="utf-8"))
    significance_payload = json.loads(significance_path.read_text(encoding="utf-8"))

    all_metrics = significance_payload["metrics"]
    best_classical = _best_classical_model(all_metrics, classical_models)
    encoder_vs_best = _find_mcnemar_row(significance_payload["mcnemar"], model_name, best_classical) if best_classical else None
    fuzzy_vs_best = _find_mcnemar_row(significance_payload["mcnemar"], "neuro_fuzzy", best_classical) if best_classical else None

    encoder_weight = float(ci_payload["knee_point"]["weights"].get(model_name, 0.0))

    return {
        "model": model_name,
        "run_dir": str(run_dir),
        "encoder_test": {
            "macro_f1": float(
                metrics_payload["metrics"]["test"]["test_macro_f1"]
                if metrics_payload
                else all_metrics[model_name]["macro_f1"]
            ),
            "accuracy": float(
                metrics_payload["metrics"]["test"]["test_accuracy"]
                if metrics_payload
                else all_metrics[model_name]["accuracy"]
            ),
        },
        "encoder_calibration": {
            "ece_before": (
                float(calibration_payload["metrics"]["test_before"]["ece"])
                if calibration_payload
                else None
            ),
            "ece_after": (
                float(calibration_payload["metrics"]["test_after"]["ece"])
                if calibration_payload
                else None
            ),
            "temperature": float(calibration_payload["temperature"]) if calibration_payload else None,
        },
        "best_classical_model": best_classical,
        "best_classical_metrics": all_metrics.get(best_classical) if best_classical else None,
        "ci_knee": {
            "macro_f1": float(ci_payload["knee_point"]["test"]["macro_f1"]),
            "ece": float(ci_payload["knee_point"]["test"]["ece"]),
            "coverage": float(ci_payload["knee_point"]["test"]["coverage"]),
            "encoder_weight": encoder_weight,
        },
        "neuro_fuzzy_test": {
            "macro_f1": float(fuzzy_payload["test_metrics"]["macro_f1"]),
            "accuracy": float(fuzzy_payload["test_metrics"]["accuracy"]),
            "ece": float(fuzzy_payload["test_metrics"]["ece"]),
        },
        "significance": {
            "encoder_vs_best_classical": encoder_vs_best,
            "neuro_fuzzy_vs_best_classical": fuzzy_vs_best,
        },
    }


def _write_summary_markdown(output_path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "| Encoder | Enc F1 | Enc Acc | Enc ECE→Cal | Best Classical | CI F1 | CI ECE | CI Coverage | Enc Weight | Fuzzy F1 | Fuzzy ECE | Encoder vs Classical | Fuzzy vs Classical |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        encoder_sig = row["significance"]["encoder_vs_best_classical"]
        fuzzy_sig = row["significance"]["neuro_fuzzy_vs_best_classical"]
        encoder_sig_text = "na"
        if encoder_sig:
            encoder_sig_text = "sig" if encoder_sig.get("significant") else "ns"
        fuzzy_sig_text = "na"
        if fuzzy_sig:
            fuzzy_sig_text = "sig" if fuzzy_sig.get("significant") else "ns"
        ece_before = row["encoder_calibration"]["ece_before"]
        ece_after = row["encoder_calibration"]["ece_after"]
        ece_transition = (
            f"{ece_before:.4f}→{ece_after:.4f}"
            if ece_before is not None and ece_after is not None
            else "na"
        )

        lines.append(
            "| {model} | {enc_f1:.4f} | {enc_acc:.4f} | {ece_transition} | {best_classical} | "
            "{ci_f1:.4f} | {ci_ece:.4f} | {ci_cov:.4f} | {enc_weight:.4f} | {fuzzy_f1:.4f} | {fuzzy_ece:.4f} | "
            "{encoder_sig} | {fuzzy_sig} |".format(
                model=row["model"],
                enc_f1=row["encoder_test"]["macro_f1"],
                enc_acc=row["encoder_test"]["accuracy"],
                ece_transition=ece_transition,
                best_classical=row["best_classical_model"] or "na",
                ci_f1=row["ci_knee"]["macro_f1"],
                ci_ece=row["ci_knee"]["ece"],
                ci_cov=row["ci_knee"]["coverage"],
                enc_weight=row["ci_knee"]["encoder_weight"],
                fuzzy_f1=row["neuro_fuzzy_test"]["macro_f1"],
                fuzzy_ece=row["neuro_fuzzy_test"]["ece"],
                encoder_sig=encoder_sig_text,
                fuzzy_sig=fuzzy_sig_text,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Route A encoder sweep and summarize the results.")
    parser.add_argument("--split_dir", required=True, help="Prepared split directory with train.csv / val.csv / test.csv.")
    parser.add_argument(
        "--models",
        default="deberta_v3,modernbert",
        help="Comma-separated encoder presets to benchmark on the same split.",
    )
    parser.add_argument(
        "--classical_models",
        default="logreg,svm",
        help="Comma-separated classical models to combine with each encoder in the CI stage.",
    )
    parser.add_argument(
        "--run_prefix",
        default=None,
        help="Prefix for generated run tags. Defaults to <split_dir_name>_sweep.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsga_pop", type=int, default=32)
    parser.add_argument("--nsga_gen", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--skip_ci", action="store_true")
    parser.add_argument("--skip_fuzzy", action="store_true")
    parser.add_argument("--skip_significance", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = (BACKEND_ROOT / split_dir).resolve()
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")

    models = _parse_models(args.models)
    classical_models = [item.strip().lower().replace("-", "_") for item in args.classical_models.split(",") if item.strip()]
    run_prefix = args.run_prefix or f"{split_dir.name}_sweep"

    sweep_dir = Config.BACKEND_DIR / "results" / "route_a_sweeps" / run_prefix
    sweep_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    base_python = sys.executable

    for model_name in models:
        run_tag = f"{run_prefix}_{model_name}"
        ci_models = ",".join([model_name, *classical_models])
        command = [
            base_python,
            "research/route_a/run_benchmark_pipeline.py",
            "--split_dir",
            str(split_dir),
            "--model_preset",
            model_name,
            "--run_tag",
            run_tag,
            "--ci_models",
            ci_models,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--max_length",
            str(args.max_length),
            "--learning_rate",
            str(args.learning_rate),
            "--early_stopping_patience",
            str(args.early_stopping_patience),
            "--logging_steps",
            str(args.logging_steps),
            "--seed",
            str(args.seed),
            "--nsga_pop",
            str(args.nsga_pop),
            "--nsga_gen",
            str(args.nsga_gen),
            "--confidence",
            str(args.confidence),
            "--bootstrap",
            str(args.bootstrap),
        ]
        if args.skip_train:
            command.append("--skip_train")
        if args.skip_calibration:
            command.append("--skip_calibration")
        if args.skip_ci:
            command.append("--skip_ci")
        if args.skip_fuzzy:
            command.append("--skip_fuzzy")
        if args.skip_significance:
            command.append("--skip_significance")
        if args.overwrite_output_dir:
            command.append("--overwrite_output_dir")

        _run(command)
        run_dir = Config.BACKEND_DIR / "results" / "route_a_runs" / run_tag
        run_rows.append(_collect_run_summary(run_dir, model_name, classical_models))

    summary_payload = {
        "created_at": _utcnow(),
        "split_dir": str(split_dir),
        "models": models,
        "classical_models": classical_models,
        "run_prefix": run_prefix,
        "runs": run_rows,
    }
    summary_json = sweep_dir / "sweep_summary.json"
    summary_md = sweep_dir / "sweep_summary.md"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md, run_rows)

    print(f"\nSaved JSON → {summary_json}")
    print(f"Saved Markdown → {summary_md}")


if __name__ == "__main__":
    main()
