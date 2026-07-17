#!/usr/bin/env python3
"""
Run the full Route A benchmark pipeline on a prepared split directory.

Expected split layout:
  <split_dir>/train.csv
  <split_dir>/val.csv
  <split_dir>/test.csv

Example
-------
cd backend
python research/route_a/run_benchmark_pipeline.py \
  --split_dir data/route_a_benchmark_cpu \
  --model_preset deberta_v3 \
  --run_tag route_a_benchmark_cpu
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.model_registry import list_encoder_presets
from src.utils.config import Config


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or f"route_a_{_utc_slug()}"


def _run(command: List[str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=BACKEND_ROOT, check=True)


def _seed_existing_artifact(target_path: Path, source_candidates: List[Path]) -> None:
    if target_path.exists():
        return
    for candidate in source_candidates:
        if candidate.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target_path)
            print(f"Reused existing artifact: {candidate} -> {target_path}", flush=True)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Route A benchmark pipeline.")
    parser.add_argument("--split_dir", required=True, help="Directory containing train.csv / val.csv / test.csv.")
    parser.add_argument(
        "--model_preset",
        default="deberta_v3",
        choices=list_encoder_presets(),
        help="Encoder preset to fine-tune.",
    )
    parser.add_argument(
        "--run_tag",
        default=None,
        help="Artifact tag. Defaults to <model>_<split_dir_name>_<timestamp>.",
    )
    parser.add_argument(
        "--ci_models",
        default=None,
        help="Comma-separated models for the CI stage. Defaults to <model_preset>,logreg,svm.",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsga_pop", type=int, default=16)
    parser.add_argument("--nsga_gen", type=int, default=16)
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
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"
    for path in (train_csv, val_csv, test_csv):
        if not path.exists():
            raise SystemExit(f"Missing required split file: {path}")

    run_tag = _slugify(
        args.run_tag or f"{args.model_preset}_{split_dir.name}_{_utc_slug()}"
    )
    ci_models = args.ci_models or f"{args.model_preset},logreg,svm"

    run_dir = Config.BACKEND_DIR / "results" / "route_a_runs" / run_tag
    prob_cube_dir = run_dir / "prob_cubes"
    ci_dir = run_dir / "ci"
    fuzzy_dir = run_dir / "fuzzy"
    run_dir.mkdir(parents=True, exist_ok=True)
    prob_cube_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = run_dir / f"{args.model_preset}_metrics.json"
    calibration_output = run_dir / f"{args.model_preset}_temperature_scaling.json"
    val_cube_output = prob_cube_dir / f"{run_tag}_val_{ci_models.replace(',', '_')}.npz"
    test_cube_output = prob_cube_dir / f"{run_tag}_test_{ci_models.replace(',', '_')}.npz"
    ci_output = ci_dir
    fuzzy_output = fuzzy_dir
    significance_output = run_dir / "significance.json"
    manifest_path = run_dir / "run_manifest.json"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_tag": run_tag,
        "split_dir": str(split_dir),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "model_preset": args.model_preset,
        "ci_models": ci_models.split(","),
        "artifacts": {
            "metrics": str(metrics_output),
            "calibration": str(calibration_output),
            "val_cube": str(val_cube_output),
            "test_cube": str(test_cube_output),
            "ci_dir": str(ci_output),
            "fuzzy_dir": str(fuzzy_output),
            "significance": str(significance_output),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_python = sys.executable
    root_results_dir = Config.BACKEND_DIR / "results"
    split_slug = split_dir.name

    if not args.skip_train:
        train_command = [
            base_python,
            "research/transformers/train_encoder.py",
            "--model_preset",
            args.model_preset,
            "--train_csv",
            str(train_csv),
            "--val_csv",
            str(val_csv),
            "--test_csv",
            str(test_csv),
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
            "--run_name",
            run_tag,
            "--results_output",
            str(metrics_output),
        ]
        if args.overwrite_output_dir:
            train_command.append("--overwrite_output_dir")
        _run(train_command)
    else:
        _seed_existing_artifact(
            metrics_output,
            [
                root_results_dir / f"{args.model_preset}_{split_slug}_metrics.json",
                root_results_dir / f"{args.model_preset}_metrics.json",
            ],
        )

    if not args.skip_calibration:
        _run(
            [
                base_python,
                "research/transformers/calibrate_encoder.py",
                "--model_preset",
                args.model_preset,
                "--val_csv",
                str(val_csv),
                "--test_csv",
                str(test_csv),
                "--results_output",
                str(calibration_output),
            ]
        )
    else:
        _seed_existing_artifact(
            calibration_output,
            [
                root_results_dir / f"{args.model_preset}_{split_slug}_temperature_scaling.json",
                root_results_dir / f"{args.model_preset}_temperature_scaling.json",
            ],
        )

    _run(
        [
            base_python,
            "research/transformers/export_prob_cube.py",
            "--data_csv",
            str(val_csv),
            "--models",
            ci_models,
            "--output",
            str(val_cube_output),
        ]
    )
    _run(
        [
            base_python,
            "research/transformers/export_prob_cube.py",
            "--data_csv",
            str(test_csv),
            "--models",
            ci_models,
            "--output",
            str(test_cube_output),
        ]
    )

    ci_results_json = ci_output / "multi_objective_ensemble.json"
    if not args.skip_ci:
        _run(
            [
                base_python,
                "research/ci/multi_objective_ensemble.py",
                "--val_cube",
                str(val_cube_output),
                "--test_cube",
                str(test_cube_output),
                "--pop",
                str(args.nsga_pop),
                "--gen",
                str(args.nsga_gen),
                "--confidence",
                str(args.confidence),
                "--output",
                str(ci_output),
            ]
        )

    if not args.skip_fuzzy:
        _run(
            [
                base_python,
                "research/ci/neuro_fuzzy_gate.py",
                "--val_cube",
                str(val_cube_output),
                "--test_cube",
                str(test_cube_output),
                "--output",
                str(fuzzy_output),
            ]
        )

    if not args.skip_significance:
        significance_command = [
            base_python,
            "research/route_a/prob_cube_significance.py",
            "--val_cube",
            str(val_cube_output),
            "--test_cube",
            str(test_cube_output),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--output",
            str(significance_output),
        ]
        if ci_results_json.exists():
            significance_command.extend(["--ci_results", str(ci_results_json)])
        _run(significance_command)

    print("\nRoute A pipeline completed.")
    print(f"Run manifest: {manifest_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
