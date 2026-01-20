"""
End-to-end training pipeline for thesis-grade experiments.

Steps:
  - classic: train LogReg, SVM, TF-IDF models
  - prepare: build train/val/test splits from YouTube videos or labeled CSV
  - hybrid: train Hybrid CNN-BiLSTM-Attention model
  - ensemble: optimize ensemble weights with PSO
  - meta: train stacking meta-learner

Usage:
  python research/run_thesis_pipeline.py --steps all
  python research/run_thesis_pipeline.py --steps classic,ensemble,meta
  python research/run_thesis_pipeline.py --steps prepare,hybrid --video_list videos.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STEP_ORDER = ["classic", "prepare", "hybrid", "ensemble", "meta"]
AVAILABLE_STEPS = set(STEP_ORDER)

CLASSIC_SCRIPT_MAP = {
    "logreg": "train_logreg_youtube.py",
    "svm": "train_svm_youtube.py",
    "tfidf": "train_tfidf_youtube.py",
}


def run_command(args, cwd=BASE_DIR):
    printable = " ".join(str(item) for item in args)
    print(f"\n==> {printable}")
    subprocess.run(args, check=True, cwd=str(cwd))


def parse_steps(value: str | None) -> list[str]:
    if not value or value.strip().lower() in ("all", "*"):
        return STEP_ORDER[:]
    steps = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in steps if item not in AVAILABLE_STEPS]
    if unknown:
        raise ValueError(
            f"Unknown steps: {', '.join(unknown)}. "
            f"Valid steps: {', '.join(STEP_ORDER)}"
        )
    ordered = [step for step in STEP_ORDER if step in steps]
    return ordered


def latest_match(directory: Path, pattern: str) -> Path | None:
    matches = sorted(
        directory.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def resolve_split_paths(args, prepared_dir: Path | None) -> tuple[Path, Path, Path]:
    if args.train_csv and args.val_csv and args.test_csv:
        return Path(args.train_csv), Path(args.val_csv), Path(args.test_csv)

    if prepared_dir:
        train = latest_match(prepared_dir, "train_*.csv")
        val = latest_match(prepared_dir, "val_*.csv")
        test = latest_match(prepared_dir, "test_*.csv")
        if train and val and test:
            return train, val, test

    default_dir = BASE_DIR / "data"
    return default_dir / "train.csv", default_dir / "val.csv", default_dir / "test.csv"


def ensure_exists(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def default_meta_data() -> Path:
    for candidate in ("full_dataset.csv", "train.csv"):
        path = BASE_DIR / "data" / candidate
        if path.exists():
            return path
    return BASE_DIR / "data" / "train.csv"


def default_ensemble_data() -> Path:
    for candidate in ("train.csv", "full_dataset.csv"):
        path = BASE_DIR / "data" / candidate
        if path.exists():
            return path
    return BASE_DIR / "data" / "train.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run thesis-grade training pipeline steps."
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps: classic,prepare,hybrid,ensemble,meta (default: all)",
    )

    # Classic model training
    parser.add_argument(
        "--classic_models",
        default="logreg,svm,tfidf",
        help="Comma-separated classic models to train",
    )
    parser.add_argument(
        "--raw_csv",
        default=None,
        help="CSV for classic models (default: data/raw/youtube_comments_cleaned.csv)",
    )
    parser.add_argument("--classic_text_column", default="CommentText")
    parser.add_argument("--classic_label_column", default="Sentiment")
    parser.add_argument(
        "--classic_output_root",
        default=None,
        help="Optional output root for classic models (creates subdirs per model)",
    )

    # Data preparation
    parser.add_argument("--video", default=None, help="Single YouTube video URL")
    parser.add_argument("--video_list", default=None, help="File with YouTube URLs")
    parser.add_argument("--labeled_csv", default=None, help="Pre-labeled CSV (text,label)")
    parser.add_argument("--label_method", default="auto", choices=["auto", "manual"])
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--merge_mode", default="replace", choices=["replace", "append"])
    parser.add_argument("--max_comments", type=int, default=500)
    parser.add_argument("--prepare_output_dir", default=None)

    # Hybrid DL
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--hybrid_config", default=None)
    parser.add_argument("--hybrid_output_dir", default=None)
    parser.add_argument("--hybrid_experiment_name", default="thesis_hybrid_v1")
    parser.add_argument("--hybrid_device", default=None)

    # Ensemble optimization
    parser.add_argument("--ensemble_data", default=None)
    parser.add_argument("--ensemble_models", default="logreg,svm,tfidf")
    parser.add_argument("--ensemble_particles", type=int, default=30)
    parser.add_argument("--ensemble_iterations", type=int, default=50)
    parser.add_argument("--ensemble_output", default=None)

    # Meta-learner
    parser.add_argument("--meta_data", default=None)
    parser.add_argument("--meta_base_models", default="logreg,svm,tfidf")
    parser.add_argument(
        "--meta_learner",
        default="logistic_regression",
        choices=["logistic_regression", "xgboost", "lightgbm"],
    )
    parser.add_argument("--meta_folds", type=int, default=5)
    parser.add_argument("--meta_output", default=None)

    return parser


def run_classic(args):
    raw_csv = Path(args.raw_csv) if args.raw_csv else (
        BASE_DIR / "data" / "raw" / "youtube_comments_cleaned.csv"
    )
    ensure_exists(raw_csv, "Classic training dataset")

    models = [item.strip().lower() for item in args.classic_models.split(",") if item.strip()]
    for model in models:
        script_name = CLASSIC_SCRIPT_MAP.get(model)
        if not script_name:
            raise ValueError(f"Unknown classic model: {model}")
        script_path = BASE_DIR / script_name
        output_dir = None
        if args.classic_output_root:
            output_dir = Path(args.classic_output_root) / model
        cmd = [PYTHON, str(script_path), "--data", str(raw_csv)]
        cmd += ["--text_column", args.classic_text_column]
        cmd += ["--label_column", args.classic_label_column]
        if output_dir:
            cmd += ["--output_dir", str(output_dir)]
        run_command(cmd)


def run_prepare(args) -> Path:
    if not (args.video or args.video_list or args.labeled_csv):
        raise ValueError(
            "Prepare step requires --video, --video_list, or --labeled_csv."
        )

    output_dir = (
        Path(args.prepare_output_dir)
        if args.prepare_output_dir
        else BASE_DIR / "data" / "youtube_training"
    )

    cmd = [PYTHON, str(BASE_DIR / "prepare_youtube_training_data.py")]
    if args.video:
        cmd += ["--video", args.video]
    if args.video_list:
        cmd += ["--video_list", args.video_list]
    if args.labeled_csv:
        cmd += ["--labeled_csv", args.labeled_csv]
    cmd += ["--label_method", args.label_method]
    cmd += ["--confidence_threshold", str(args.confidence_threshold)]
    cmd += ["--merge_mode", args.merge_mode]
    cmd += ["--max_comments", str(args.max_comments)]
    cmd += ["--output_dir", str(output_dir)]
    run_command(cmd)

    return output_dir


def run_hybrid(args, prepared_dir: Path | None):
    train_csv, val_csv, test_csv = resolve_split_paths(args, prepared_dir)
    ensure_exists(train_csv, "Train CSV")
    ensure_exists(val_csv, "Validation CSV")
    ensure_exists(test_csv, "Test CSV")

    script_path = BASE_DIR / "research" / "train_hybrid_dl.py"
    cmd = [PYTHON, str(script_path)]
    if args.hybrid_config:
        cmd += ["--config", args.hybrid_config]
    cmd += ["--train_csv", str(train_csv)]
    cmd += ["--val_csv", str(val_csv)]
    cmd += ["--test_csv", str(test_csv)]
    if args.hybrid_output_dir:
        cmd += ["--output_dir", args.hybrid_output_dir]
    if args.hybrid_experiment_name:
        cmd += ["--experiment_name", args.hybrid_experiment_name]
    if args.hybrid_device:
        cmd += ["--device", args.hybrid_device]
    run_command(cmd)


def run_ensemble(args):
    data_path = Path(args.ensemble_data) if args.ensemble_data else default_ensemble_data()
    ensure_exists(data_path, "Ensemble dataset")

    output_path = (
        Path(args.ensemble_output)
        if args.ensemble_output
        else BASE_DIR / "results" / "pso_ensemble_weights.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_path = BASE_DIR / "research" / "optimize_ensemble.py"
    cmd = [
        PYTHON,
        str(script_path),
        "--data",
        str(data_path),
        "--models",
        args.ensemble_models,
        "--particles",
        str(args.ensemble_particles),
        "--iterations",
        str(args.ensemble_iterations),
        "--output",
        str(output_path),
    ]
    run_command(cmd)


def run_meta(args):
    data_path = Path(args.meta_data) if args.meta_data else default_meta_data()
    ensure_exists(data_path, "Meta-learner dataset")

    output_path = (
        Path(args.meta_output)
        if args.meta_output
        else BASE_DIR / "models" / "meta_learner.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_path = BASE_DIR / "research" / "meta_learner.py"
    cmd = [
        PYTHON,
        str(script_path),
        "--data",
        str(data_path),
        "--base_models",
        args.meta_base_models,
        "--meta_learner",
        args.meta_learner,
        "--n_folds",
        str(args.meta_folds),
        "--output",
        str(output_path),
    ]
    run_command(cmd)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    prepared_dir = None

    for step in steps:
        if step == "classic":
            run_classic(args)
        elif step == "prepare":
            prepared_dir = run_prepare(args)
        elif step == "hybrid":
            run_hybrid(args, prepared_dir)
        elif step == "ensemble":
            run_ensemble(args)
        elif step == "meta":
            run_meta(args)
        else:
            raise ValueError(f"Unsupported step: {step}")

    print("\nAll requested steps completed.")


if __name__ == "__main__":
    main()
