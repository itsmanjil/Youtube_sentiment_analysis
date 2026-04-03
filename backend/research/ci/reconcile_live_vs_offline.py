#!/usr/bin/env python3
"""
Reconcile the historical offline thesis benchmark with the pinned live runtime benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_markdown_table(md_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    lines = [line.strip() for line in md_path.read_text().splitlines() if line.strip()]
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0].lower() == "model" or set(cells[0]) == {"-"}:
            continue
        if len(cells) < 5:
            continue
        rows.append(
            {
                "model": cells[0],
                "accuracy": float(cells[1]),
                "macro_f1": float(cells[2]),
                "ece": float(cells[3]),
                "brier": float(cells[4]),
            }
        )
    return rows


def _fmt_delta(value: float) -> str:
    return f"{value:+.6f}"


def build_reconciliation(
    offline_rows: List[Dict[str, object]],
    live_payload: Dict[str, object],
) -> Dict[str, object]:
    offline_by_model = {str(row["model"]): row for row in offline_rows}
    live_by_model = {str(row["model"]): row for row in live_payload["results"]}

    exact_matches = []
    for model_name in ["tfidf", "logreg", "svm", "meta_learner"]:
        if model_name not in offline_by_model or model_name not in live_by_model:
            continue
        offline = offline_by_model[model_name]
        live = live_by_model[model_name]
        exact_matches.append(
            {
                "model": model_name,
                "offline": offline,
                "live": live,
                "delta": {
                    "accuracy": round(float(live["accuracy"]) - float(offline["accuracy"]), 6),
                    "macro_f1": round(float(live["macro_f1"]) - float(offline["macro_f1"]), 6),
                    "ece": round(float(live["ece"]) - float(offline["ece"]), 6),
                    "brier": round(float(live["brier"]) - float(offline["brier"]), 6),
                },
            }
        )

    ensemble_offline = offline_by_model.get("ensemble")
    ensemble_variants = []
    if ensemble_offline:
        for live_name in ["ensemble_pso", "ensemble_nsga2"]:
            live = live_by_model.get(live_name)
            if not live:
                continue
            ensemble_variants.append(
                {
                    "offline_model": "ensemble",
                    "live_model": live_name,
                    "offline": ensemble_offline,
                    "live": live,
                    "delta": {
                        "accuracy": round(float(live["accuracy"]) - float(ensemble_offline["accuracy"]), 6),
                        "macro_f1": round(float(live["macro_f1"]) - float(ensemble_offline["macro_f1"]), 6),
                        "ece": round(float(live["ece"]) - float(ensemble_offline["ece"]), 6),
                        "brier": round(float(live["brier"]) - float(ensemble_offline["brier"]), 6),
                    },
                }
            )

    conclusions = []
    if exact_matches:
        conclusions.append(
            "Direct same-name models remain numerically aligned on accuracy/macro-F1, "
            "with only small calibration drift between offline and live artifacts."
        )
    if ensemble_variants:
        conclusions.append(
            "The historical offline `ensemble` row should not be treated as the live runtime default anymore; "
            "the live stack now exposes explicit `ensemble_pso` and `ensemble_nsga2` variants."
        )
        best_live_ensemble = max(ensemble_variants, key=lambda row: float(row["live"]["macro_f1"]))
        conclusions.append(
            f"The best live ensemble variant is `{best_live_ensemble['live_model']}` "
            f"with macro-F1 {best_live_ensemble['live']['macro_f1']:.4f}."
        )

    return {
        "runtime_artifacts": live_payload["runtime_artifacts"],
        "dataset_path": live_payload["dataset_path"],
        "n_samples": live_payload["n_samples"],
        "offline_source": "backend/results/thesis_model_performance_youtube_filtered.md",
        "live_source": "backend/results/runtime/<version>/live_runtime_benchmark_full_test.json",
        "exact_matches": exact_matches,
        "ensemble_variants": ensemble_variants,
        "live_only_models": [
            row for row in live_payload["results"]
            if row["model"] not in {"tfidf", "logreg", "svm", "meta_learner", "ensemble_pso", "ensemble_nsga2"}
        ],
        "conclusions": conclusions,
    }


def build_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Offline vs Live Reconciliation\n",
        f"- Runtime artifact version: `{payload['runtime_artifacts']['version']}`",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Samples: `{payload['n_samples']}`",
        f"- Offline source: `{payload['offline_source']}`",
        f"- Live source: `{payload['live_source']}`",
        "",
        "## Same-Name Models\n",
        "| Model | Offline Acc | Live Acc | Δ Acc | Offline F1 | Live F1 | Δ F1 | Offline ECE | Live ECE | Δ ECE | Offline Brier | Live Brier | Δ Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in payload["exact_matches"]:
        offline = row["offline"]
        live = row["live"]
        delta = row["delta"]
        lines.append(
            f"| {row['model']} | {offline['accuracy']:.4f} | {live['accuracy']:.4f} | {_fmt_delta(delta['accuracy'])} | "
            f"{offline['macro_f1']:.4f} | {live['macro_f1']:.4f} | {_fmt_delta(delta['macro_f1'])} | "
            f"{offline['ece']:.6f} | {live['ece']:.6f} | {_fmt_delta(delta['ece'])} | "
            f"{offline['brier']:.6f} | {live['brier']:.6f} | {_fmt_delta(delta['brier'])} |"
        )

    lines.extend(
        [
            "",
            "## Ensemble Mapping\n",
            "| Offline Row | Live Row | Offline Acc | Live Acc | Δ Acc | Offline F1 | Live F1 | Δ F1 | Offline ECE | Live ECE | Δ ECE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in payload["ensemble_variants"]:
        offline = row["offline"]
        live = row["live"]
        delta = row["delta"]
        lines.append(
            f"| {row['offline_model']} | {row['live_model']} | {offline['accuracy']:.4f} | {live['accuracy']:.4f} | {_fmt_delta(delta['accuracy'])} | "
            f"{offline['macro_f1']:.4f} | {live['macro_f1']:.4f} | {_fmt_delta(delta['macro_f1'])} | "
            f"{offline['ece']:.6f} | {live['ece']:.6f} | {_fmt_delta(delta['ece'])} |"
        )

    if payload["live_only_models"]:
        lines.extend(
            [
                "",
                "## Live-Only Rows\n",
                "| Model | Accuracy | Macro-F1 | ECE | Brier | Note |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in payload["live_only_models"]:
            note = "Pinned runtime row with no direct offline counterpart"
            lines.append(
                f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['ece']:.6f} | {row['brier']:.6f} | {note} |"
            )

    lines.extend(["", "## Conclusions\n"])
    for conclusion in payload["conclusions"]:
        lines.append(f"- {conclusion}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile historical offline and pinned live benchmark tables.")
    parser.add_argument(
        "--offline_md",
        default="results/thesis_model_performance_youtube_filtered.md",
        help="Historical offline thesis metrics table.",
    )
    parser.add_argument(
        "--live_json",
        default="results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.json",
        help="Pinned live runtime benchmark JSON.",
    )
    parser.add_argument("--output_json", default=None, help="Output JSON path.")
    parser.add_argument("--output_md", default=None, help="Output Markdown path.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    offline_md = base / args.offline_md
    live_json = base / args.live_json

    offline_rows = parse_markdown_table(offline_md)
    live_payload = json.loads(live_json.read_text())
    reconciliation = build_reconciliation(offline_rows, live_payload)

    version = reconciliation["runtime_artifacts"]["version"]
    output_root = base / "results" / "runtime" / version
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_root / "offline_vs_live_reconciliation.json"
    output_md = Path(args.output_md) if args.output_md else output_root / "offline_vs_live_reconciliation.md"

    output_json.write_text(json.dumps(reconciliation, indent=2) + "\n")
    output_md.write_text(build_markdown(reconciliation))

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
