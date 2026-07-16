#!/usr/bin/env python3
"""
Near-duplicate leakage audit for train/val/test splits.

Exact duplicate removal is already handled by the split builder. This script
adds a lightweight SimHash audit to catch near-duplicate texts that cross split
boundaries, such as repeated spam templates or lightly edited comments.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9]+")
SPLITS = ("train", "val", "test")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_text(text: object) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def shingles(text: str, width: int) -> List[str]:
    tokens = normalize_text(text).split()
    if len(tokens) <= width:
        return [" ".join(tokens)] if tokens else [""]
    return [" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]


def stable_hash64(value: str) -> int:
    # FNV-1a 64-bit: deterministic across Python processes.
    h = 1469598103934665603
    for byte in value.encode("utf-8", errors="ignore"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def simhash64(text: str, shingle_width: int) -> int:
    weights = [0] * 64
    for shingle in shingles(text, shingle_width):
        h = stable_hash64(shingle)
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def load_split(path: Path, split: str, text_column: str, max_rows: int | None) -> List[Dict[str, object]]:
    frame = pd.read_csv(path, keep_default_na=False)
    if text_column not in frame.columns:
        raise ValueError(f"{path} does not contain text column '{text_column}'.")
    if max_rows:
        frame = frame.iloc[:max_rows].copy()
    records = []
    for index, text in enumerate(frame[text_column].astype(str).tolist()):
        normalized = normalize_text(text)
        if not normalized:
            continue
        records.append(
            {
                "split": split,
                "row_index": index,
                "text": text,
                "normalized": normalized,
            }
        )
    return records


def band_keys(signature: int, bands: int, bits_per_band: int) -> Iterable[Tuple[int, int]]:
    mask = (1 << bits_per_band) - 1
    for band in range(bands):
        yield band, (signature >> (band * bits_per_band)) & mask


def audit(records: Sequence[Dict[str, object]], *, shingle_width: int, max_distance: int, max_examples: int) -> Dict[str, object]:
    enriched = []
    for record in records:
        item = dict(record)
        item["simhash"] = simhash64(str(record["normalized"]), shingle_width)
        enriched.append(item)

    bands = 8
    bits_per_band = 8
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, item in enumerate(enriched):
        for key in band_keys(int(item["simhash"]), bands, bits_per_band):
            buckets[key].append(idx)

    seen_pairs = set()
    cross_split_pairs = []
    exact_cross_split_pairs = []
    candidate_pairs = 0

    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i_pos in range(len(bucket)):
            for j_pos in range(i_pos + 1, len(bucket)):
                i = bucket[i_pos]
                j = bucket[j_pos]
                a = enriched[i]
                b = enriched[j]
                if a["split"] == b["split"]:
                    continue
                pair_key = tuple(sorted((i, j)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidate_pairs += 1

                distance = hamming_distance(int(a["simhash"]), int(b["simhash"]))
                if distance <= max_distance:
                    pair = {
                        "split_a": a["split"],
                        "row_a": a["row_index"],
                        "split_b": b["split"],
                        "row_b": b["row_index"],
                        "hamming_distance": distance,
                        "text_a": str(a["text"])[:240],
                        "text_b": str(b["text"])[:240],
                    }
                    if a["normalized"] == b["normalized"]:
                        exact_cross_split_pairs.append(pair)
                    else:
                        cross_split_pairs.append(pair)

    cross_split_pairs.sort(key=lambda item: item["hamming_distance"])
    exact_cross_split_pairs.sort(key=lambda item: item["hamming_distance"])
    return {
        "records_scanned": len(enriched),
        "candidate_pairs_checked": candidate_pairs,
        "near_duplicate_cross_split_count": len(cross_split_pairs),
        "exact_duplicate_cross_split_count": len(exact_cross_split_pairs),
        "max_hamming_distance": max_distance,
        "examples": cross_split_pairs[:max_examples],
        "exact_examples": exact_cross_split_pairs[:max_examples],
    }


def build_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Near-Duplicate Leakage Audit\n",
        f"- Created at: `{payload['created_at']}`",
        f"- Split directory: `{payload['split_dir']}`",
        f"- Text column: `{payload['text_column']}`",
        f"- Records scanned: `{payload['audit']['records_scanned']}`",
        f"- Candidate pairs checked: `{payload['audit']['candidate_pairs_checked']}`",
        f"- Exact cross-split duplicates: `{payload['audit']['exact_duplicate_cross_split_count']}`",
        f"- Near-duplicate cross-split pairs: `{payload['audit']['near_duplicate_cross_split_count']}`",
        f"- Status: `{'PASS' if payload['passed'] else 'REVIEW'}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]

    if payload["audit"]["examples"]:
        lines.extend(["## Near-Duplicate Examples", ""])
        for item in payload["audit"]["examples"]:
            lines.extend(
                [
                    f"- `{item['split_a']}:{item['row_a']}` vs `{item['split_b']}:{item['row_b']}` "
                    f"(Hamming distance {item['hamming_distance']})",
                    f"  - A: {item['text_a']}",
                    f"  - B: {item['text_b']}",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit near-duplicate leakage across train/val/test CSV splits.")
    parser.add_argument("--split_dir", default="data/route_a_benchmark_cpu")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--shingle_width", type=int, default=5)
    parser.add_argument("--max_distance", type=int, default=3)
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument(
        "--fail_on_findings",
        action="store_true",
        help="Exit non-zero when exact or near-duplicate cross-split pairs are found.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parents[2]
    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = backend_root / split_dir

    all_records = []
    split_counts = {}
    for split in SPLITS:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        records = load_split(path, split, args.text_column, args.max_rows_per_split)
        split_counts[split] = len(records)
        all_records.extend(records)

    audit_result = audit(
        all_records,
        shingle_width=args.shingle_width,
        max_distance=args.max_distance,
        max_examples=args.max_examples,
    )
    passed = (
        audit_result["exact_duplicate_cross_split_count"] == 0
        and audit_result["near_duplicate_cross_split_count"] == 0
    )
    interpretation = (
        "No exact or near-duplicate cross-split leakage was found under the configured SimHash threshold."
        if passed
        else (
            "Potential cross-split near duplicates were found. Review the examples, "
            "tighten deduplication, or report the residual risk as a limitation."
        )
    )

    payload = {
        "title": "Near-Duplicate Leakage Audit",
        "created_at": _utcnow(),
        "split_dir": str(split_dir),
        "text_column": args.text_column,
        "shingle_width": args.shingle_width,
        "split_counts": split_counts,
        "audit": audit_result,
        "passed": passed,
        "interpretation": interpretation,
    }

    output_root = backend_root / "results" / "leakage"
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_root / "near_duplicate_audit.json"
    output_md = Path(args.output_md) if args.output_md else output_root / "near_duplicate_audit.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if args.fail_on_findings and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
