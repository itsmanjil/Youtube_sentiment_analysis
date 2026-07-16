"""
Inter-Annotator Agreement (IAA) Metrics
========================================

Provides:
  - percent_agreement      : simple observed agreement across annotators
  - cohen_kappa_pairwise   : Cohen's kappa for every annotator pair
  - fleiss_kappa           : Fleiss' kappa for 3+ annotators
  - krippendorff_alpha     : Krippendorff's alpha (nominal)
  - majority_vote          : reconcile multiple labels via majority vote
  - iaa_report             : full summary dict for JSON/Markdown export

All functions accept an *annotations* matrix of shape (n_items, n_annotators)
where each cell is a string label or None / "" for missing.

Labels used in this project: "Positive", "Neutral", "Negative"

References
----------
- Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

LABELS = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_pair(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b)


def _to_label(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _normalize_annotations(
    matrix: Sequence[Sequence[object]],
) -> List[List[Optional[str]]]:
    return [[_to_label(cell) for cell in row] for row in matrix]


# ---------------------------------------------------------------------------
# Percent Agreement
# ---------------------------------------------------------------------------

def percent_agreement(matrix: Sequence[Sequence[object]]) -> float:
    """
    Fraction of items where all annotators agree (ignoring items with any
    missing label).
    """
    norm = _normalize_annotations(matrix)
    agreed = 0
    total = 0
    for row in norm:
        valid = [v for v in row if v is not None]
        if len(valid) < 2:
            continue
        total += 1
        if len(set(valid)) == 1:
            agreed += 1
    return round(agreed / total, 6) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cohen's Kappa (pairwise)
# ---------------------------------------------------------------------------

def _cohen_kappa(y1: List[Optional[str]], y2: List[Optional[str]]) -> float:
    """Compute Cohen's kappa between two annotators, skipping missing pairs."""
    pairs = [(a, b) for a, b in zip(y1, y2) if _valid_pair(a, b)]
    if len(pairs) < 2:
        return float("nan")

    n = len(pairs)
    a_labels, b_labels = zip(*pairs)

    # observed agreement
    po = sum(a == b for a, b in pairs) / n

    # expected agreement
    all_labels = sorted(set(a_labels) | set(b_labels))
    pe = 0.0
    for lbl in all_labels:
        pa = sum(1 for a in a_labels if a == lbl) / n
        pb = sum(1 for b in b_labels if b == lbl) / n
        pe += pa * pb

    denom = 1.0 - pe
    if abs(denom) < 1e-10:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / denom, 6)


def cohen_kappa_pairwise(
    matrix: Sequence[Sequence[object]],
    annotator_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute pairwise Cohen's kappa for every annotator pair.

    Returns a dict with keys like "ann_0_vs_ann_1" -> kappa value.
    """
    norm = _normalize_annotations(matrix)
    n_annotators = max(len(row) for row in norm) if norm else 0
    names = annotator_names or [f"ann_{i}" for i in range(n_annotators)]

    # Transpose: columns = annotators
    cols: List[List[Optional[str]]] = [[] for _ in range(n_annotators)]
    for row in norm:
        for j in range(n_annotators):
            cols[j].append(row[j] if j < len(row) else None)

    result: Dict[str, float] = {}
    for i, j in combinations(range(n_annotators), 2):
        key = f"{names[i]}_vs_{names[j]}"
        result[key] = _cohen_kappa(cols[i], cols[j])
    return result


# ---------------------------------------------------------------------------
# Fleiss' Kappa (multi-annotator)
# ---------------------------------------------------------------------------

def fleiss_kappa(
    matrix: Sequence[Sequence[object]],
    categories: Optional[List[str]] = None,
) -> float:
    """
    Fleiss' kappa for multi-annotator nominal data.

    Each row is one item; each column is one annotator.
    Missing annotations are allowed but each item must have at least 2.
    """
    norm = _normalize_annotations(matrix)
    if categories is None:
        all_vals = {v for row in norm for v in row if v is not None}
        categories = sorted(all_vals)
    if not categories:
        return float("nan")

    cat_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    # Build n_ij: number of annotators who assigned category j to item i
    n_ij: List[List[int]] = []
    n_i: List[int] = []
    for row in norm:
        counts = [0] * n_cats
        n = 0
        for v in row:
            if v is not None and v in cat_idx:
                counts[cat_idx[v]] += 1
                n += 1
        if n < 2:
            continue
        n_ij.append(counts)
        n_i.append(n)

    N = len(n_ij)
    if N == 0:
        return float("nan")

    # Mean annotations per item (allow variable annotator counts)
    # Fleiss' kappa with variable n_i (Randolph / Conger extension)
    # p_j = proportion of all assignments in category j
    total_assignments = sum(n_i)
    p_j = [
        sum(n_ij[i][j] for i in range(N)) / total_assignments
        for j in range(n_cats)
    ]

    # P_i = fraction of annotator pairs that agree on item i
    P_i = []
    for i in range(N):
        ni = n_i[i]
        if ni < 2:
            P_i.append(0.0)
            continue
        agree_pairs = sum(n_ij[i][j] * (n_ij[i][j] - 1) for j in range(n_cats))
        P_i.append(agree_pairs / (ni * (ni - 1)))

    P_bar = sum(P_i) / N
    P_e = sum(p ** 2 for p in p_j)

    denom = 1.0 - P_e
    if abs(denom) < 1e-10:
        return 1.0 if P_bar >= 1.0 else 0.0
    return round((P_bar - P_e) / denom, 6)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha (nominal)
# ---------------------------------------------------------------------------

def krippendorff_alpha(
    matrix: Sequence[Sequence[object]],
    categories: Optional[List[str]] = None,
) -> float:
    """
    Krippendorff's alpha for nominal data.

    matrix: (n_items, n_annotators) — missing values allowed (None / "")
    categories: if None, inferred from data

    Formula (Krippendorff 2004, nominal metric):
        alpha = 1 - D_o / D_e

    where D_o = observed disagreement (from coincidence matrix)
          D_e = expected disagreement (from marginal counts)
    """
    norm = _normalize_annotations(matrix)
    if categories is None:
        all_vals = {v for row in norm for v in row if v is not None}
        categories = sorted(all_vals)
    if not categories or len(categories) < 2:
        return float("nan")

    cat_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    # Build coincidence matrix e[c][k]
    # For each item, for every ordered pair of annotators (u, v) where u < v,
    # if both assigned labels c and k: increment e[c][k] and e[k][c] by 1/(n_i - 1)
    e = [[0.0] * n_cats for _ in range(n_cats)]

    for row in norm:
        valid = [(cat_idx[v], v) for v in row if v is not None and v in cat_idx]
        ni = len(valid)
        if ni < 2:
            continue
        weight = 1.0 / (ni - 1)
        for a in range(len(valid)):
            for b in range(len(valid)):
                if a == b:
                    continue
                ci = valid[a][0]
                ki = valid[b][0]
                e[ci][ki] += weight

    # n_c = row (or column) marginals
    n_c = [sum(e[c][k] for k in range(n_cats)) for c in range(n_cats)]
    n_total = sum(n_c)

    if n_total < 2:
        return float("nan")

    # D_o = observed disagreement = sum_{c != k} e[c][k] / n_total
    D_o = sum(
        e[c][k]
        for c in range(n_cats)
        for k in range(n_cats)
        if c != k
    ) / n_total

    # D_e = expected disagreement = sum_{c != k} n_c * n_k / (n_total * (n_total - 1))
    D_e = sum(
        n_c[c] * n_c[k]
        for c in range(n_cats)
        for k in range(n_cats)
        if c != k
    ) / (n_total * (n_total - 1))

    if abs(D_e) < 1e-10:
        return 1.0 if D_o < 1e-10 else 0.0

    alpha = 1.0 - D_o / D_e
    return round(alpha, 6)


# ---------------------------------------------------------------------------
# Majority Vote Reconciliation
# ---------------------------------------------------------------------------

def majority_vote(
    labels: Sequence[Optional[str]],
) -> Tuple[Optional[str], bool]:
    """
    Return (majority_label, is_clear_majority).

    - is_clear_majority = True if one label has strictly more votes than all others
    - is_clear_majority = False if there is a tie (item needs adjudication)
    - Returns (None, False) if no valid labels
    """
    valid = [v for v in labels if v]
    if not valid:
        return None, False

    counts = Counter(valid)
    most_common = counts.most_common()

    if len(most_common) == 1:
        return most_common[0][0], True

    top_count = most_common[0][1]
    # Tie if top two counts are equal
    if most_common[1][1] == top_count:
        return None, False  # Tie — needs adjudication

    return most_common[0][0], True


def reconcile_annotations(
    matrix: Sequence[Sequence[object]],
    annotator_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    For each item, compute majority vote and flag disputed items.

    Returns a list of dicts:
      {
        "gold_label": str | None,
        "is_disputed": bool,
        "annotator_labels": {name: label},
        "vote_counts": {label: count},
      }
    """
    norm = _normalize_annotations(matrix)
    n_annotators = max(len(row) for row in norm) if norm else 0
    names = annotator_names or [f"ann_{i}" for i in range(n_annotators)]

    results = []
    for row in norm:
        padded = list(row) + [None] * (n_annotators - len(row))
        ann_map = {names[i]: padded[i] for i in range(n_annotators)}
        gold, clear = majority_vote([v for v in padded if v])
        votes = Counter(v for v in padded if v)
        results.append(
            {
                "gold_label": gold,
                "is_disputed": not clear,
                "annotator_labels": ann_map,
                "vote_counts": dict(votes),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Full IAA Report
# ---------------------------------------------------------------------------

def iaa_report(
    matrix: Sequence[Sequence[object]],
    annotator_names: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> Dict:
    """
    Compute all IAA metrics and return a structured dict.

    matrix: (n_items, n_annotators)
    annotator_names: optional names for each annotator column
    categories: optional label set (inferred if None)
    """
    norm = _normalize_annotations(matrix)
    n_items = len(norm)
    n_annotators = max(len(row) for row in norm) if norm else 0

    if categories is None:
        categories = sorted({v for row in norm for v in row if v is not None})

    # Count fully-annotated items (all annotators provided a label)
    fully_annotated = sum(
        1 for row in norm
        if all(v is not None for v in (list(row) + [None] * (n_annotators - len(row)))[: n_annotators])
    )

    reconciled = reconcile_annotations(norm, annotator_names)
    n_disputed = sum(1 for r in reconciled if r["is_disputed"])

    # Label distribution of gold labels
    gold_labels = [r["gold_label"] for r in reconciled if r["gold_label"]]
    gold_distribution = dict(Counter(gold_labels))

    # Per-annotator label distributions
    names = annotator_names or [f"ann_{i}" for i in range(n_annotators)]
    per_annotator_dist: Dict[str, Dict[str, int]] = {}
    for j, name in enumerate(names):
        col = [row[j] if j < len(row) else None for row in norm]
        per_annotator_dist[name] = dict(Counter(v for v in col if v))

    report: Dict = {
        "n_items": n_items,
        "n_annotators": n_annotators,
        "annotator_names": names,
        "categories": categories,
        "fully_annotated_items": fully_annotated,
        "disputed_items": n_disputed,
        "metrics": {
            "percent_agreement": percent_agreement(norm),
            "krippendorff_alpha": krippendorff_alpha(norm, categories=categories),
            "fleiss_kappa": fleiss_kappa(norm, categories=categories),
            "cohen_kappa_pairwise": cohen_kappa_pairwise(norm, annotator_names=names),
        },
        "gold_label_distribution": gold_distribution,
        "per_annotator_distribution": per_annotator_dist,
        "interpretation": _interpret_alpha(
            krippendorff_alpha(norm, categories=categories)
        ),
    }
    return report


def _interpret_alpha(alpha: float) -> str:
    if alpha != alpha:  # nan
        return "insufficient data"
    if alpha >= 0.80:
        return "strong agreement (alpha >= 0.80)"
    if alpha >= 0.67:
        return "tentative agreement (0.67 <= alpha < 0.80) - Krippendorff's minimum for tentative conclusions"
    if alpha >= 0.40:
        return "moderate agreement (0.40 <= alpha < 0.67) - treat gold labels with caution"
    return "poor agreement (alpha < 0.40) - gold labels unreliable without adjudication"
