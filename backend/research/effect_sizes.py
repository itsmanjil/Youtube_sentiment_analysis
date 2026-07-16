"""
Compute effect sizes for the CI-vs-baseline comparisons.

Motivation
----------
`results/ci_significance_tests.json` reports McNemar p-values (with
Holm-Bonferroni correction) for pairwise comparisons of every method on the
held-out test set. With n_test = 20,000 even tiny differences reach
p < 0.001, so the p-values alone are not enough to support or refute thesis
claims. Modern reporting (APA 7, Lakens 2013) expects **effect sizes**
alongside significance.

For paired binary outcomes (McNemar) the standard effect sizes are:
  - Cohen's g    = (max(b,c) / (b+c)) - 0.5          # deviation from 50/50
  - Odds ratio   = b / c                             # ratio of discordants
  - Risk diff    = (b - c) / n_test                  # pp-accuracy difference
where b = n01 = "a wrong, b right", c = n10 = "a right, b wrong".

Convention: positive risk diff means method_b beats method_a.

This script reads `results/ci_significance_tests.json`, computes Cohen's g,
odds ratio, and risk difference for every pair, flags the "logreg vs CI"
comparisons specifically, and writes:
  - results/effect_sizes.json       (machine-readable)
  - results/effect_sizes.md         (thesis-ready markdown table)
"""

from __future__ import annotations

import json
import math
from pathlib import Path


INPUT_JSON = Path(
    "/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend/results/ci_significance_tests.json"
)
OUT_JSON = Path(
    "/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend/results/effect_sizes.json"
)
OUT_MD = Path(
    "/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend/results/effect_sizes.md"
)


def cohens_g(b: int, c: int) -> float:
    """Cohen's g for McNemar: deviation of the larger discordant proportion from 0.5.

    Range [0, 0.5]. Cohen (1988) suggests small=0.05, medium=0.15, large=0.25.
    """
    discordant = b + c
    if discordant == 0:
        return 0.0
    return max(b, c) / discordant - 0.5


def odds_ratio(b: int, c: int) -> float:
    if c == 0:
        if b == 0:
            return 1.0
        return float("inf")
    return b / c


def risk_difference(b: int, c: int, n: int) -> float:
    """(b - c) / n: positive means method_b has MORE corrections than method_a.

    Equivalent to the accuracy gain of method_b over method_a in percentage
    points (x100).
    """
    if n == 0:
        return 0.0
    return (b - c) / n


def interpret_g(g: float) -> str:
    ag = abs(g)
    if ag < 0.05:
        return "negligible"
    if ag < 0.15:
        return "small"
    if ag < 0.25:
        return "medium"
    return "large"


def main() -> None:
    with open(INPUT_JSON) as f:
        data = json.load(f)

    n_test = int(data["n_test"])
    pairs = data["pairs"]
    method_f1 = data["method_f1"]

    rows = []
    for p in pairs:
        a, b_name = p["method_a"], p["method_b"]
        b_cell = int(p["n01"])  # a wrong, b right
        c_cell = int(p["n10"])  # a right, b wrong
        g = cohens_g(b_cell, c_cell)
        orr = odds_ratio(b_cell, c_cell)
        rd = risk_difference(b_cell, c_cell, n_test)
        # sign: positive rd means method_b gains over method_a
        rows.append(
            {
                "method_a": a,
                "method_b": b_name,
                "n01": b_cell,
                "n10": c_cell,
                "n_discordant": b_cell + c_cell,
                "p_raw": p["p_raw"],
                "p_adj": p["p_adj"],
                "significant": p["significant"],
                "cohens_g": g,
                "cohens_g_magnitude": interpret_g(g),
                "odds_ratio_b_over_a": orr,
                "risk_diff_b_minus_a_pp": rd * 100,
                "f1_a": method_f1.get(a),
                "f1_b": method_f1.get(b_name),
                "f1_delta_b_minus_a": (
                    None
                    if method_f1.get(a) is None or method_f1.get(b_name) is None
                    else method_f1[b_name] - method_f1[a]
                ),
            }
        )

    out = {
        "metadata": {
            "n_test": n_test,
            "source": str(INPUT_JSON.name),
            "note": (
                "Cohen's g uses Cohen (1988) thresholds: "
                "small=0.05, medium=0.15, large=0.25. "
                "risk_diff_b_minus_a_pp is the accuracy gain of method_b "
                "over method_a on the paired test set, in percentage points."
            ),
        },
        "pairs": rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Markdown: focus on logreg-vs-CI row, then full table
    ci_methods = ["ensemble", "meta_learner", "pso", "nsga2", "neuro_fuzzy", "ts_meta"]
    logreg_rows = []
    for r in rows:
        if r["method_a"] == "logreg" and r["method_b"] in ci_methods:
            logreg_rows.append(r)
        elif r["method_b"] == "logreg" and r["method_a"] in ci_methods:
            # flip sign/direction so the reported comparison is logreg -> CI
            logreg_rows.append(
                {
                    "method_a": "logreg",
                    "method_b": r["method_a"],
                    "n01": r["n10"],
                    "n10": r["n01"],
                    "n_discordant": r["n_discordant"],
                    "p_raw": r["p_raw"],
                    "p_adj": r["p_adj"],
                    "significant": r["significant"],
                    "cohens_g": cohens_g(r["n10"], r["n01"]),
                    "cohens_g_magnitude": interpret_g(cohens_g(r["n10"], r["n01"])),
                    "odds_ratio_b_over_a": (
                        r["n10"] / r["n01"] if r["n01"] else float("inf")
                    ),
                    "risk_diff_b_minus_a_pp": -r["risk_diff_b_minus_a_pp"],
                    "f1_a": method_f1.get("logreg"),
                    "f1_b": method_f1.get(r["method_a"]),
                    "f1_delta_b_minus_a": (
                        method_f1.get(r["method_a"], 0)
                        - method_f1.get("logreg", 0)
                    ),
                }
            )

    lines = []
    lines.append("# Effect Sizes for CI vs Baseline Comparisons\n\n")
    lines.append(
        f"Source: `{INPUT_JSON.name}`  |  n_test = {n_test:,}  |  "
        "paired McNemar contingency cells (b=n01, c=n10)\n\n"
    )
    lines.append(
        "**Why this matters.** With n_test = 20k even a 0.1 pp accuracy gap "
        "achieves p < 0.001, so raw p-values are not informative. Effect sizes "
        "(Cohen's g for McNemar, risk difference in percentage points) quantify "
        "how much the CI methods actually improve on the logreg baseline.\n\n"
    )
    lines.append("## CI methods vs Logistic Regression baseline\n\n")
    lines.append(
        "| Comparison | ΔF1 | Acc gain (pp) | Cohen's g | Magnitude | "
        "Odds ratio (b/c) | p_adj | Sig? |\n"
    )
    lines.append(
        "|---|---:|---:|---:|:---:|---:|---:|:---:|\n"
    )
    # Sort CI rows so the biggest F1 gains come first
    logreg_rows.sort(
        key=lambda r: -(r.get("f1_delta_b_minus_a") or 0.0)
    )
    for r in logreg_rows:
        delta = r.get("f1_delta_b_minus_a") or 0.0
        lines.append(
            f"| logreg → {r['method_b']} | "
            f"{delta:+.4f} | "
            f"{r['risk_diff_b_minus_a_pp']:+.2f} | "
            f"{r['cohens_g']:+.3f} | "
            f"{r['cohens_g_magnitude']} | "
            f"{r['odds_ratio_b_over_a']:.3f} | "
            f"{r['p_adj']:.2e} | "
            f"{'✓' if r['significant'] else '✗'} |\n"
        )

    lines.append("\n## Interpretation\n\n")
    # Find the largest effect size among the CI methods vs logreg
    max_rd = max(logreg_rows, key=lambda r: r["risk_diff_b_minus_a_pp"])
    min_rd = min(logreg_rows, key=lambda r: r["risk_diff_b_minus_a_pp"])
    lines.append(
        f"- Best CI method vs logreg: **{max_rd['method_b']}** with a "
        f"+{max_rd['risk_diff_b_minus_a_pp']:.2f} pp accuracy gain on the "
        f"paired test set and Cohen's g = {max_rd['cohens_g']:+.3f} "
        f"({max_rd['cohens_g_magnitude']}).\n"
    )
    lines.append(
        f"- Worst CI method vs logreg: **{min_rd['method_b']}** with "
        f"{min_rd['risk_diff_b_minus_a_pp']:+.2f} pp and g = "
        f"{min_rd['cohens_g']:+.3f} ({min_rd['cohens_g_magnitude']}).\n"
    )
    lines.append(
        "\n- By Cohen's (1988) conventions, g < 0.05 is negligible and g < 0.15 "
        "is small. Even when the McNemar test is statistically significant (which "
        "is nearly automatic at n=20k), a negligible Cohen's g means the "
        "practical improvement is below what a human examiner would notice.\n"
    )
    lines.append(
        "- This justifies the thesis's reframing: CI contributions should be "
        "defended on **calibration** (ECE reduction), **Pareto trade-offs**, and "
        "the **negative result** that fuzzy / NSGA-II / PSO ensembles do not "
        "beat a tuned logreg on F1 at scale.\n"
    )

    lines.append("\n## Full pairwise table\n\n")
    lines.append(
        "| A | B | n01 (a✗,b✓) | n10 (a✓,b✗) | Cohen's g | Risk diff b-a (pp) | "
        "Odds ratio (b/a) | p_adj | Sig |\n"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|\n"
    )
    for r in rows:
        orr_str = (
            "∞" if math.isinf(r["odds_ratio_b_over_a"])
            else f"{r['odds_ratio_b_over_a']:.3f}"
        )
        lines.append(
            f"| {r['method_a']} | {r['method_b']} | "
            f"{r['n01']} | {r['n10']} | {r['cohens_g']:+.3f} | "
            f"{r['risk_diff_b_minus_a_pp']:+.2f} | {orr_str} | "
            f"{r['p_adj']:.2e} | {'✓' if r['significant'] else '✗'} |\n"
        )

    with open(OUT_MD, "w") as f:
        f.writelines(lines)

    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")
    print()
    print("Summary of logreg-vs-CI comparisons:")
    for r in logreg_rows:
        delta = r.get("f1_delta_b_minus_a") or 0.0
        print(
            f"  logreg -> {r['method_b']:14s}  "
            f"ΔF1={delta:+.4f}  "
            f"pp={r['risk_diff_b_minus_a_pp']:+.2f}  "
            f"g={r['cohens_g']:+.3f}  "
            f"({r['cohens_g_magnitude']})"
        )


if __name__ == "__main__":
    main()
