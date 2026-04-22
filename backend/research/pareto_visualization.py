"""
Pareto front visualization for the NSGA-II ensemble-weight optimization.

Reads `results/multi_objective_ensemble.json` and writes three scatter plots
(F1 vs ECE, F1 vs Coverage, ECE vs Coverage) as standalone SVG files plus a
combined HTML viewer. SVG is used so the viewer has no matplotlib dependency
and the figures can be embedded directly in LaTeX via `\\includegraphics`.

Outputs
-------
  - results/pareto_f1_vs_ece.svg
  - results/pareto_f1_vs_coverage.svg
  - results/pareto_ece_vs_coverage.svg
  - results/pareto_visualization.html   (combined viewer, knee highlighted)
"""

from __future__ import annotations

import json
from pathlib import Path


BACKEND = Path(
    "/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend"
)
IN_JSON = BACKEND / "results" / "multi_objective_ensemble.json"
OUT_DIR = BACKEND / "results"

# SVG canvas
W, H = 640, 480
PAD_L, PAD_R = 80, 40
PAD_T, PAD_B = 50, 70


def _rescale(v, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def render_scatter(
    points: list[tuple[float, float]],
    knee_idx: int,
    title: str,
    xlabel: str,
    ylabel: str,
    invert_x: bool = False,
    invert_y: bool = False,
) -> str:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    # Pad the axes 5%
    xpad = (x_hi - x_lo) * 0.05 or 1e-4
    ypad = (y_hi - y_lo) * 0.05 or 1e-4
    x_lo -= xpad
    x_hi += xpad
    y_lo -= ypad
    y_hi += ypad

    def mx(v: float) -> float:
        if invert_x:
            return _rescale(v, x_lo, x_hi, W - PAD_R, PAD_L)
        return _rescale(v, x_lo, x_hi, PAD_L, W - PAD_R)

    def my(v: float) -> float:
        # SVG y grows downward, so the "higher" value maps to smaller y
        if invert_y:
            return _rescale(v, y_lo, y_hi, PAD_T, H - PAD_B)
        return _rescale(v, y_lo, y_hi, H - PAD_B, PAD_T)

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="13">'
    )
    # Background
    svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # Title
    svg.append(
        f'<text x="{W/2}" y="{PAD_T-20}" text-anchor="middle" '
        f'font-size="18" font-weight="bold">{title}</text>'
    )
    # Axes box
    svg.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{W-PAD_L-PAD_R}" '
        f'height="{H-PAD_T-PAD_B}" fill="none" stroke="#333" stroke-width="1"/>'
    )
    # Gridlines (5 on each)
    for i in range(1, 5):
        x = PAD_L + i * (W - PAD_L - PAD_R) / 5
        svg.append(
            f'<line x1="{x}" y1="{PAD_T}" x2="{x}" y2="{H-PAD_B}" '
            'stroke="#eee" stroke-width="1"/>'
        )
        y = PAD_T + i * (H - PAD_T - PAD_B) / 5
        svg.append(
            f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" '
            'stroke="#eee" stroke-width="1"/>'
        )
    # Axis tick labels (5 per axis)
    for i in range(6):
        xv = x_lo + i * (x_hi - x_lo) / 5
        x_px = PAD_L + i * (W - PAD_L - PAD_R) / 5
        svg.append(
            f'<text x="{x_px}" y="{H-PAD_B+18}" text-anchor="middle" '
            f'fill="#444">{xv:.4g}</text>'
        )
        yv = y_hi - i * (y_hi - y_lo) / 5
        y_px = PAD_T + i * (H - PAD_T - PAD_B) / 5
        svg.append(
            f'<text x="{PAD_L-8}" y="{y_px+4}" text-anchor="end" '
            f'fill="#444">{yv:.4g}</text>'
        )

    # Axis labels
    svg.append(
        f'<text x="{W/2}" y="{H-20}" text-anchor="middle" '
        f'font-size="14" font-weight="600">{xlabel}</text>'
    )
    svg.append(
        f'<text x="20" y="{H/2}" text-anchor="middle" font-size="14" '
        f'font-weight="600" transform="rotate(-90 20 {H/2})">{ylabel}</text>'
    )

    # Points
    for i, (xv, yv) in enumerate(points):
        x = mx(xv)
        y = my(yv)
        if i == knee_idx:
            svg.append(
                f'<circle cx="{x}" cy="{y}" r="8" fill="#e74c3c" '
                'stroke="#c0392b" stroke-width="2"/>'
            )
            svg.append(
                f'<text x="{x+12}" y="{y-10}" fill="#c0392b" '
                'font-weight="bold">knee</text>'
            )
        else:
            svg.append(
                f'<circle cx="{x}" cy="{y}" r="4" fill="#3498db" '
                'fill-opacity="0.72" stroke="#2980b9" stroke-width="1"/>'
            )
    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    with open(IN_JSON) as f:
        data = json.load(f)

    points = data["pareto_front"]
    knee_idx = int(data["knee_point"]["index"])
    f1 = [p["macro_f1"] for p in points]
    ece = [p["ece"] for p in points]
    cov = [p["coverage"] for p in points]

    # Figure 1: F1 vs ECE (we want: high F1, low ECE → top-left)
    svg1 = render_scatter(
        list(zip(ece, f1)),
        knee_idx,
        "NSGA-II Pareto Front: Macro-F1 vs ECE",
        "Expected Calibration Error (ECE) — lower is better",
        "Macro-F1 — higher is better",
    )
    (OUT_DIR / "pareto_f1_vs_ece.svg").write_text(svg1)

    # Figure 2: F1 vs Coverage
    svg2 = render_scatter(
        list(zip(cov, f1)),
        knee_idx,
        "NSGA-II Pareto Front: Macro-F1 vs Coverage",
        "Coverage (fraction of predictions above τ=0.7)",
        "Macro-F1 — higher is better",
    )
    (OUT_DIR / "pareto_f1_vs_coverage.svg").write_text(svg2)

    # Figure 3: ECE vs Coverage
    svg3 = render_scatter(
        list(zip(cov, ece)),
        knee_idx,
        "NSGA-II Pareto Front: ECE vs Coverage",
        "Coverage (fraction of predictions above τ=0.7)",
        "Expected Calibration Error (ECE) — lower is better",
        invert_y=True,
    )
    (OUT_DIR / "pareto_ece_vs_coverage.svg").write_text(svg3)

    knee = data["knee_point"]
    knee_txt = (
        f"Knee point (index {knee_idx}): "
        f"weights logreg={knee['weights']['logreg']:.3f}, "
        f"svm={knee['weights']['svm']:.3f}, "
        f"tfidf={knee['weights']['tfidf']:.3f}; "
        f"val F1={knee['val']['macro_f1']:.4f}, ECE={knee['val']['ece']:.4f}, "
        f"coverage={knee['val']['coverage']:.4f}; "
        f"test F1={knee['test']['macro_f1']:.4f}, ECE={knee['test']['ece']:.4f}."
    )

    # Combined HTML viewer
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NSGA-II Pareto Front — Ensemble Weights</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; max-width: 1080px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ font-size: 24px; }}
  h2 {{ font-size: 18px; margin-top: 2em; }}
  figure {{ margin: 1em 0 2em 0; }}
  figcaption {{ color: #555; margin-top: 0.5em; }}
  .knee {{ background: #fff3e0; border-left: 4px solid #e67e22; padding: 0.8em 1em; margin: 1.5em 0; }}
  code {{ background: #f6f6f6; padding: 2px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>NSGA-II Pareto Front — Ensemble Weights (Multi-Objective Optimization)</h1>

<p>Source: <code>{IN_JSON.name}</code> &middot;
Algorithm: <b>{data['algorithm']}</b> &middot;
Pop size: {data['population_size']} &middot;
Generations: {data['generations']} &middot;
Objectives: {', '.join(data['objectives'])} &middot;
Pareto front size: {data['pareto_front_size']} &middot;
Runtime: {data['runtime_s']:.1f} s</p>

<div class="knee"><b>Knee point:</b> {knee_txt}</div>

<h2>Macro-F1 vs Expected Calibration Error</h2>
<figure>
{svg1}
<figcaption>Each point is one Pareto-optimal ensemble weight vector (logreg/svm/tfidf).
The knee point (highlighted in red) is the trade-off recommended for the thesis headline number.
Top-left is the best region: high F1, low ECE. The Pareto front shows that the CI layer
primarily trades a small amount of F1 for a substantial reduction in ECE — which
is the defensible contribution of the multi-objective framework.</figcaption>
</figure>

<h2>Macro-F1 vs Coverage</h2>
<figure>
{svg2}
<figcaption>Coverage = fraction of predictions exceeding the confidence threshold τ=0.7.
More coverage means fewer abstentions. This view shows that raising coverage beyond
the knee costs F1 — another real trade-off for the selective-prediction story.</figcaption>
</figure>

<h2>ECE vs Coverage</h2>
<figure>
{svg3}
<figcaption>The two "operational" objectives that have no direct tension with F1.
Together with the first plot they give the examiner a three-axis view of the
Pareto surface projected onto all three pairs.</figcaption>
</figure>

</body>
</html>
"""
    (OUT_DIR / "pareto_visualization.html").write_text(html)

    print("Wrote:")
    for p in [
        OUT_DIR / "pareto_f1_vs_ece.svg",
        OUT_DIR / "pareto_f1_vs_coverage.svg",
        OUT_DIR / "pareto_ece_vs_coverage.svg",
        OUT_DIR / "pareto_visualization.html",
    ]:
        print(" ", p)
    print(f"Knee index: {knee_idx}")


if __name__ == "__main__":
    main()
