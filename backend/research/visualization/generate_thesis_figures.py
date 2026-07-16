# -*- coding: utf-8 -*-
"""Regenerate thesis figures from real pinned artifacts (no synthetic data)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BACKEND = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BACKEND, "figures", "thesis")
os.makedirs(OUT, exist_ok=True)

def load(p):
    with open(os.path.join(BACKEND, p), encoding="utf-8") as f:
        return json.load(f)

# ---------- style ----------
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURFACE = "#ffffff"
# fixed entity colors (categorical slots, fixed order — never cycled)
MODEL_COLOR = {
    "logreg":         "#2a78d6",  # blue
    "svm":            "#1baf7a",  # aqua
    "tfidf":          "#eda100",  # yellow
    "ensemble_pso":   "#008300",  # green
    "ensemble_nsga2": "#4a3aa7",  # violet
    "meta_learner":   "#e34948",  # red
    "fuzzy_ensemble": "#e87ba4",  # magenta
}
MODEL_LABEL = {
    "logreg": "Logistic Regression", "svm": "Linear SVM", "tfidf": "Naive Bayes (TF-IDF)",
    "ensemble_pso": "PSO ensemble", "ensemble_nsga2": "NSGA-II ensemble",
    "meta_learner": "Meta-learner", "fuzzy_ensemble": "Fuzzy ensemble",
}
# sequential blue ramp (light -> dark)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORDINAL3 = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal: low/mid/high

from matplotlib.colors import LinearSegmentedColormap
BLUES = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11.5,
    "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 200, "figure.constrained_layout.use": True,
})

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)

def bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, fontsize=9.5, color=INK2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=fontsize, color=color)

# ================================================================
# FIG 5 replacement: PSO convergence (results/pso_convergence.json)
# ================================================================
pso = load("results/pso_convergence.json")
hist = pso["convergence_history"]
it = [h["iteration"] for h in hist]
gb = [h["global_best_f1"] for h in hist]
mb = [h["mean_personal_best_f1"] for h in hist]
best = pso["results"]["pso_val_f1"]

fig, ax = plt.subplots(figsize=(9.2, 5.5))
ax.plot(it, gb, color=MODEL_COLOR["logreg"], lw=2.4, zorder=3)
ax.plot(it, mb, color=MUTED, lw=1.8, ls="--", zorder=2)
ax.axhline(best, color=AXIS, lw=1, ls=":")
ax.scatter([it[-1]], [gb[-1]], s=55, color=MODEL_COLOR["logreg"], zorder=4,
           edgecolor=SURFACE, linewidth=1.5)
ax.annotate(f"best {best:.4f}", (it[-1], gb[-1]), xytext=(-8, 10),
            textcoords="offset points", ha="right", fontsize=10.5, color=INK, fontweight="bold")
ax.text(it[-1], mb[-1] - 0.0006, "swarm mean personal best", ha="right", va="top",
        fontsize=10, color=MUTED)
ax.text(6.2, gb[5] + 0.0004, "global best", fontsize=10, color=MODEL_COLOR["logreg"], va="bottom")
ax.set_xlabel("PSO iteration")
ax.set_ylabel("Validation macro-F1")
ax.set_title("PSO ensemble-weight search converges by iteration ~30")
ax.set_xlim(1, it[-1])
cfg = pso["config"]
ax.text(0, -0.16, f"{cfg['n_particles']} particles · {cfg['n_iterations']} iterations · seed {cfg['seed']} · "
        f"models: logreg, svm, tfidf · artifact: results/pso_convergence.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_pso_convergence.png")

# ================================================================
# FIG 6 replacement: neuro-fuzzy learned MFs over base-model confidence
# (results/runtime/route_a_live_v1/neuro_fuzzy_gate.json)
# ================================================================
nf = load("results/runtime/route_a_live_v1/neuro_fuzzy_gate.json")
mfs = nf["learned_mfs"]
models = ["logreg", "svm", "tfidf"]
sets_order = ["Low", "Medium", "High"]
set_color = dict(zip(sets_order, ORDINAL3))

x = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
for ax, m in zip(axes, models):
    for s in sets_order:
        rec = next(r for r in mfs if r["model"] == m and r["fuzzy_set"] == s)
        mu = np.exp(-0.5 * ((x - rec["center"]) / max(rec["width"], 1e-6)) ** 2)
        ax.plot(x, mu, color=set_color[s], lw=2.2, label=s if m == "logreg" else None)
        ax.fill_between(x, mu, color=set_color[s], alpha=0.12)
    ax.set_title(MODEL_LABEL[m], fontsize=12, loc="center")
    ax.set_xlabel("Base-model confidence")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
axes[0].legend(loc="center left", title="Fuzzy set", title_fontsize=10)
axes[0].set_ylabel("Membership degree")
fig.suptitle("Learned Gaussian membership functions of the neuro-fuzzy gate (ANFIS, 27 parameters)",
             fontsize=13.5, fontweight="bold", x=0.005, ha="left")
axes[0].text(0, -0.22, "Centers/widths learned on validation · artifact: route_a_live_v1/neuro_fuzzy_gate.json",
             transform=axes[0].transAxes, fontsize=9, color=MUTED)
save(fig, "fig_fuzzy_mfs.png")

# ================================================================
# FIG 8 replacement: category-slice heatmap
# (results/domain_shift/category_domain_shift.json)
# ================================================================
ds = load("results/domain_shift/category_domain_shift.json")
CAT = {"1": "Film & Animation", "2": "Autos & Vehicles", "15": "Pets & Animals",
       "17": "Sports", "20": "Gaming", "22": "People & Blogs", "24": "Entertainment",
       "25": "News & Politics", "26": "Howto & Style", "27": "Education",
       "28": "Science & Tech"}
ds_models = ds["models"]
slices = sorted(ds["results"][ds_models[0]]["slices"], key=lambda s: -s["n_samples"])
slice_ids = [s["slice"] for s in slices]
M = np.zeros((len(ds_models), len(slice_ids)))
for i, m in enumerate(ds_models):
    bys = {s["slice"]: s["macro_f1"] for s in ds["results"][m]["slices"]}
    for j, sid in enumerate(slice_ids):
        M[i, j] = bys[sid]

fig, ax = plt.subplots(figsize=(11.6, 5.4))
vmin, vmax = M.min(), M.max()
im = ax.imshow(M, cmap=BLUES, aspect="auto", vmin=vmin - 0.02, vmax=vmax)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        frac = (v - (vmin - 0.02)) / (vmax - (vmin - 0.02))
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                color=(SURFACE if frac > 0.62 else INK))
ax.set_xticks(range(len(slice_ids)),
              [f"{CAT.get(s, 'Cat ' + s)} (n={slices[j]['n_samples']})" for j, s in enumerate(slice_ids)],
              fontsize=9.5, rotation=32, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(ds_models)), [MODEL_LABEL.get(m, m) for m in ds_models], fontsize=10.5)
ax.grid(False)
for spine in ax.spines.values(): spine.set_visible(False)
# white gaps between cells
ax.set_xticks(np.arange(-0.5, len(slice_ids), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ds_models), 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="both", length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
cb.set_label("Macro-F1", color=INK2)
cb.outline.set_visible(False)
ax.set_title("Macro-F1 by YouTube category slice (n=1,641 metadata-matched test comments)")
ax.text(0, -0.30, "Slices ordered by size; min slice n=30 · artifact: results/domain_shift/category_domain_shift.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_category_heatmap.png")

# ================================================================
# FIG 10 replacement: model comparison on the full 165,110 test set
# (results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.json)
# ================================================================
rt = load("results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.json")
rows = rt["results"]
order = ["meta_learner", "ensemble_pso", "fuzzy_ensemble", "ensemble_nsga2",
         "logreg", "svm", "tfidf"]
rows = sorted(rows, key=lambda r: order.index(r["model"]))
names = [MODEL_LABEL[r["model"]] for r in rows]
cols = [MODEL_COLOR[r["model"]] for r in rows]
acc = [r["accuracy"] for r in rows]
f1 = [r["macro_f1"] for r in rows]
ece = [r["ece"] for r in rows]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.1),
                             gridspec_kw={"width_ratios": [1.45, 1]})
y = np.arange(len(rows))
h = 0.38
b1 = a1.barh(y - h/2, acc, height=h, color=cols, edgecolor=SURFACE, linewidth=1)
b2 = a1.barh(y + h/2, f1, height=h, color=cols, alpha=0.55, edgecolor=SURFACE, linewidth=1)
for yi, (a, f) in enumerate(zip(acc, f1)):
    a1.text(a + 0.002, yi - h/2, f"{a:.4f}", va="center", fontsize=9, color=INK2)
    a1.text(f + 0.002, yi + h/2, f"{f:.4f}", va="center", fontsize=9, color=MUTED)
a1.set_yticks(y, names)
a1.invert_yaxis()
a1.set_xlim(0.60, 0.72)
a1.set_xlabel("Score (full 165,110-comment held-out test)")
a1.set_title("Accuracy (solid) and macro-F1 (tint)")
a1.grid(axis="y", visible=False)

b3 = a2.barh(y, ece, height=0.55, color=cols, edgecolor=SURFACE, linewidth=1)
for yi, e in enumerate(ece):
    a2.text(e + 0.0003, yi, f"{e:.4f}", va="center", fontsize=9, color=INK2)
a2.set_yticks(y, ["" for _ in y])
a2.invert_yaxis()
a2.set_xlabel("Expected Calibration Error  (lower is better)")
a2.set_title("Calibration (ECE)")
a2.grid(axis="y", visible=False)
a2.set_xlim(0, max(ece) * 1.22)
fig.suptitle("route_a_live_v1 runtime benchmark: ensembles tie on F1; NSGA-II & fuzzy win calibration",
             fontsize=13.5, fontweight="bold", x=0.005, ha="left")
a1.text(0, -0.19, "artifact: results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.json",
        transform=a1.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_model_comparison.png")

# ================================================================
# FIG 11 replacement: per-class F1 (Table 12, 5,000-comment sample)
# ================================================================
pcf = {
    "meta_learner":   {"Negative": 0.707, "Neutral": 0.623, "Positive": 0.758},
    "ensemble_nsga2": {"Negative": 0.710, "Neutral": 0.615, "Positive": 0.755},
    "logreg":         {"Negative": 0.708, "Neutral": 0.611, "Positive": 0.753},
    "tfidf":          {"Negative": 0.684, "Neutral": 0.558, "Positive": 0.721},
}
classes = ["Negative", "Neutral", "Positive"]
fig, ax = plt.subplots(figsize=(10.6, 5.2))
n_m = len(pcf); w = 0.19
xpos = np.arange(len(classes))
for k, (m, vals) in enumerate(pcf.items()):
    xs = xpos + (k - (n_m - 1)/2) * w
    bars = ax.bar(xs, [vals[c] for c in classes], width=w - 0.02,
                  color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=1,
                  label=MODEL_LABEL[m])
    bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, fontsize=8.8)
# highlight the Neutral gap
ax.axvspan(0.62, 1.38, color="#f0efec", zorder=0)
ax.text(1, 0.78, "Neutral lags every model\nby 0.09–0.16 F1", ha="center",
        fontsize=10.5, color=INK, fontweight="bold")
ax.set_xticks(xpos, classes, fontsize=11.5)
ax.set_ylim(0.5, 0.82)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1: Neutral is the consistent weak point (5,000-comment sample)")
ax.legend(ncols=2, loc="upper left")
ax.grid(axis="x", visible=False)
ax.text(0, -0.14, "artifact: results/confusion_matrices.json · Table 8",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_perclass_f1.png")

# ================================================================
# FIG 12 replacement: confusion matrix (meta_learner, 5k sample)
# ================================================================
cms = load("results/confusion_matrices.json")
mm = cms["meta_learner"]
Mn = np.array(mm["normalized"]); Mc = np.array(mm["matrix"]); labels = mm["labels"]
fig, ax = plt.subplots(figsize=(7.6, 6.2))
im = ax.imshow(Mn, cmap=BLUES, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        c = SURFACE if Mn[i, j] > 0.55 else INK
        ax.text(j, i, f"{Mn[i,j]*100:.1f}%", ha="center", va="center",
                fontsize=15, fontweight="bold", color=c)
        ax.text(j, i + 0.22, f"n={Mc[i,j]:,}", ha="center", va="center",
                fontsize=9.5, color=(SURFACE if Mn[i, j] > 0.55 else MUTED))
ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.grid(False)
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=3)
ax.tick_params(which="both", length=0)
ax.set_title("Row-normalised confusion matrix — meta-learner")
ax.text(0, -0.14, "5,000-comment sample, seed 42 · artifact: results/confusion_matrices.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_confusion_matrix.png")

# ================================================================
# FIG 13 replacement: coverage–accuracy curves (the broken figure)
# (results/route_a_live_v1_ci/coverage_accuracy_curve.json)
# ================================================================
cov = load("results/route_a_live_v1_ci/coverage_accuracy_curve.json")
fig, ax = plt.subplots(figsize=(9.6, 5.7))
summ = {s["model"]: s for s in cov["summary"]}
plot_order = ["meta_learner", "ensemble_pso", "fuzzy_ensemble", "ensemble_nsga2",
              "logreg", "svm", "tfidf"]
for m in plot_order:
    pts = cov["curves"][m]
    cvg = [p["coverage"] for p in pts]; a = [p["accuracy"] for p in pts]
    lw = 2.4 if m in ("meta_learner", "tfidf", "logreg") else 1.6
    ax.plot(cvg, a, color=MODEL_COLOR[m], lw=lw, label=f"{MODEL_LABEL[m]} (AUCA {summ[m]['auca']:.3f})")
ax.axhline(summ["meta_learner"]["acc_full"], color=AXIS, lw=0.9, ls=":")
ax.text(0.015, summ["meta_learner"]["acc_full"] + 0.006,
        f"full-coverage accuracy ≈ {summ['meta_learner']['acc_full']:.2f}",
        fontsize=9.5, color=MUTED)
# direct labels
ax.text(0.28, 0.955, "meta-learner / ensembles", fontsize=10, color=MODEL_COLOR["meta_learner"],
        fontweight="bold")
ax.text(0.60, 0.735, "Naive Bayes (TF-IDF)", fontsize=10, color="#8a5d00", fontweight="bold")
ax.set_xlabel("Coverage (share of comments not abstained)")
ax.set_ylabel("Accuracy on covered comments")
ax.set_title("Selective prediction: abstaining on low-confidence comments buys accuracy")
ax.set_xlim(0, 1); ax.set_ylim(0.6, 1.0)
ax.legend(loc="lower left", fontsize=8.8, ncols=2)
ax.text(0, -0.14, "20,000-comment sample of the 165,110-row held-out test split, seed 42 · "
        "artifact: results/route_a_live_v1_ci/coverage_accuracy_curve.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_coverage_accuracy.png")

# ================================================================
# FIG 15 replacement: PSO vs NSGA-II ensemble weights (live artifacts)
# ================================================================
psow = load("results/runtime/route_a_live_v1/pso_ensemble_weights.json")["weights"]
nsga = load("results/runtime/route_a_live_v1/multi_objective_ensemble.json")["knee_point"]["weights"]
base = ["logreg", "svm", "tfidf"]
fig, ax = plt.subplots(figsize=(9.2, 5.2))
xpos = np.arange(len(base)); w = 0.34
bp = ax.bar(xpos - w/2, [psow[b] for b in base], width=w - 0.02,
            color=MODEL_COLOR["ensemble_pso"], edgecolor=SURFACE, linewidth=1, label="PSO (single-objective)")
bn = ax.bar(xpos + w/2, [nsga[b] for b in base], width=w - 0.02,
            color=MODEL_COLOR["ensemble_nsga2"], edgecolor=SURFACE, linewidth=1, label="NSGA-II (knee point)")
bar_labels(ax, bp, fmt="{:.3f}", dy=0.008, fontsize=10)
bar_labels(ax, bn, fmt="{:.3f}", dy=0.008, fontsize=10)
ax.set_xticks(xpos, [MODEL_LABEL[b] for b in base], fontsize=11.5)
ax.set_ylabel("Ensemble weight")
ax.set_ylim(0, 1.02)
ax.set_title("Both optimisers concentrate weight on Logistic Regression")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
ax.text(0, -0.14, "artifacts: route_a_live_v1/pso_ensemble_weights.json · multi_objective_ensemble.json (knee point)",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_ensemble_weights.png")

# ================================================================
# NEW: class distribution across splits (Table 3 numbers)
# ================================================================
splits = ["Train", "Validation", "Test"]
counts = {
    "Negative": [184266, 47023, 59614],
    "Neutral":  [160744, 39037, 50540],
    "Positive": [171876, 42794, 54956],
}
totals = [516886, 128854, 165110]
CLASS_COLOR = {"Negative": "#e34948", "Neutral": "#898781", "Positive": "#008300"}
fig, ax = plt.subplots(figsize=(9.6, 4.4))
left = np.zeros(3)
ypos = np.arange(3)
for cls in ["Negative", "Neutral", "Positive"]:
    shares = [counts[cls][i] / totals[i] for i in range(3)]
    b = ax.barh(ypos, shares, left=left, height=0.52, color=CLASS_COLOR[cls],
                edgecolor=SURFACE, linewidth=2, label=cls)
    for i, (s, c) in enumerate(zip(shares, counts[cls])):
        ax.text(left[i] + s/2, i, f"{cls}\n{s*100:.1f}%  ({c:,})",
                ha="center", va="center", fontsize=9.6, color=SURFACE, fontweight="bold")
    left += shares
for i, t in enumerate(totals):
    ax.text(1.008, i, f"{t:,}", va="center", fontsize=10, color=INK2)
ax.text(1.008, -0.62, "rows", fontsize=9, color=MUTED)
ax.set_yticks(ypos, splits, fontsize=11.5)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.set_title("Class distribution is stable across group-aware splits (by VideoID)")
ax.grid(axis="y", visible=False)
ax.text(0, -0.20, "Split sizes from Table 3 · group-aware split prevents topical leakage",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_class_distribution.png")

# ================================================================
# NEW: comment length (words) by class (results/eda/eda_report.json)
# ================================================================
eda = load("results/eda/eda_report.json")
pc = eda["length"]["per_class"]
stats = ["median", "mean", "p90"]
stat_labels = {"median": "Median", "mean": "Mean", "p90": "90th percentile"}
fig, ax = plt.subplots(figsize=(9.6, 4.8))
xpos = np.arange(3); w = 0.26
for k, s in enumerate(stats):
    vals = [pc[c]["words"][s] for c in classes]
    bars = ax.bar(xpos + (k - 1) * w, vals, width=w - 0.02, color=ORDINAL3[k],
                  edgecolor=SURFACE, linewidth=1, label=stat_labels[s])
    bar_labels(ax, bars, fmt="{:.0f}", dy=0.5, fontsize=10)
ax.set_xticks(xpos, classes, fontsize=11.5)
ax.set_ylabel("Comment length (words)")
ax.set_ylim(0, 52)
ax.set_title("Neutral comments are the shortest — less lexical evidence per decision")
ax.legend(loc="upper center", ncols=3)
ax.grid(axis="x", visible=False)
ax.text(0, -0.15, f"50,000-comment EDA sample (n per class: "
        f"{pc['Negative']['n']:,} / {pc['Neutral']['n']:,} / {pc['Positive']['n']:,}) · "
        "artifact: results/eda/eda_report.json", transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_comment_length.png")

# ================================================================
# NEW: one-vs-rest ROC-AUC per class (Table 7 numbers, 5k sample)
# ================================================================
auc = {
    "ensemble_nsga2": {"macro": 0.8596, "Positive": 0.8948, "Neutral": 0.8184, "Negative": 0.8655},
    "logreg":         {"macro": 0.8589, "Positive": 0.8945, "Neutral": 0.8172, "Negative": 0.8652},
    "meta_learner":   {"macro": 0.8577, "Positive": 0.8922, "Neutral": 0.8186, "Negative": 0.8623},
    "ensemble_pso":   {"macro": 0.8511, "Positive": 0.8897, "Neutral": 0.8060, "Negative": 0.8575},
    "svm":            {"macro": 0.8434, "Positive": 0.8847, "Neutral": 0.7954, "Negative": 0.8501},
    "tfidf":          {"macro": 0.8315, "Positive": 0.8684, "Neutral": 0.7925, "Negative": 0.8335},
}
fig, ax = plt.subplots(figsize=(9.8, 5.4))
ypos = np.arange(len(auc))
mnames = list(auc.keys())
for i, m in enumerate(mnames):
    v = auc[m]
    ax.plot([min(v["Neutral"], v["Negative"], v["Positive"]),
             max(v["Neutral"], v["Negative"], v["Positive"])], [i, i],
            color=GRID, lw=2, zorder=1)
    for cls, marker in [("Positive", "o"), ("Negative", "o"), ("Neutral", "o")]:
        ax.scatter(v[cls], i, s=95, color=CLASS_COLOR[cls], zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
    ax.scatter(v["macro"], i, s=120, marker="D", facecolor=SURFACE,
               edgecolor=INK, linewidth=1.4, zorder=4)
ax.set_yticks(ypos, [MODEL_LABEL[m] for m in mnames], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("One-vs-rest ROC-AUC")
ax.set_title("Neutral is the weakest class for every model (one-vs-rest AUC)")
# legend via proxy artists
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", markersize=9, color=CLASS_COLOR[c], label=c)
           for c in ["Positive", "Negative", "Neutral"]]
handles.append(Line2D([], [], marker="D", ls="", markersize=9, markerfacecolor=SURFACE,
                      markeredgecolor=INK, label="Macro AUC"))
ax.legend(handles=handles, loc="upper center", ncols=4,
          bbox_to_anchor=(0.5, -0.14))
ax.grid(axis="y", visible=False)
ax.set_xlim(0.78, 0.91)
ax.text(0, -0.30, "5,000-comment sample, seed 42 · Table 7 / artifact: results/roc_auc.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_roc_auc.png")

# ================================================================
# NEW: gold-set agreement (Table 10 numbers, 291 human labels)
# ================================================================
gold = {
    "ensemble_pso":   {"acc": 0.7010, "f1": 0.7042, "neu": 0.6226},
    "ensemble_nsga2": {"acc": 0.6976, "f1": 0.7006, "neu": 0.6262},
    "meta_learner":   {"acc": 0.6976, "f1": 0.7001, "neu": 0.6393},
    "tfidf":          {"acc": 0.7010, "f1": 0.6988, "neu": 0.5946},
    "svm":            {"acc": 0.6942, "f1": 0.6978, "neu": 0.6140},
    "logreg":         {"acc": 0.6907, "f1": 0.6940, "neu": 0.6168},
}
fig, ax = plt.subplots(figsize=(10.6, 5.0))
gm = list(gold.keys())
xpos = np.arange(len(gm)); w = 0.26
metrics = [("acc", "Accuracy", 0), ("f1", "Macro-F1", 1), ("neu", "Neutral-F1", 2)]
for key, lbl, k in metrics:
    vals = [gold[m][key] for m in gm]
    bars = ax.bar(xpos + (k - 1) * w, vals, width=w - 0.02, color=ORDINAL3[k],
                  edgecolor=SURFACE, linewidth=1, label=lbl)
    bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, fontsize=8.6)
ax.set_xticks(xpos, [MODEL_LABEL[m].replace(" ", "\n", 1) for m in gm], fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.set_ylabel("Score vs human gold labels")
ax.set_title("Agreement with the 291-item human gold set (Krippendorff's α = 0.9547)")
ax.legend(loc="upper right", ncols=3)
ax.grid(axis="x", visible=False)
ax.text(0, -0.19, "Two independent annotation passes, reconciled · Table 10 / artifact: results/gold_set_metrics_from_dataset.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_gold_set.png")

# ================================================================
# NEW: Neutral-prior threshold sweep (results/neutral_analysis/neutral_analysis.json)
# ================================================================
na = load("results/neutral_analysis/neutral_analysis.json")
sweep = na["validation_sweep"]
al = [s["alpha"] for s in sweep]
series = [
    ("neutral_recall",    "Neutral recall",    MODEL_COLOR["logreg"], "-"),
    ("neutral_f1",        "Neutral F1",        MODEL_COLOR["ensemble_nsga2"], "-"),
    ("macro_f1",          "Macro-F1",          INK, "-"),
    ("neutral_precision", "Neutral precision", MODEL_COLOR["tfidf"], "-"),
]
fig, ax = plt.subplots(figsize=(9.8, 5.6))
for key, lbl, colr, ls in series:
    vals = [s[key] for s in sweep]
    ax.plot(al, vals, color=colr, lw=2.2, ls=ls)
    ax.text(al[-1] + 0.015, vals[-1], lbl, fontsize=10.2, color=colr,
            va="center", fontweight="bold")
chosen = na["chosen_alpha"]
ax.axvline(chosen, color=AXIS, lw=1.2, ls="--")
ax.text(chosen, ax.get_ylim()[1], f"  chosen α = {chosen}", fontsize=10.5,
        color=INK, fontweight="bold", va="top")
ax.set_xlabel("Neutral prior multiplier α")
ax.set_ylabel("Validation score")
ax.set_title("Threshold tuning trades Neutral precision for recall; macro-F1 barely moves")
ax.set_xlim(al[0], al[-1] + 0.32)
ax.text(0, -0.14, "Logistic regression, 8,000-comment validation sample · artifact: results/neutral_analysis/neutral_analysis.json",
        transform=ax.transAxes, fontsize=9, color=MUTED)
save(fig, "fig_neutral_sweep.png")

print("ALL DONE")
