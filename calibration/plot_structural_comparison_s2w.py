"""
plot_structural_comparison_s2w.py

Figure: CMA-ES-S2 vs CMA-ES-S2W vs TabICL structural metrics (SS_rld, SS_angle, Chamfer)
for the 20 Scenario-2 test cases.

Layout:
  Top row   — dumbbell plots per test case (sorted by quintile + g_axial)
  Bottom row — quintile-mean bar charts ± SE

Output: calibration/figures/structural_comparison_s2w.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(CALIB_DIR, "comparison_structure_metrics_s2w.csv")
FIG_PATH  = os.path.join(CALIB_DIR, "figures", "structural_comparison_s2w.png")
os.makedirs(os.path.join(CALIB_DIR, "figures"), exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

s2    = df[df["method"] == "CMA-ES-S2"].copy()
s2w   = df[df["method"] == "CMA-ES-S2W"].copy()
tabicl = df[df["method"] == "TabICL"].copy()

keys = ["config", "sample_id"]
meta_cols = keys + ["g_axial_quintile", "true_g_axial"]

merged = s2[meta_cols + ["ss_rld", "ss_angle", "chamfer_cm"]].merge(
    s2w[keys   + ["ss_rld", "ss_angle", "chamfer_cm"]], on=keys, suffixes=("_s2", "_s2w")
).merge(
    tabicl[keys + ["ss_rld", "ss_angle", "chamfer_cm"]], on=keys
)
merged = merged.rename(columns={
    "ss_rld":    "ss_rld_tab",
    "ss_angle":  "ss_angle_tab",
    "chamfer_cm":"chamfer_cm_tab",
})
merged = merged.sort_values(["g_axial_quintile", "true_g_axial"]).reset_index(drop=True)
merged["label"] = merged["config"] + "\n" + merged["sample_id"].astype(str)
N = len(merged)

# ── Style ─────────────────────────────────────────────────────────────────────

C_S2    = "#2E6DA4"   # steel blue  — CMA-ES cold-start
C_S2W   = "#2CA02C"   # green       — CMA-ES warm-start
C_TAB   = "#D95F02"   # burnt orange — TabICL

Q_COLORS = {"Q1": "#4393C3", "Q2": "#92C5DE", "Q3": "#B8D97C",
             "Q4": "#F4A582", "Q5": "#D6604D"}
Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.32,
                       left=0.07, right=0.97, top=0.91, bottom=0.08)

axes_top = [fig.add_subplot(gs[0, c]) for c in range(3)]
axes_bot = [fig.add_subplot(gs[1, c]) for c in range(3)]

metrics = [
    ("ss_rld",     r"$SS_{RLD}$",   True,  "Root length density skill score"),
    ("ss_angle",   r"$SS_{angle}$", True,  "Crossing-angle skill score"),
    ("chamfer_cm", "Chamfer (cm)",  False, "Chamfer distance"),
]

suffixes = {"s2": "_s2", "s2w": "_s2w", "tab": "_tab"}

# ── Top row: dumbbell plots ───────────────────────────────────────────────────

for col, (metric, ylabel, has_zero, _) in enumerate(metrics):
    ax = axes_top[col]

    ys  = np.arange(N)
    yc  = merged[f"{metric}_s2"].values
    yw  = merged[f"{metric}_s2w"].values
    yt  = merged[f"{metric}_tab"].values

    # Connecting lines between all three points
    for i in range(N):
        vals = sorted([yc[i], yw[i], yt[i]])
        ax.plot([vals[0], vals[-1]], [ys[i], ys[i]],
                lw=0.8, color="0.78", zorder=1)

    # Markers: square=S2, diamond=S2W, circle=TabICL
    for i, row in merged.iterrows():
        qc = Q_COLORS[row["g_axial_quintile"]]
        ax.scatter(row[f"{metric}_s2"],  i, marker="s", s=26, color=qc,
                   edgecolors=C_S2,  linewidths=0.9, zorder=3)
        ax.scatter(row[f"{metric}_s2w"], i, marker="D", s=24, color=qc,
                   edgecolors=C_S2W, linewidths=0.9, zorder=4)
        ax.scatter(row[f"{metric}_tab"], i, marker="o", s=26, color=qc,
                   edgecolors=C_TAB, linewidths=0.9, zorder=3)

    if has_zero:
        ax.axvline(0, color="0.50", lw=0.8, ls="--", zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels(merged["label"].values, fontsize=6.2)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(["(a)", "(b)", "(c)"][col], loc="left", fontsize=9, fontweight="bold")

    # Quintile bands + labels
    for q in Q_ORDER:
        rows_q = merged[merged["g_axial_quintile"] == q].index.tolist()
        if not rows_q:
            continue
        ymin, ymax = min(rows_q) - 0.45, max(rows_q) + 0.45
        ax.axhspan(ymin, ymax, alpha=0.07, color=Q_COLORS[q], zorder=0)

    ax.tick_params(axis="x", labelsize=8)

# ── Bottom row: quintile bar charts ──────────────────────────────────────────

for col, (metric, ylabel, has_zero, full_label) in enumerate(metrics):
    ax = axes_bot[col]

    x = np.arange(len(Q_ORDER))
    w = 0.24

    for qi, q in enumerate(Q_ORDER):
        sub_s2  = merged[merged["g_axial_quintile"] == q][f"{metric}_s2"]
        sub_s2w = merged[merged["g_axial_quintile"] == q][f"{metric}_s2w"]
        sub_tab = merged[merged["g_axial_quintile"] == q][f"{metric}_tab"]

        for sub, color, offset, lbl in [
            (sub_s2,  C_S2,  -w,   "CMA-ES-S2"),
            (sub_s2w, C_S2W,  0,   "CMA-ES-S2W"),
            (sub_tab, C_TAB, +w,   "TabICL"),
        ]:
            m = sub.mean()
            e = sub.sem() if len(sub) > 1 else 0
            ax.bar(qi + offset, m, w, color=color, alpha=0.85,
                   label=lbl if qi == 0 else "")
            ax.errorbar(qi + offset, m, yerr=e, fmt="none",
                        color="0.3", lw=1.0, capsize=2)

    if has_zero:
        ax.axhline(0, color="0.45", lw=0.8, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(Q_ORDER, fontsize=8)
    ax.set_xlabel("g_axial quintile (Q1=slow → Q5=fast)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(["(d)", "(e)", "(f)"][col], loc="left", fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    if col == 0:
        ax.legend(fontsize=7.5, frameon=False)

# ── Overall means annotation ──────────────────────────────────────────────────

means = {
    "CMA-ES-S2":  {m: merged[f"{m}_s2"].mean()  for m, *_ in metrics},
    "CMA-ES-S2W": {m: merged[f"{m}_s2w"].mean() for m, *_ in metrics},
    "TabICL":     {m: merged[f"{m}_tab"].mean()  for m, *_ in metrics},
}

for col, (metric, ylabel, *_) in enumerate(metrics):
    ax = axes_bot[col]
    ms2, ms2w, mt = means["CMA-ES-S2"][metric], means["CMA-ES-S2W"][metric], means["TabICL"][metric]
    if metric == "chamfer_cm":
        ann = f"mean: S2={ms2:.3f}  S2W={ms2w:.3f}  Tab={mt:.3f} cm"
    else:
        ann = f"mean: S2={ms2:+.3f}  S2W={ms2w:+.3f}  Tab={mt:+.3f}"
    ax.set_title(ax.get_title() + f"   {ann}", loc="left", fontsize=7, fontweight="bold")

# ── Shared legend ─────────────────────────────────────────────────────────────

legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.65",
           markeredgecolor=C_S2,  markersize=7, label="CMA-ES-S2 (cold-start)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="0.65",
           markeredgecolor=C_S2W, markersize=7, label="CMA-ES-S2W (warm-start)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.65",
           markeredgecolor=C_TAB, markersize=7, label="TabICL"),
]
q_patches = [mpatches.Patch(color=Q_COLORS[q], alpha=0.7, label=f"{q} band")
             for q in Q_ORDER]

fig.legend(handles=legend_elements + q_patches,
           loc="upper center", ncol=8, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 0.997))

fig.suptitle(
    "Structural similarity: CMA-ES-S2 vs CMA-ES-S2W vs TabICL  "
    "(Scenario 2, N=20 test cases)",
    y=1.003, fontsize=10.5, fontweight="bold")

plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_PATH}")
