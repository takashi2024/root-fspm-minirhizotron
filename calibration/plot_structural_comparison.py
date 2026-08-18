"""
plot_structural_comparison.py

Figure: CMA-ES vs TabICL structural metrics (SS_rld, SS_angle, Chamfer)
for the 20 Scenario-2 test cases.

Layout:
  Top row   — dumbbell plots per test case (sorted by quintile + g_axial)
  Bottom row — quintile-mean bar charts ± SE

Output: calibration/figures/structural_comparison_cmae_tabicl.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CALIB_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(CALIB_DIR, "comparison_structure_metrics.csv")
FIG_PATH   = os.path.join(CALIB_DIR, "figures", "structural_comparison_cmae_tabicl.png")
os.makedirs(os.path.join(CALIB_DIR, "figures"), exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

cmaes  = df[df["method"] == "CMA-ES"].copy()
tabicl = df[df["method"] == "TabICL"].copy()

keys = ["config", "sample_id"]
merged = cmaes[keys + ["g_axial_quintile", "true_g_axial",
                        "ss_rld", "ss_angle", "chamfer_cm"]].merge(
    tabicl[keys + ["ss_rld", "ss_angle", "chamfer_cm"]],
    on=keys, suffixes=("_cmaes", "_tabicl")
)
merged = merged.sort_values(["g_axial_quintile", "true_g_axial"]).reset_index(drop=True)
merged["label"] = merged["config"] + "\n" + merged["sample_id"].astype(str)
N = len(merged)

# ── Style ─────────────────────────────────────────────────────────────────────

C_CMAES  = "#2E6DA4"   # steel blue
C_TABICL = "#D95F02"   # burnt orange

Q_COLORS = {"Q1": "#4393C3", "Q2": "#92C5DE", "Q3": "#B8D97C",
             "Q4": "#F4A582", "Q5": "#D6604D"}
Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.linewidth": 0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 8.5))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.30,
                       left=0.07, right=0.97, top=0.93, bottom=0.08)

axes_top = [fig.add_subplot(gs[0, c]) for c in range(3)]
axes_bot = [fig.add_subplot(gs[1, c]) for c in range(3)]

metrics = [
    ("ss_rld",    r"$SS_{RLD}$",   True,  "Root length density skill score"),
    ("ss_angle",  r"$SS_{angle}$", True,  "Crossing-angle skill score"),
    ("chamfer_cm","Chamfer (cm)", False, "Chamfer distance"),
]

# ── Top row: dumbbell plots ───────────────────────────────────────────────────

for col, (metric, ylabel, has_zero, _) in enumerate(metrics):
    ax = axes_top[col]

    yc = merged[f"{metric}_cmaes"].values
    yt = merged[f"{metric}_tabicl"].values
    ys = np.arange(N)

    # Connecting line (gray)
    for i in range(N):
        color = "0.78"
        ax.plot([yc[i], yt[i]], [ys[i], ys[i]],
                lw=0.9, color=color, zorder=1)

    # Quintile-colored markers for CMA-ES (square)
    for i, row in merged.iterrows():
        qc = Q_COLORS[row["g_axial_quintile"]]
        ax.scatter(row[f"{metric}_cmaes"], i,
                   marker="s", s=28, color=qc,
                   edgecolors=C_CMAES, linewidths=0.9, zorder=3)
        ax.scatter(row[f"{metric}_tabicl"], i,
                   marker="o", s=28, color=qc,
                   edgecolors=C_TABICL, linewidths=0.9, zorder=3)

    if has_zero:
        ax.axvline(0, color="0.55", lw=0.8, ls="--", zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels(merged["label"].values, fontsize=6.5)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(["(a)", "(b)", "(c)"][col], loc="left", fontsize=9, fontweight="bold")

    # Quintile bands
    for q_idx, q in enumerate(Q_ORDER):
        rows_q = merged[merged["g_axial_quintile"] == q].index.tolist()
        if not rows_q:
            continue
        ymin, ymax = min(rows_q) - 0.45, max(rows_q) + 0.45
        ax.axhspan(ymin, ymax, alpha=0.07, color=Q_COLORS[q], zorder=0)
        ax.text(ax.get_xlim()[0] if col == 0 else ax.get_xlim()[0],
                (ymin + ymax) / 2, q,
                fontsize=6, color=Q_COLORS[q],
                ha="right" if col == 0 else "right", va="center",
                fontweight="bold")

    ax.tick_params(axis="x", labelsize=8)

# ── Bottom row: quintile bar charts ──────────────────────────────────────────

for col, (metric, ylabel, has_zero, full_label) in enumerate(metrics):
    ax = axes_bot[col]

    x  = np.arange(len(Q_ORDER))
    w  = 0.33

    for qi, q in enumerate(Q_ORDER):
        sub_c = merged[merged["g_axial_quintile"] == q][f"{metric}_cmaes"]
        sub_t = merged[merged["g_axial_quintile"] == q][f"{metric}_tabicl"]
        mc, mt = sub_c.mean(), sub_t.mean()
        ec = sub_c.sem() if len(sub_c) > 1 else 0
        et = sub_t.sem() if len(sub_t) > 1 else 0

        ax.bar(qi - w/2, mc, w, color=C_CMAES,  alpha=0.85, label="CMA-ES"  if qi == 0 else "")
        ax.bar(qi + w/2, mt, w, color=C_TABICL, alpha=0.85, label="TabICL"  if qi == 0 else "")
        ax.errorbar(qi - w/2, mc, yerr=ec, fmt="none", color="0.3", lw=1.0, capsize=2)
        ax.errorbar(qi + w/2, mt, yerr=et, fmt="none", color="0.3", lw=1.0, capsize=2)

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

def overall_row(metric):
    mc = merged[f"{metric}_cmaes"].mean()
    mt = merged[f"{metric}_tabicl"].mean()
    return mc, mt

for col, (metric, ylabel, *_) in enumerate(metrics):
    ax = axes_bot[col]
    mc, mt = overall_row(metric)
    winner = "CMA-ES" if (metric != "ss_angle") else "TabICL"
    label  = f"mean  CMA-ES={mc:+.3f}  TabICL={mt:+.3f}"
    if metric == "chamfer_cm":
        label = f"mean  CMA-ES={mc:.3f}  TabICL={mt:.3f} cm"
    ax.set_title(ax.get_title() + f"   {label}", loc="left", fontsize=7.5, fontweight="bold")

# ── Legend (top, shared) ──────────────────────────────────────────────────────

legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.65",
           markeredgecolor=C_CMAES, markersize=7, label="CMA-ES"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.65",
           markeredgecolor=C_TABICL, markersize=7, label="TabICL"),
]
q_patches = [mpatches.Patch(color=Q_COLORS[q], alpha=0.7, label=f"{q} band") for q in Q_ORDER]
fig.legend(handles=legend_elements + q_patches,
           loc="upper center", ncol=7, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 0.995))

fig.suptitle("Structural similarity: CMA-ES vs TabICL (Scenario 2, N=20 test cases)",
             y=1.002, fontsize=10.5, fontweight="bold")

plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_PATH}")
