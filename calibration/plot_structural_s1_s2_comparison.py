"""
plot_structural_s1_s2_comparison.py

Figure: CMA-ES vs TabICL structural metrics across Scenario 1 and Scenario 2.
Four conditions: CMA-ES S2, CMA-ES S1, TabICL S2, TabICL S1.

Layout (3 × 2):
  Top row    — per-case dot plots (20 cases, 4 series each)
  Bottom row — quintile-mean bar charts ± SE

Output: calibration/figures/structural_s1_s2_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(CALIB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

s2 = pd.read_csv(os.path.join(CALIB_DIR, "comparison_structure_metrics.csv"))
s1 = pd.read_csv(os.path.join(CALIB_DIR, "comparison_structure_metrics_s1.csv"))
s2["scenario"] = "S2"
s1["scenario"] = "S1"
df = pd.concat([s2, s1], ignore_index=True)

# Sort cases by quintile then g_axial (consistent with S2 figure)
case_order = (s2[s2["method"] == "CMA-ES"]
              .sort_values(["g_axial_quintile", "true_g_axial"])
              [["config", "sample_id"]].reset_index(drop=True))
case_order["label"] = case_order["config"] + "\n" + case_order["sample_id"].astype(str)
case_order["rank"]  = case_order.index

df = df.merge(case_order[["config", "sample_id", "rank", "label"]], on=["config", "sample_id"])
df = df.sort_values("rank")

N = len(case_order)

# ── Style ─────────────────────────────────────────────────────────────────────

CONDITIONS = [
    ("CMA-ES", "S2", "#1B5E9E", "s", "CMA-ES  S2"),
    ("CMA-ES", "S1", "#74B3E3", "s", "CMA-ES  S1"),
    ("TabICL", "S2", "#B84500", "o", "TabICL  S2"),
    ("TabICL", "S1", "#F4A460", "o", "TabICL  S1"),
]

Q_COLORS  = {"Q1": "#4393C3", "Q2": "#92C5DE", "Q3": "#B8D97C",
             "Q4": "#F4A582", "Q5": "#D6604D"}
Q_ORDER   = ["Q1", "Q2", "Q3", "Q4", "Q5"]

BAR_COLORS = {
    ("CMA-ES","S2"): "#1B5E9E",
    ("CMA-ES","S1"): "#74B3E3",
    ("TabICL","S2"): "#B84500",
    ("TabICL","S1"): "#F4A460",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

metrics = [
    ("ss_rld",    r"$SS_{RLD}$",    True),
    ("ss_angle",  r"$SS_{angle}$",  True),
    ("chamfer_cm","Chamfer (cm)",   False),
]

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.30,
                        left=0.08, right=0.97, top=0.91, bottom=0.08)
axes_top = [fig.add_subplot(gs[0, c]) for c in range(3)]
axes_bot = [fig.add_subplot(gs[1, c]) for c in range(3)]

# ── Top row: per-case dot plots ───────────────────────────────────────────────

# vertical offsets to separate 4 conditions within each case row
Y_OFFSETS = [-0.27, -0.09, +0.09, +0.27]

for col, (metric, ylabel, has_zero) in enumerate(metrics):
    ax = axes_top[col]
    ys = np.arange(N)

    # clip extreme outliers so they don't compress the main distribution
    clip_lo = df[metric].quantile(0.02)
    clip_hi = df[metric].quantile(0.98)

    for (method, scen, color, marker, _), yoff in zip(CONDITIONS, Y_OFFSETS):
        sub = df[(df["method"] == method) & (df["scenario"] == scen)].sort_values("rank")
        vals = sub[metric].values
        clipped = np.clip(vals, clip_lo, clip_hi)
        is_clipped = (vals < clip_lo) | (vals > clip_hi)

        for i in range(N):
            qc = Q_COLORS[sub.iloc[i]["g_axial_quintile"]]
            ax.scatter(clipped[i], ys[i] + yoff,
                       marker=marker, s=22,
                       color=qc, edgecolors=color, linewidths=0.8,
                       zorder=3,
                       alpha=0.5 if is_clipped[i] else 1.0)
            # triangle marker at edge for clipped points
            if is_clipped[i]:
                edge_x = clip_hi if vals[i] > clip_hi else clip_lo
                ax.scatter(edge_x, ys[i] + yoff, marker=">", s=18,
                           color=color, zorder=4, alpha=0.7)

        # Connect S1-S2 pairs within same method (thin line at midpoint y)
        if scen == "S1":
            s2_sub = df[(df["method"] == method) & (df["scenario"] == "S2")].sort_values("rank")
            s2_clipped = np.clip(s2_sub[metric].values, clip_lo, clip_hi)
            for i in range(N):
                ax.plot([s2_clipped[i], clipped[i]],
                        [ys[i] + yoff, ys[i] + yoff],
                        lw=0.7, color=color, alpha=0.30, zorder=1)

    if has_zero:
        ax.axvline(0, color="0.50", lw=0.8, ls="--", zorder=0)

    # Quintile bands
    for q in Q_ORDER:
        rows_q = case_order[case_order["label"].str.startswith(
            df[df["g_axial_quintile"] == q]["config"].iloc[0]
            if len(df[df["g_axial_quintile"] == q]) else "")].index.tolist() \
            if False else \
            [i for i, r in case_order.iterrows()
             if df[(df["config"] == r["config"]) &
                   (df["sample_id"] == r["sample_id"])]["g_axial_quintile"].iloc[0] == q]
        if not rows_q:
            continue
        ax.axhspan(min(rows_q) - 0.45, max(rows_q) + 0.45,
                   alpha=0.07, color=Q_COLORS[q], zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels(case_order["label"].values, fontsize=6.2)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(["(a)", "(b)", "(c)"][col], loc="left", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

# ── Bottom row: quintile bar charts ──────────────────────────────────────────

n_cond = 4
w_bar  = 0.18
offsets_bar = np.array([-1.5, -0.5, 0.5, 1.5]) * w_bar

for col, (metric, ylabel, has_zero) in enumerate(metrics):
    ax = axes_bot[col]
    x  = np.arange(len(Q_ORDER))

    for ci, (method, scen, color, _, label) in enumerate(CONDITIONS):
        sub_all = df[(df["method"] == method) & (df["scenario"] == scen)]
        means, sems = [], []
        for q in Q_ORDER:
            sub_q = sub_all[sub_all["g_axial_quintile"] == q][metric]
            means.append(sub_q.mean())
            sems.append(sub_q.sem() if len(sub_q) > 1 else 0)

        ax.bar(x + offsets_bar[ci], means, w_bar,
               color=color, alpha=0.85,
               label=label if col == 0 else "")
        ax.errorbar(x + offsets_bar[ci], means, yerr=sems,
                    fmt="none", color="0.3", lw=0.9, capsize=2)

    if has_zero:
        ax.axhline(0, color="0.45", lw=0.8, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(Q_ORDER, fontsize=8)
    ax.set_xlabel("g_axial quintile (Q1=slow → Q5=fast)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(["(d)", "(e)", "(f)"][col], loc="left", fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    if col == 0:
        ax.legend(fontsize=7.5, frameon=False, ncol=2)

# ── Overall means in panel titles ─────────────────────────────────────────────

for col, (metric, *_) in enumerate(metrics):
    ax = axes_bot[col]
    parts = []
    for method, scen, color, _, _ in CONDITIONS:
        sub = df[(df["method"] == method) & (df["scenario"] == scen)]
        v = sub[metric].mean()
        fmt = f"{v:+.3f}" if metric != "chamfer_cm" else f"{v:.3f}"
        parts.append(f"{method}/{scen}={fmt}")
    ax.set_title(["(d)", "(e)", "(f)"][col] + "   " + "  ".join(parts),
                 loc="left", fontsize=6.8, fontweight="bold")

# ── Shared legend (top) ───────────────────────────────────────────────────────

legend_elements = []
for method, scen, color, marker, label in CONDITIONS:
    legend_elements.append(
        Line2D([0], [0], marker=marker, color="w",
               markerfacecolor="0.65", markeredgecolor=color,
               markersize=7, label=label))
legend_elements += [
    mpatches.Patch(color=Q_COLORS[q], alpha=0.7, label=f"{q}") for q in Q_ORDER
]

fig.legend(handles=legend_elements, loc="upper center", ncol=9,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))

fig.suptitle(
    "Structural similarity: CMA-ES vs TabICL — Scenario 1 (all params) vs Scenario 2 (elongation known)  [N=20]",
    y=1.002, fontsize=10.5, fontweight="bold")

out = os.path.join(FIG_DIR, "structural_s1_s2_comparison.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
