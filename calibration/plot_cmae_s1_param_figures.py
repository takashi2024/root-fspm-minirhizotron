"""
plot_cmae_s1_param_figures.py

Two diagnostic figures for CMA-ES Scenario 1 (all 8 params calibrated):
  1. True vs Predicted scatter (2 rows × 4 cols: arch params top, elong params bottom)
  2. True g_axial (log) vs |Error| scatter (same layout)

Output:
  calibration/figures/actual_vs_pred_cmae_Scen1_all_unknown.png
  calibration/figures/gaxial_vs_error_cmae_Scen1_all_unknown.png
"""

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(CALIB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TC_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
tc_df  = pd.read_csv(TC_CSV)

# ── Collect S1 results ────────────────────────────────────────────────────────

ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]
ELONG_PARAMS = ["elongation_rate_min", "elongation_rate_max",
                "elongation_logistic_k", "elongation_logistic_t50"]
ALL_PARAMS = ARCH_PARAMS + ELONG_PARAMS

PARAM_LABELS = {
    "branch_angle_order1":   "branch_angle_order1",
    "g_axial":               "g_axial",
    "branching_prob_order1": "branching_prob_order1",
    "branching_prob_order2": "branching_prob_order2",
    "elongation_rate_min":   "elong_rate_min",
    "elongation_rate_max":   "elong_rate_max",
    "elongation_logistic_k": "elong_logistic_k",
    "elongation_logistic_t50": "elong_logistic_t50",
}

rows = []
for _, tc in tc_df.iterrows():
    cfg, sid = tc["config"], int(tc["sample_id"])
    tag  = f"tc_{cfg}_{sid}_s1"
    best = os.path.join(CALIB_DIR, "experiments", tag, "best_params.json")
    if not os.path.isfile(best):
        continue
    with open(best) as f:
        bp = json.load(f)
    row = {"config": cfg, "sample_id": sid,
           "g_axial_quintile": tc["g_axial_quintile"],
           "true_g_axial": tc["g_axial"],
           "windows_with_roots": tc["windows_with_roots"]}
    for p in ALL_PARAMS:
        row[f"true_{p}"]  = float(tc[p]) if p in tc.index else np.nan
        row[f"pred_{p}"]  = float(bp[p]) if p in bp else np.nan
        row[f"abserr_{p}"] = abs(row[f"pred_{p}"] - row[f"true_{p}"])
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} S1 cases")

# ── Quintile colour map (green→red) ──────────────────────────────────────────

Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_COLORS = {"Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf",
            "Q4": "#fdae61", "Q5": "#d7191c"}

def q_color(q): return Q_COLORS.get(q, "gray")

# ── Figure 1: True vs Predicted ───────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8.5))
fig1.suptitle("CMA-ES [Scen1: all unknown] — True vs Predicted (N=20)",
              fontsize=12, fontweight="bold")

for row_idx, param_set in enumerate([ELONG_PARAMS, ARCH_PARAMS]):
    for col_idx, param in enumerate(param_set):
        ax = axes1[row_idx, col_idx]

        true_vals = df[f"true_{param}"].values
        pred_vals = df[f"pred_{param}"].values
        mae = np.nanmean(np.abs(pred_vals - true_vals))

        # diagonal reference
        lo = min(np.nanmin(true_vals), np.nanmin(pred_vals)) * 0.95
        hi = max(np.nanmax(true_vals), np.nanmax(pred_vals)) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.6)

        for _, r in df.iterrows():
            ax.scatter(r[f"true_{param}"], r[f"pred_{param}"],
                       color=q_color(r["g_axial_quintile"]),
                       s=70, edgecolors="0.3", linewidths=0.5, zorder=3)

        mae_str = f"{mae:.4g}" if param not in ["branch_angle_order1"] else f"{mae:.3g}"
        ax.set_title(f"{PARAM_LABELS[param]}\nMAE={mae_str}", fontsize=9)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, lw=0.4, alpha=0.5)

# legend
legend_handles = [mpatches.Patch(color=Q_COLORS[q],
                  label=f"{q}" + (" (low g)" if q=="Q1" else " (high g)" if q=="Q5" else ""))
                  for q in Q_ORDER]
axes1[0, 0].legend(handles=legend_handles, title="g_axial",
                   fontsize=7.5, title_fontsize=8, framealpha=0.8)

fig1.tight_layout()
out1 = os.path.join(FIG_DIR, "actual_vs_pred_cmae_Scen1_all_unknown.png")
fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out1}")

# ── Figure 2: g_axial vs |Error| ─────────────────────────────────────────────

# Reserve right margin for colorbar
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8.5))
fig2.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.08,
                     hspace=0.40, wspace=0.35)
fig2.suptitle("CMA-ES [Scen1: all unknown] — g_axial vs Recovery Error",
              fontsize=12, fontweight="bold")

# colormap for obs windows (viridis)
win_vals = df["windows_with_roots"].values
vmin, vmax = win_vals.min(), win_vals.max()
cmap = cm.viridis

for row_idx, param_set in enumerate([ELONG_PARAMS, ARCH_PARAMS]):
    for col_idx, param in enumerate(param_set):
        ax = axes2[row_idx, col_idx]

        mae = np.nanmean(df[f"abserr_{param}"].values)

        for i, r in df.iterrows():
            fill_c = cmap((r["windows_with_roots"] - vmin) / max(vmax - vmin, 1))
            ring_c = q_color(r["g_axial_quintile"])
            ax.scatter(r["true_g_axial"], r[f"abserr_{param}"],
                       color=fill_c, s=90,
                       edgecolors=ring_c, linewidths=1.5, zorder=3)

        mae_str = f"{mae:.4g}" if param != "branch_angle_order1" else f"{mae:.3g}"
        ax.set_title(f"{PARAM_LABELS[param]}\nmean={mae_str}", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("True g_axial (log)", fontsize=8)
        ax.set_ylabel("|Error|", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, lw=0.4, alpha=0.5, which="both")

# Colorbar — dedicated axes on the right, vertically centered across both rows
cbar_ax = fig2.add_axes([0.905, 0.12, 0.018, 0.70])   # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig2.colorbar(sm, cax=cbar_ax)
cbar.set_label("obs windows", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Quintile ring legend — inside top-left panel, upper-right corner
ring_handles = [Line2D([0],[0], marker="o", color="w",
                markerfacecolor="0.7", markeredgecolor=Q_COLORS[q],
                markersize=8, markeredgewidth=1.5,
                label=f"{q}" + (" (low g)" if q=="Q1" else " (high g)" if q=="Q5" else ""))
                for q in Q_ORDER]
axes2[0, 0].legend(handles=ring_handles, title="Quintile",
                   loc="upper right", fontsize=7.5, title_fontsize=8, framealpha=0.8)

out2 = os.path.join(FIG_DIR, "gaxial_vs_error_cmae_Scen1_all_unknown.png")
fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out2}")
