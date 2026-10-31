"""
plot_fig3_parameter_recovery.py

Figure 3 — Parameter recovery: true vs predicted for all 8 parameters.
Scenario 1: all 8 parameters unknown (elongation NOT fixed), k=7.
Layout: 4 rows × 4 cols with parameter group separator
  Rows 0–1: elongation parameters (r_min, r_max, k_e, t_50)
  Separator row: "Architectural parameters" label
  Rows 2–3: architectural parameters (θ, g_axial, p_1, p_2)
  Row 0 / Row 2: CMA-ES (inverse modelling)
  Row 1 / Row 3: TabICL-MS (ML-based inference)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB   = os.path.join(ROOT, "calibration")
SBI     = os.path.join(ROOT, "sbi", "data", "combined")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "fig3_parameter_recovery.png")
TC_CSV  = os.path.join(CALIB, "test_cases_selected_v2.csv")

ELONG_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
]
ARCH_PARAMS = [
    "branch_angle_order1", "g_axial",
    "branching_prob_order1", "branching_prob_order2",
]
PARAM_LABELS = {
    "elongation_rate_min":    "$r_{min}$",
    "elongation_rate_max":    "$r_{max}$",
    "elongation_logistic_k":  "$k_e$",
    "elongation_logistic_t50":"$t_{50}$",
    "branch_angle_order1":    "θ",
    "g_axial":                "$g_{axial}$",
    "branching_prob_order1":  "$p_1$",
    "branching_prob_order2":  "$p_2$",
}
Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_COLORS = {"Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf",
            "Q4": "#fdae61", "Q5": "#d7191c"}
Q_LABELS = {"Q1": "Q1 (slow $g_{axial}$)", "Q2": "Q2", "Q3": "Q3",
            "Q4": "Q4", "Q5": "Q5 (fast $g_{axial}$)"}


def r2_score(tv, pv):
    mask = ~(np.isnan(tv) | np.isnan(pv))
    t, p = tv[mask], pv[mask]
    if len(t) < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


tc = pd.read_csv(TC_CSV)

# ── Load CMA-ES S1MS (k=7) ────────────────────────────────────────────────────

ALL_PARAMS = ELONG_PARAMS + ARCH_PARAMS
cmae_rows = []
for _, r in tc.iterrows():
    tag  = f"tc_v2_{r['config']}_{int(r['sample_id'])}_s1ms"
    path = os.path.join(CALIB, "experiments", tag, "best_params.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        bp = json.load(f)
    row = {"config": r["config"], "sample_id": int(r["sample_id"]),
           "g_axial_quintile": r["g_axial_quintile"]}
    for p in ALL_PARAMS:
        row[f"true_{p}"] = float(r[p]) if p in r.index else float("nan")
        row[f"pred_{p}"] = float(bp[p]) if p in bp else float("nan")
    cmae_rows.append(row)
cmae_df = pd.DataFrame(cmae_rows)
print(f"CMA-ES S1MS k=7: {len(cmae_df)}")

# ── Load TabICL-MS S1 (k=7) ───────────────────────────────────────────────────

tab_raw = pd.read_csv(os.path.join(SBI, "test_case_eval_multiseed_v2",
                                   "results_tabicl_multiseed_v2.csv"))
tab_s1  = tab_raw[tab_raw["scenario"] == "Scen1: all unknown"].copy()
tab_rows = []
for (sid, cfg), grp in tab_s1.groupby(["sample_id", "config"]):
    tc_row = tc[(tc["sample_id"] == int(sid)) & (tc["config"] == cfg)]
    if len(tc_row) == 0:
        continue
    row = {"config": cfg, "sample_id": int(sid),
           "g_axial_quintile": tc_row["g_axial_quintile"].values[0]}
    for p in ALL_PARAMS:
        prow = grp[grp["parameter"] == p]
        row[f"true_{p}"] = float(prow["true_value"].values[0]) if len(prow) else float("nan")
        row[f"pred_{p}"] = float(prow["pred_value"].values[0]) if len(prow) else float("nan")
    tab_rows.append(row)
tab_df = pd.DataFrame(tab_rows)
print(f"TabICL-MS S1 k=7: {len(tab_df)}")

# ── Layout: GridSpec with 5 rows (spacer at row 2) ───────────────────────────
# height_ratios: [elong-cmae, elong-tab, spacer, arch-cmae, arch-tab]

fig = plt.figure(figsize=(14, 20))
# 6-row GridSpec: row 0 = elongation header, rows 1-2 = elongation panels,
#                 row 3 = architectural header, rows 4-5 = architectural panels
gs  = GridSpec(6, 4, figure=fig,
               height_ratios=[0.12, 1, 1, 0.12, 1, 1],
               hspace=0.80, wspace=0.45,
               left=0.07, right=0.97, top=0.97, bottom=0.07)

def add_group_header(row, label):
    ax = fig.add_subplot(gs[row, :])
    ax.axis("off")
    ax.text(0.5, 0.5, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#e8eef5", ec="0.65", lw=1.0))

add_group_header(0, "Elongation parameters")
add_group_header(3, "Architectural parameters")

# Group header for elongation (above row 0, in figure coords)
# Placed via the first axis annotation instead of fig.text for simplicity

GROUP_BLOCKS = [
    (1, ELONG_PARAMS,
     "Inverse modelling  —  Scenario 1, $k$=7",
     "ML-based inference  —  Scenario 1, $k$=7"),
    (4, ARCH_PARAMS,
     "Inverse modelling  —  Scenario 1, $k$=7",
     "ML-based inference  —  Scenario 1, $k$=7"),
]

for gs_start, params, cmae_label, tab_label in GROUP_BLOCKS:
    for method_offset, (df, row_label) in enumerate(
            [(cmae_df, cmae_label), (tab_df, tab_label)]):
        gs_row = gs_start + method_offset

        for col_idx, param in enumerate(params):
            ax = fig.add_subplot(gs[gs_row, col_idx])

            true_col, pred_col = f"true_{param}", f"pred_{param}"
            tv = df[true_col].values.astype(float)
            pv = df[pred_col].values.astype(float)
            mask = ~(np.isnan(tv) | np.isnan(pv))

            lo = min(tv[mask].min(), pv[mask].min())
            hi = max(tv[mask].max(), pv[mask].max())
            pad = (hi - lo) * 0.08
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=1.0, ls="--", zorder=1)

            for i, r in df.iterrows():
                idx = df.index.get_loc(i)
                if np.isnan(tv[idx]) or np.isnan(pv[idx]):
                    continue
                ax.scatter(tv[idx], pv[idx],
                           color=Q_COLORS.get(r["g_axial_quintile"], "gray"),
                           edgecolors="0.25", linewidths=0.5, s=50, zorder=3)

            r2 = r2_score(tv, pv)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("True", fontsize=11)
            ax.set_ylabel("Predicted", fontsize=11) if col_idx == 0 else ax.set_ylabel("")
            ax.tick_params(labelsize=10)
            ax.grid(True, lw=0.35, alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"{PARAM_LABELS[param]}\n$R^2$={r2:+.2f}", fontsize=13, pad=8)

            # Row annotation on leftmost column
            if col_idx == 0:
                ax.annotate(
                    row_label,
                    xy=(0.0, 1.32), xycoords="axes fraction",
                    fontsize=11, fontweight="bold", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.30", fc="#eef2f7", ec="0.7", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────

legend_patches = [
    mpatches.Patch(facecolor=Q_COLORS[q], edgecolor="0.25",
                   linewidth=0.5, label=Q_LABELS[q])
    for q in Q_ORDER
]
fig.legend(handles=legend_patches, title="$g_{axial}$ quintile",
           loc="lower center", ncol=5, fontsize=11, title_fontsize=11,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.01))

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
