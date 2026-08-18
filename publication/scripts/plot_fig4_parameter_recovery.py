"""
plot_fig4_parameter_recovery.py

Figure 4 (2 mm skin depth) — Parameter recovery: true vs predicted for 4 architectural parameters.
Scenario 2: elongation known (4 arch params calibrated).
Layout: 2 rows × 4 columns
  Row 1: CMA-ES S2MS (k=7, 0.2 cm skin depth)
  Row 2: TabICL-MS (k=7, 0.2 cm skin depth, elongation known)
Points coloured by g_axial quintile (Q1=slow growth → Q5=fast growth).

Run from project root:
  python3 publication/scripts/plot_fig4_parameter_recovery.py
Output: publication/figures/fig4_parameter_recovery.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB   = os.path.join(ROOT, "calibration")
SBI     = os.path.join(ROOT, "sbi")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "fig4_parameter_recovery.png")

TC_CSV  = os.path.join(CALIB, "test_cases_selected_v2.csv")
TAB_CSV = os.path.join(SBI, "data", "combined",
                        "test_case_eval_multiseed_v2",
                        "results_tabicl_multiseed_v2.csv")

ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

PARAM_LABELS = {
    "branch_angle_order1":   "Branch angle (°)",
    "g_axial":               "Growth conductance\n$g_{axial}$",
    "branching_prob_order1": "Branching probability\n$bp_{order1}$",
    "branching_prob_order2": "Branching probability\n$bp_{order2}$",
}

Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_COLORS = {"Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf",
            "Q4": "#fdae61", "Q5": "#d7191c"}
Q_LABELS = {"Q1": "Q1 (slow g)",  "Q2": "Q2", "Q3": "Q3",
            "Q4": "Q4", "Q5": "Q5 (fast g)"}

# ── Load CMA-ES S2MS (k=7, skin_depth=0.2 cm) ─────────────────────────────────

tc = pd.read_csv(TC_CSV)
cmae_rows = []
for _, r in tc.iterrows():
    tag  = f"tc_v2_{r['config']}_{int(r['sample_id'])}_s2ms"
    path = os.path.join(CALIB, "experiments", tag, "best_params.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        bp = json.load(f)
    row = {"config": r["config"], "sample_id": int(r["sample_id"]),
           "g_axial_quintile": r["g_axial_quintile"]}
    for p in ARCH_PARAMS:
        row[f"true_{p}"]  = float(r[p])
        row[f"pred_{p}"]  = float(bp[p]) if p in bp else float("nan")
    cmae_rows.append(row)

cmae_df = pd.DataFrame(cmae_rows)
print(f"CMA-ES S2MS v2: {len(cmae_df)} test cases loaded")

# ── Load TabICL-MS S2 (k=7, elongation known) ────────────────────────────────

tab_raw = pd.read_csv(TAB_CSV)
tab_s2  = tab_raw[tab_raw["scenario"] == "Scen2: elong known"].copy()

tab_rows = []
for (sid, cfg), grp in tab_s2.groupby(["sample_id", "config"]):
    row = {"config": cfg, "sample_id": int(sid)}
    tc_row = tc[(tc["sample_id"] == int(sid)) & (tc["config"] == cfg)]
    if len(tc_row) == 0:
        continue
    row["g_axial_quintile"] = tc_row["g_axial_quintile"].values[0]
    for p in ARCH_PARAMS:
        prow = grp[grp["parameter"] == p]
        if len(prow) == 0:
            continue
        row[f"true_{p}"] = float(prow["true_value"].values[0])
        row[f"pred_{p}"] = float(prow["pred_value"].values[0])
    tab_rows.append(row)

tab_df = pd.DataFrame(tab_rows)
print(f"TabICL-MS S2 v2: {len(tab_df)} test cases loaded")

# ── Plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 9.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.11,
                    hspace=0.52, wspace=0.35)

ROW_CONFIGS = [
    (cmae_df, "CMA-ES  —  S2, $k$=7  (elongation known, 0.2 cm skin depth)"),
    (tab_df,  "TabICL-MS  —  S2, $k$=7  (elongation known, 0.2 cm skin depth)"),
]

for row_idx, (df, row_label) in enumerate(ROW_CONFIGS):
    for col_idx, param in enumerate(ARCH_PARAMS):
        ax = axes[row_idx, col_idx]

        true_col = f"true_{param}"
        pred_col = f"pred_{param}"

        if true_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue

        true_vals = df[true_col].values
        pred_vals = df[pred_col].values
        mae = np.nanmean(np.abs(pred_vals - true_vals))

        lo = min(np.nanmin(true_vals), np.nanmin(pred_vals))
        hi = max(np.nanmax(true_vals), np.nanmax(pred_vals))
        pad = (hi - lo) * 0.07
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=1.0, ls="--", zorder=1)

        for _, r in df.iterrows():
            ax.scatter(r[true_col], r[pred_col],
                       color=Q_COLORS.get(r["g_axial_quintile"], "gray"),
                       edgecolors="0.25", linewidths=0.5, s=55, zorder=3)

        if param == "branch_angle_order1":
            mae_str = f"MAE = {mae:.1f}°"
        elif param == "g_axial":
            mae_str = f"MAE = {mae:.4f}"
        else:
            mae_str = f"MAE = {mae:.3f}"

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.35, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        title = f"{PARAM_LABELS[param]}\n{mae_str}"
        ax.set_title(title, fontsize=9, pad=4)

    ax0 = axes[row_idx, 0]
    ax0.set_ylabel("Predicted", fontsize=9)
    ax0.annotate(row_label,
                 xy=(0.0, 1.18), xycoords="axes fraction",
                 fontsize=9.5, fontweight="bold",
                 ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#eef2f7", ec="0.7", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────

legend_patches = [
    mpatches.Patch(facecolor=Q_COLORS[q], edgecolor="0.25",
                   linewidth=0.5, label=Q_LABELS[q])
    for q in Q_ORDER
]
fig.legend(handles=legend_patches, title="$g_{axial}$ quintile",
           loc="lower center", ncol=5, fontsize=9, title_fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.01))

fig.suptitle("Parameter recovery: true vs predicted  (N=20 test cases, Scenario 2 — elongation known, skin depth = 0.2 cm)",
             fontsize=12, y=0.97)

fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
