"""
plot_fig3_parameter_recovery.py

Figure 3 (2 mm skin depth) — Parameter recovery: true vs predicted for all 8 parameters.
Scenario 1: all 8 parameters unknown (elongation NOT fixed).
Layout: 2 rows × 8 columns
  Row 1: CMA-ES S1MS (k=7, 0.2 cm skin depth)
  Row 2: TabICL-MS (k=7, 0.2 cm skin depth, all parameters unknown)
Points coloured by g_axial quintile (Q1=slow growth → Q5=fast growth).
R² annotated on each panel.

Run from project root:
  python3 publication/scripts/plot_fig3_parameter_recovery.py
Output: publication/figures/fig3_parameter_recovery.png
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
FIG_OUT = os.path.join(ROOT, "publication", "figures", "fig3_parameter_recovery.png")

TC_CSV  = os.path.join(CALIB, "test_cases_selected_v2.csv")
TAB_CSV = os.path.join(SBI, "data", "combined",
                        "test_case_eval_multiseed_v2",
                        "results_tabicl_multiseed_v2.csv")

# All 8 parameters: elongation first, then architectural
ALL_PARAMS = [
    "elongation_rate_min",
    "elongation_rate_max",
    "elongation_logistic_k",
    "elongation_logistic_t50",
    "branch_angle_order1",
    "g_axial",
    "branching_prob_order1",
    "branching_prob_order2",
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


def r2_score(true_vals, pred_vals):
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[mask], pred_vals[mask]
    if len(t) < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ── Load CMA-ES S1MS (k=7, skin_depth=0.2 cm) ────────────────────────────────

tc = pd.read_csv(TC_CSV)
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
print(f"CMA-ES S1MS v2: {len(cmae_df)} test cases loaded")

# ── Load TabICL-MS S1 (k=7, all params unknown) ───────────────────────────────

tab_raw = pd.read_csv(TAB_CSV)
tab_s1  = tab_raw[tab_raw["scenario"] == "Scen1: all unknown"].copy()

tab_rows = []
for (sid, cfg), grp in tab_s1.groupby(["sample_id", "config"]):
    row = {"config": cfg, "sample_id": int(sid)}
    tc_row = tc[(tc["sample_id"] == int(sid)) & (tc["config"] == cfg)]
    if len(tc_row) == 0:
        continue
    row["g_axial_quintile"] = tc_row["g_axial_quintile"].values[0]
    for p in ALL_PARAMS:
        prow = grp[grp["parameter"] == p]
        if len(prow) == 0:
            row[f"true_{p}"] = float("nan")
            row[f"pred_{p}"] = float("nan")
            continue
        row[f"true_{p}"] = float(prow["true_value"].values[0])
        row[f"pred_{p}"] = float(prow["pred_value"].values[0])
    tab_rows.append(row)

tab_df = pd.DataFrame(tab_rows)
print(f"TabICL-MS S1 v2: {len(tab_df)} test cases loaded")

# ── Plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 8, figsize=(24, 9.5))
fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.13,
                    hspace=0.52, wspace=0.42)

ROW_CONFIGS = [
    (cmae_df, "Inverse modelling  —  Scenario 1, $k$=7"),
    (tab_df,  "ML-based inference  —  Scenario 1, $k$=7"),
]

for row_idx, (df, row_label) in enumerate(ROW_CONFIGS):
    for col_idx, param in enumerate(ALL_PARAMS):
        ax = axes[row_idx, col_idx]

        true_col = f"true_{param}"
        pred_col = f"pred_{param}"

        if true_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue

        true_vals = df[true_col].values.astype(float)
        pred_vals = df[pred_col].values.astype(float)

        mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        tv, pv = true_vals[mask], pred_vals[mask]

        lo = min(tv.min(), pv.min())
        hi = max(tv.max(), pv.max())
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=1.0, ls="--", zorder=1)

        quintiles = df["g_axial_quintile"].values
        for i, r in df.iterrows():
            if np.isnan(true_vals[df.index.get_loc(i)]) or np.isnan(pred_vals[df.index.get_loc(i)]):
                continue
            ax.scatter(true_vals[df.index.get_loc(i)], pred_vals[df.index.get_loc(i)],
                       color=Q_COLORS.get(r["g_axial_quintile"], "gray"),
                       edgecolors="0.25", linewidths=0.5, s=50, zorder=3)

        r2 = r2_score(true_vals, pred_vals)
        r2_str = f"$R^2$ = {r2:+.2f}"

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True", fontsize=8.5)
        ax.set_ylabel("Predicted", fontsize=8.5) if col_idx == 0 else ax.set_ylabel("")
        ax.tick_params(labelsize=7.5)
        ax.grid(True, lw=0.35, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(f"{PARAM_LABELS[param]}\n{r2_str}", fontsize=9.5, pad=4)

    axes[row_idx, 0].annotate(
        row_label,
        xy=(0.0, 1.22), xycoords="axes fraction",
        fontsize=10, fontweight="bold", ha="left", va="bottom",
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

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
