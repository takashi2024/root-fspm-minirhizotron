"""
plot_figS2_parameter_recovery_k1_k14.py

Figure S2 — Parameter recovery at k=1 and k=14, Scenario 1 (all 8 params).
Layout: 4 rows x 8 cols
  Rows 0-1: k=1  (CMA-ES, TabICL)
  Rows 2-3: k=14 (CMA-ES, TabICL)

Run from project root:
  python3 publication/scripts/plot_figS2_parameter_recovery_k1_k14.py
Output: publication/figures/figS2_parameter_recovery_k1_k14.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB   = os.path.join(ROOT, "calibration")
SBI     = os.path.join(ROOT, "sbi", "data", "combined")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "figS2_parameter_recovery_k1_k14.png")
TC_CSV  = os.path.join(CALIB, "test_cases_selected_v2.csv")

ALL_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
    "branch_angle_order1", "g_axial",
    "branching_prob_order1", "branching_prob_order2",
]
PARAM_LABELS = {
    "elongation_rate_min":    "$r_{min}$",
    "elongation_rate_max":    "$r_{max}$",
    "elongation_logistic_k":  "$k_e$",
    "elongation_logistic_t50":"$t_{50}$",
    "branch_angle_order1":    "theta",
    "g_axial":                "$g_{axial}$",
    "branching_prob_order1":  "$p_1$",
    "branching_prob_order2":  "$p_2$",
}
CFG_ORDER  = ["A", "B2", "C", "D2"]
CFG_COLORS = {"A": "#2166ac", "B2": "#4dac26", "C": "#d01c8b", "D2": "#f1a340"}
CFG_LABEL  = {"A": "Config A", "B2": "Config B", "C": "Config C", "D2": "Config D"}

K_CONFIGS = [
    {"k": 1,  "cma_suffix": "s1",
     "tab_csv": os.path.join(SBI, "test_case_eval_multiseed_v2_k1",  "results_tabicl_ms_v2_k1.csv")},
    {"k": 14, "cma_suffix": "s1ms_k14",
     "tab_csv": os.path.join(SBI, "test_case_eval_multiseed_v2_k14", "results_tabicl_ms_v2_k14.csv")},
]


def r2_score(tv, pv):
    mask = ~(np.isnan(tv) | np.isnan(pv))
    t, p = tv[mask], pv[mask]
    if len(t) < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_cmae(tc, suffix):
    rows = []
    for _, r in tc.iterrows():
        tag  = f"tc_v2_{r['config']}_{int(r['sample_id'])}_{suffix}"
        path = os.path.join(CALIB, "experiments", tag, "best_params.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            bp = json.load(f)
        row = {"config": r["config"], "sample_id": int(r["sample_id"])}
        for p in ALL_PARAMS:
            row[f"true_{p}"] = float(r[p]) if p in r.index else float("nan")
            row[f"pred_{p}"] = float(bp[p]) if p in bp else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def load_tabicl(tc, csv_path):
    raw = pd.read_csv(csv_path)
    sub = raw[raw["scenario"] == "Scen1: all unknown"].copy()
    rows = []
    for (sid, cfg), grp in sub.groupby(["sample_id", "config"]):
        tc_row = tc[(tc["sample_id"] == int(sid)) & (tc["config"] == cfg)]
        if len(tc_row) == 0:
            continue
        row = {"config": cfg, "sample_id": int(sid)}
        for p in ALL_PARAMS:
            prow = grp[grp["parameter"] == p]
            row[f"true_{p}"] = float(prow["true_value"].values[0]) if len(prow) else float("nan")
            row[f"pred_{p}"] = float(prow["pred_value"].values[0]) if len(prow) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


tc = pd.read_csv(TC_CSV)

ROW_CONFIGS = []
for cfg in K_CONFIGS:
    k = cfg["k"]
    cmae_df = load_cmae(tc, cfg["cma_suffix"])
    tab_df  = load_tabicl(tc, cfg["tab_csv"])
    print(f"k={k}: CMA-ES={len(cmae_df)}, TabICL={len(tab_df)}")
    ROW_CONFIGS.append((cmae_df, f"Inverse modelling  --  Scenario 1, $k$={k}"))
    ROW_CONFIGS.append((tab_df,  f"ML-based inference  --  Scenario 1, $k$={k}"))

# Plot 4x8

fig, axes = plt.subplots(4, 8, figsize=(26, 17))
fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.08,
                    hspace=0.45, wspace=0.35)

for row_idx, (df, row_label) in enumerate(ROW_CONFIGS):
    for col_idx, param in enumerate(ALL_PARAMS):
        ax = axes[row_idx, col_idx]
        true_col, pred_col = f"true_{param}", f"pred_{param}"

        if true_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue

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
                       color=CFG_COLORS.get(r["config"], "gray"),
                       edgecolors="0.25", linewidths=0.5, s=45, zorder=3)

        r2 = r2_score(tv, pv)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        lct = ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        tks = [t for t in lct.tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9]
        ax.set_xticks(tks)
        ax.set_yticks(tks)
        ax.set_xlabel("True", fontsize=11) if row_idx % 2 == 1 else ax.set_xlabel("")
        ax.set_ylabel("Predicted", fontsize=11) if col_idx == 0 else ax.set_ylabel("")
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{PARAM_LABELS[param]}\n$R^2$={r2:+.2f}", fontsize=12, pad=3)

    axes[row_idx, 0].annotate(
        row_label,
        xy=(0.0, 1.24), xycoords="axes fraction",
        fontsize=12, fontweight="bold", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.28", fc="#eef2f7", ec="0.7", lw=0.8))

# Legend

legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CFG_COLORS[c], markeredgecolor="0.25",
           markeredgewidth=0.6, markersize=11, label=CFG_LABEL[c])
    for c in CFG_ORDER
]
fig.legend(handles=legend_handles, title="Planting configuration",
           loc="lower center", ncol=4, fontsize=12, title_fontsize=12,
           frameon=False, bbox_to_anchor=(0.52, 0.005))

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved -> {FIG_OUT}")
