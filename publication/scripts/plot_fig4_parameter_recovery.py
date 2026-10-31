"""
plot_fig4_parameter_recovery.py

Figure 4 — Parameter recovery: true vs predicted for 4 architectural parameters, Scenario 2.
Layout: 6 rows × 4 cols (k=1/7/14, each with CMA-ES + TabICL row).
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
SBI     = os.path.join(ROOT, "sbi", "data", "combined")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "fig4_parameter_recovery.png")
TC_CSV  = os.path.join(CALIB, "test_cases_selected_v2.csv")

ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]
PARAM_LABELS = {
    "branch_angle_order1":   "θ",
    "g_axial":               "$g_{axial}$",
    "branching_prob_order1": "$p_1$",
    "branching_prob_order2": "$p_2$",
}
Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_COLORS = {"Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf",
            "Q4": "#fdae61", "Q5": "#d7191c"}
Q_LABELS = {"Q1": "Q1 (slow $g_{axial}$)", "Q2": "Q2", "Q3": "Q3",
            "Q4": "Q4", "Q5": "Q5 (fast $g_{axial}$)"}

K_CONFIGS = [
    {"k": 1,  "cma_suffix": "s2",
     "tab_csv": os.path.join(SBI, "test_case_eval_multiseed_v2_k1",  "results_tabicl_ms_v2_k1.csv")},
    {"k": 7,  "cma_suffix": "s2ms",
     "tab_csv": os.path.join(SBI, "test_case_eval_multiseed_v2",     "results_tabicl_multiseed_v2.csv")},
    {"k": 14, "cma_suffix": "s2ms_k14",
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
        row = {"config": r["config"], "sample_id": int(r["sample_id"]),
               "g_axial_quintile": r["g_axial_quintile"]}
        for p in ARCH_PARAMS:
            row[f"true_{p}"] = float(r[p])
            row[f"pred_{p}"] = float(bp[p]) if p in bp else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def load_tabicl(tc, csv_path):
    raw = pd.read_csv(csv_path)
    sub = raw[raw["scenario"] == "Scen2: elong known"].copy()
    rows = []
    for (sid, cfg), grp in sub.groupby(["sample_id", "config"]):
        tc_row = tc[(tc["sample_id"] == int(sid)) & (tc["config"] == cfg)]
        if len(tc_row) == 0:
            continue
        row = {"config": cfg, "sample_id": int(sid),
               "g_axial_quintile": tc_row["g_axial_quintile"].values[0]}
        for p in ARCH_PARAMS:
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
    ROW_CONFIGS.append((cmae_df, f"Inverse modelling  —  Scenario 2, $k$={k}"))
    ROW_CONFIGS.append((tab_df,  f"ML-based inference  —  Scenario 2, $k$={k}"))

# ── Plot 6×4 ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(6, 4, figsize=(14, 23))
fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06,
                    hspace=0.55, wspace=0.40)

for row_idx, (df, row_label) in enumerate(ROW_CONFIGS):
    for col_idx, param in enumerate(ARCH_PARAMS):
        ax = axes[row_idx, col_idx]

        tv = df[f"true_{param}"].values.astype(float)
        pv = df[f"pred_{param}"].values.astype(float)
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
                       edgecolors="0.25", linewidths=0.5, s=55, zorder=3)

        r2 = r2_score(tv, pv)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9) if col_idx == 0 else ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.35, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{PARAM_LABELS[param]}\n$R^2$={r2:+.2f}", fontsize=10, pad=4)

    axes[row_idx, 0].annotate(
        row_label,
        xy=(0.0, 1.22), xycoords="axes fraction",
        fontsize=9, fontweight="bold", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="#eef2f7", ec="0.7", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────

legend_patches = [
    mpatches.Patch(facecolor=Q_COLORS[q], edgecolor="0.25",
                   linewidth=0.5, label=Q_LABELS[q])
    for q in Q_ORDER
]
fig.legend(handles=legend_patches, title="$g_{axial}$ quintile",
           loc="lower center", ncol=5, fontsize=9, title_fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.005))

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
