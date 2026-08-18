"""
plot_cmae_s2w_param_figures.py

True vs Predicted scatter for CMA-ES Scenario 2W (TabICL warm-start).
Mirrors the S2 figure style: 1 row x 4 cols, arch params only.

Output:
  calibration/figures/actual_vs_pred_cmae_Scen2W_tabicl_warmstart.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(CALIB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TC_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
tc_df  = pd.read_csv(TC_CSV)

ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

PARAM_LABELS = {
    "branch_angle_order1":   "branch_angle_order1",
    "g_axial":               "g_axial",
    "branching_prob_order1": "branching_prob_order1",
    "branching_prob_order2": "branching_prob_order2",
}

Q_ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_COLORS = {"Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf",
            "Q4": "#fdae61", "Q5": "#d7191c"}

def q_color(q): return Q_COLORS.get(q, "gray")

# ── Collect S2W results ────────────────────────────────────────────────────────

rows = []
for _, tc in tc_df.iterrows():
    cfg, sid = tc["config"], int(tc["sample_id"])
    tag  = f"tc_{cfg}_{sid}_s2w"
    best = os.path.join(CALIB_DIR, "experiments", tag, "best_params.json")
    if not os.path.isfile(best):
        print(f"  MISSING: {tag}")
        continue
    with open(best) as f:
        bp = json.load(f)
    row = {"config": cfg, "sample_id": sid,
           "g_axial_quintile": tc["g_axial_quintile"],
           "true_g_axial": tc["g_axial"]}
    for p in ARCH_PARAMS:
        row[f"true_{p}"] = float(tc[p])
        row[f"pred_{p}"] = float(bp[p]) if p in bp else np.nan
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} S2W cases")

# ── Figure: True vs Predicted ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle("CMA-ES S2W [Scen2: elong known, TabICL warm-start] — True vs Predicted (N=20)",
             fontsize=12, fontweight="bold")

for col_idx, param in enumerate(ARCH_PARAMS):
    ax = axes[col_idx]

    true_vals = df[f"true_{param}"].values
    pred_vals = df[f"pred_{param}"].values
    mae = np.nanmean(np.abs(pred_vals - true_vals))

    lo = min(np.nanmin(true_vals), np.nanmin(pred_vals)) * 0.95
    hi = max(np.nanmax(true_vals), np.nanmax(pred_vals)) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.6)

    for _, r in df.iterrows():
        ax.scatter(r[f"true_{param}"], r[f"pred_{param}"],
                   color=q_color(r["g_axial_quintile"]),
                   s=70, edgecolors="0.3", linewidths=0.5, zorder=3)

    mae_str = f"{mae:.3g}" if param == "branch_angle_order1" else f"{mae:.4g}"
    ax.set_title(f"{PARAM_LABELS[param]}\nMAE={mae_str}", fontsize=9)
    ax.set_xlabel("True", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, lw=0.4, alpha=0.5)

legend_handles = [
    mpatches.Patch(color=Q_COLORS[q],
                   label=f"{q}" + (" (low g)" if q == "Q1" else " (high g)" if q == "Q5" else ""))
    for q in Q_ORDER
]
axes[0].legend(handles=legend_handles, title="g_axial",
               fontsize=7.5, title_fontsize=8, framealpha=0.8)

fig.tight_layout()
out = os.path.join(FIG_DIR, "actual_vs_pred_cmae_Scen2W_tabicl_warmstart.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
