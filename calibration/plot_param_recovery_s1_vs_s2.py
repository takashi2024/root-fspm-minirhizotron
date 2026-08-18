"""
plot_param_recovery_s1_vs_s2.py

Compare CMA-ES parameter recovery: Scenario 1 (all unknown) vs Scenario 2 (elongation known)
for the 4 architectural parameters.

Layout:
  Top row    — True vs Predicted scatter (S1 and S2 overlaid, different markers)
  Bottom row — Per-case |% error| dumbbell (S1 vs S2) per param

Output: calibration/figures/param_recovery_cmae_s1_vs_s2.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(CALIB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

tc_df = pd.read_csv(os.path.join(CALIB_DIR, "test_cases_selected.csv"))

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

C_S2 = "#1B5E9E"   # dark blue
C_S1 = "#E07B39"   # orange

# ── Load data ──────────────────────────────────────────────────────────────────

rows = []
for _, tc in tc_df.iterrows():
    cfg, sid = tc["config"], int(tc["sample_id"])
    for scen in ["s1", "s2"]:
        tag  = f"tc_{cfg}_{sid}_{scen}"
        best = os.path.join(CALIB_DIR, "experiments", tag, "best_params.json")
        if not os.path.isfile(best):
            continue
        with open(best) as f:
            bp = json.load(f)
        row = {"scenario": scen.upper(), "config": cfg, "sample_id": sid,
               "g_axial_quintile": tc["g_axial_quintile"],
               "true_g_axial": float(tc["g_axial"]),
               "windows_with_roots": int(tc["windows_with_roots"])}
        for p in ARCH_PARAMS:
            row[f"true_{p}"]   = float(tc[p])
            row[f"pred_{p}"]   = float(bp[p])
            row[f"abserr_{p}"] = abs(float(bp[p]) - float(tc[p]))
            row[f"pcterr_{p}"] = abs(float(bp[p]) - float(tc[p])) / abs(float(tc[p])) * 100
        rows.append(row)

df = pd.DataFrame(rows)
s1 = df[df.scenario == "S1"].sort_values(["g_axial_quintile", "true_g_axial"]).reset_index(drop=True)
s2 = df[df.scenario == "S2"].sort_values(["g_axial_quintile", "true_g_axial"]).reset_index(drop=True)
N  = len(s1)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.08,
                    hspace=0.45, wspace=0.32)
fig.suptitle("CMA-ES parameter recovery: Scenario 1 (all unknown) vs Scenario 2 (elongation known)  [N=20]",
             fontsize=11, fontweight="bold")

# ── Top row: True vs Predicted ─────────────────────────────────────────────────

for col, param in enumerate(ARCH_PARAMS):
    ax = axes[0, col]

    all_true = df[f"true_{param}"].values
    all_pred = df[f"pred_{param}"].values
    lo = min(all_true.min(), all_pred.min()) * 0.93
    hi = max(all_true.max(), all_pred.max()) * 1.07
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.55, zorder=0)

    mae_s2 = s2[f"abserr_{param}"].mean()
    mae_s1 = s1[f"abserr_{param}"].mean()

    for _, r in s2.iterrows():
        ax.scatter(r[f"true_{param}"], r[f"pred_{param}"],
                   marker="s", s=55, color=Q_COLORS[r["g_axial_quintile"]],
                   edgecolors=C_S2, linewidths=0.9, zorder=3, alpha=0.9)
    for _, r in s1.iterrows():
        ax.scatter(r[f"true_{param}"], r[f"pred_{param}"],
                   marker="o", s=55, color=Q_COLORS[r["g_axial_quintile"]],
                   edgecolors=C_S1, linewidths=0.9, zorder=2, alpha=0.9)

    ax.set_title(f"{PARAM_LABELS[param]}\nMAE  S2={mae_s2:.4g}  S1={mae_s1:.4g}", fontsize=8.5)
    ax.set_xlabel("True", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, lw=0.4, alpha=0.4)

# shared legend for top row
q_handles = [mpatches.Patch(color=Q_COLORS[q],
             label=f"{q}" + (" (low g)" if q == "Q1" else " (high g)" if q == "Q5" else ""))
             for q in Q_ORDER]
scen_handles = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor="0.7",
           markeredgecolor=C_S2, markersize=8, label="S2 (elong known)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="0.7",
           markeredgecolor=C_S1, markersize=8, label="S1 (all unknown)"),
]
axes[0, 0].legend(handles=scen_handles + q_handles,
                  fontsize=7.5, framealpha=0.85, loc="upper left")

# ── Bottom row: per-case |% error| dumbbell ────────────────────────────────────

for col, param in enumerate(ARCH_PARAMS):
    ax = axes[1, col]
    ys = np.arange(N)

    e_s2 = s2[f"pcterr_{param}"].values
    e_s1 = s1[f"pcterr_{param}"].values

    for i in range(N):
        # connecting line
        ax.plot([e_s2[i], e_s1[i]], [ys[i], ys[i]],
                lw=0.8, color="0.75", zorder=1)
        qc = Q_COLORS[s2.iloc[i]["g_axial_quintile"]]
        ax.scatter(e_s2[i], ys[i], marker="s", s=28,
                   color=qc, edgecolors=C_S2, linewidths=0.9, zorder=3)
        ax.scatter(e_s1[i], ys[i], marker="o", s=28,
                   color=qc, edgecolors=C_S1, linewidths=0.9, zorder=3)

    mean_s2 = e_s2.mean()
    mean_s1 = e_s1.mean()
    ax.axvline(mean_s2, color=C_S2, lw=1.2, ls="--", alpha=0.7,
               label=f"S2 mean={mean_s2:.0f}%")
    ax.axvline(mean_s1, color=C_S1, lw=1.2, ls="--", alpha=0.7,
               label=f"S1 mean={mean_s1:.0f}%")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.config}/{r.sample_id}" for _, r in s2.iterrows()],
                       fontsize=6.0)
    ax.set_xlabel("|% error|", fontsize=8)
    ax.set_title(f"{PARAM_LABELS[param]}", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    # Quintile bands
    for q in Q_ORDER:
        idxs = [i for i, r in s2.iterrows() if r["g_axial_quintile"] == q]
        if idxs:
            ax.axhspan(min(idxs) - 0.45, max(idxs) + 0.45,
                       alpha=0.07, color=Q_COLORS[q], zorder=0)

out = os.path.join(FIG_DIR, "param_recovery_cmae_s1_vs_s2.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")

# ── Print summary table ────────────────────────────────────────────────────────
print("\n=== Mean |% error| ===")
print(f"{'Param':<30} {'S2':>8} {'S1':>8} {'Δ(S1-S2)':>10}")
print("-"*55)
for p in ARCH_PARAMS:
    v2 = df[df.scenario=="S2"][f"pcterr_{p}"].mean()
    v1 = df[df.scenario=="S1"][f"pcterr_{p}"].mean()
    print(f"  {p:<28} {v2:>8.1f} {v1:>8.1f} {v1-v2:>+10.1f}")
print("-"*55)
o2 = df[df.scenario=="S2"][[f"pcterr_{p}" for p in ARCH_PARAMS]].values.mean()
o1 = df[df.scenario=="S1"][[f"pcterr_{p}" for p in ARCH_PARAMS]].values.mean()
print(f"  {'Overall':<28} {o2:>8.1f} {o1:>8.1f} {o1-o2:>+10.1f}")
