"""
plot_fig5_structural_k_effect.py

Figure 5 — Effect of observation seed count (k) on 3D structural recovery metrics.
Two panels: SS_rld, SS_angle.
Four lines: CMA-ES S2, TabICL-MS S2, CMA-ES S1, TabICL-MS S1.

Run from project root:
  python3 publication/scripts/plot_fig4_structural_k_effect.py
Output: publication/figures/fig5_structural_k_effect.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB    = os.path.join(ROOT, "calibration")
FIG_OUT  = os.path.join(ROOT, "publication", "figures", "fig5_structural_k_effect.png")

# ── Load data ─────────────────────────────────────────────────────────────────

CSV_MAP = {
    ("S2", 1):  (os.path.join(CALIB, "comparison_structure_metrics_s2_k1.csv"),
                 "CMA-ES-S2",       "TabICL-MS-k1"),
    ("S2", 7):  (os.path.join(CALIB, "comparison_structure_metrics_s2ms.csv"),
                 "CMA-ES-S2MS",     "TabICL-MS"),
    ("S2", 14): (os.path.join(CALIB, "comparison_structure_metrics_s2ms_k14.csv"),
                 "CMA-ES-S2MS-k14", "TabICL-MS-k14"),
    ("S1", 1):  (os.path.join(CALIB, "comparison_structure_metrics_s1_k1.csv"),
                 "CMA-ES-S1",       "TabICL-MS-k1"),
    ("S1", 7):  (os.path.join(CALIB, "comparison_structure_metrics_s1ms.csv"),
                 "CMA-ES-S1MS",     "TabICL-MS"),
    ("S1", 14): (os.path.join(CALIB, "comparison_structure_metrics_s1ms_k14.csv"),
                 "CMA-ES-S1MS-k14", "TabICL-MS-k14"),
}

K_VALS    = [1, 7, 14]
SCENARIOS = ["S2", "S1"]
METHODS   = ["CMA-ES", "TabICL-MS"]
METRICS   = ["ss_rld", "ss_angle"]

results = {s: {m: {met: [] for met in METRICS} for m in METHODS} for s in SCENARIOS}

for scen in SCENARIOS:
    for k in K_VALS:
        path, cmae_label, tabicl_label = CSV_MAP[(scen, k)]
        df = pd.read_csv(path)
        for method, label in [("CMA-ES", cmae_label), ("TabICL-MS", tabicl_label)]:
            sub = df[df["method"] == label]
            for met in METRICS:
                results[scen][method][met].append(sub[met].mean())

# ── Style ─────────────────────────────────────────────────────────────────────

STYLE = {
    ("S2", "TabICL-MS"): dict(color="#1a6faf", ls="-",  lw=2.2, marker="o", ms=7),
    ("S2", "CMA-ES"):    dict(color="#1a6faf", ls="--", lw=1.8, marker="s", ms=6),
    ("S1", "TabICL-MS"): dict(color="#c0392b", ls="-",  lw=2.2, marker="o", ms=7),
    ("S1", "CMA-ES"):    dict(color="#c0392b", ls="--", lw=1.8, marker="s", ms=6),
}

PANEL = {
    "ss_rld":   dict(ylabel="SS$_{\\mathrm{RLD}}$",   title="Root length density",
                     hline=0.0, ylim=(-0.45, 0.42)),
    "ss_angle": dict(ylabel="SS$_{\\mathrm{angle}}$", title="Crossing angle",
                     hline=0.0, ylim=(-0.05, 0.55)),
}

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(9, 5.0))
fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.26, wspace=0.38)

x = np.array(K_VALS)

for ax, met in zip(axes, METRICS):
    cfg = PANEL[met]
    if cfg["hline"] is not None:
        ax.axhline(cfg["hline"], color="0.6", lw=0.9, ls=":", zorder=0)
    for method in ["CMA-ES", "TabICL-MS"]:
        for scen in SCENARIOS:
            y  = np.array(results[scen][method][met])
            st = STYLE[(scen, method)]
            z  = 3 if method == "TabICL-MS" else 2
            ax.plot(x, y, zorder=z, **st)
    ax.set_xticks(K_VALS)
    ax.set_xlabel("Number of observation seeds ($k$)", fontsize=10)
    ax.set_ylabel(cfg["ylabel"], fontsize=10)
    ax.set_title(cfg["title"], fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(cfg["ylim"])
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5, color="0.75")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────

legend_handles = [
    Line2D([0], [0], color="#1a6faf", ls="-",  lw=2.2, marker="o", ms=7,
           label="TabICL-MS  S2 (elongation known)"),
    Line2D([0], [0], color="#1a6faf", ls="--", lw=1.8, marker="s", ms=6,
           label="CMA-ES        S2 (elongation known)"),
    Line2D([0], [0], color="#c0392b", ls="-",  lw=2.2, marker="o", ms=7,
           label="TabICL-MS  S1 (all params unknown)"),
    Line2D([0], [0], color="#c0392b", ls="--", lw=1.8, marker="s", ms=6,
           label="CMA-ES        S1 (all params unknown)"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=2, fontsize=9.5, framealpha=0.9,
           bbox_to_anchor=(0.52, 0.01), columnspacing=1.5, handlelength=2.2)

fig.suptitle("Effect of observation seed count on 3D structural recovery  (N=20 test cases)",
             fontsize=11, y=0.93)

fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
