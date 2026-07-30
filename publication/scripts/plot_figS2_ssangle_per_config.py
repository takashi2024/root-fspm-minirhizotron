"""
plot_figS2_ssangle_per_config.py

Figure S2 — Per-config SS_angle breakdown across all 6 observation conditions.
4 configs × 6 conditions (S2/S1 × k=1/7/14) × 2 methods.
Layout: 2 rows (CMA-ES top, TabICL-MS bottom) × 3 cols (k=1, k=7, k=14).
Within each panel: grouped bars by config (A, B2, C, D2); S2=blue, S1=red shading.

Run from project root:
  python3 publication/scripts/plot_figS2_ssangle_per_config.py
Output: publication/figures/figS2_ssangle_per_config.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB   = os.path.join(ROOT, "calibration")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "figS2_ssangle_per_config.png")

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

CONFIGS  = ["A", "B2", "C", "D2"]
K_VALS   = [1, 7, 14]
SCENS    = ["S2", "S1"]
METHODS  = ["CMA-ES", "TabICL-MS"]

# Config colors — distinct, colorblind-safe
CFG_COLORS = {
    "A":  "#2166ac",   # blue
    "B2": "#4dac26",   # green
    "C":  "#d01c8b",   # pink/magenta
    "D2": "#f1a340",   # amber
}

# Hatch to distinguish S2 vs S1 within same config (S2=solid fill, S1=hatched)
HATCH = {"S2": "", "S1": "///"}

# ── Load per-config means ─────────────────────────────────────────────────────

data = {}   # data[(scen, k, method, config)] = mean ss_angle

for (scen, k), (path, cmae_label, tab_label) in CSV_MAP.items():
    df = pd.read_csv(path)
    for method_key, label in [("CMA-ES", cmae_label), ("TabICL-MS", tab_label)]:
        sub = df[df["method"] == label]
        for cfg in CONFIGS:
            vals = sub[sub["config"] == cfg]["ss_angle"].dropna()
            data[(scen, k, method_key, cfg)] = vals.mean() if len(vals) > 0 else float("nan")

# ── Plot: 2 rows (method) × 3 cols (k) ───────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14,
                    hspace=0.55, wspace=0.12)

N_CONFIGS = len(CONFIGS)
x_base    = np.arange(N_CONFIGS)
bar_w     = 0.35            # width per scenario bar (S2 and S1 side by side)
offsets   = {"S2": -bar_w / 2, "S1": bar_w / 2}

for row_idx, method in enumerate(METHODS):
    for col_idx, k in enumerate(K_VALS):
        ax = axes[row_idx, col_idx]

        for scen in SCENS:
            xpos = x_base + offsets[scen]
            yvals = [data.get((scen, k, method, cfg), float("nan")) for cfg in CONFIGS]
            colors = [CFG_COLORS[c] for c in CONFIGS]

            bars = ax.bar(xpos, yvals, width=bar_w * 0.88,
                          color=colors,
                          hatch=HATCH[scen],
                          edgecolor="0.25", linewidth=0.6,
                          alpha=0.85 if scen == "S2" else 0.65,
                          zorder=3)

        # reference line at 0
        ax.axhline(0, color="0.5", lw=0.8, ls=":", zorder=2)

        ax.set_xticks(x_base)
        ax.set_xticklabels(CONFIGS, fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.35, alpha=0.5, color="0.75")

        if col_idx == 0:
            ax.set_ylabel("SS$_{\\mathrm{angle}}$", fontsize=10)
        ax.set_title(f"$k$ = {k}", fontsize=9.5, fontweight="bold", pad=4)

        # row label on leftmost panel
        if col_idx == 0:
            row_label = "CMA-ES" if method == "CMA-ES" else "TabICL-MS"
            ax.annotate(row_label,
                        xy=(0.0, 1.22), xycoords="axes fraction",
                        fontsize=10, fontweight="bold",
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="#eef2f7", ec="0.7", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────

cfg_patches = [mpatches.Patch(facecolor=CFG_COLORS[c], edgecolor="0.25",
                               linewidth=0.5, label=f"Config {c}")
               for c in CONFIGS]

s2_patch = mpatches.Patch(facecolor="0.6", edgecolor="0.25", linewidth=0.5,
                           hatch="", label="S2 — elongation known (solid)")
s1_patch = mpatches.Patch(facecolor="0.6", edgecolor="0.25", linewidth=0.5,
                           hatch="///", label="S1 — all unknown (hatched)")

fig.legend(handles=cfg_patches + [s2_patch, s1_patch],
           loc="lower center", ncol=6, fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.52, 0.01), handlelength=1.5, handleheight=1.2)

fig.suptitle(
    "Figure S2 — Per-configuration SS$_{angle}$ across methods and observation conditions  (N=5 per config)",
    fontsize=10.5, y=0.96
)

fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
