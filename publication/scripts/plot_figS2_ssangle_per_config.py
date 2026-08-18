"""
plot_figS2_ssangle_per_config.py

Figure S2 (2 mm skin depth) — Per-config SS_angle at k=7.
4 configs × 2 scenarios (S2/S1) × 2 methods, skin_depth = 0.2 cm.
Layout: 1 row × 2 cols (CMA-ES left, TabICL-MS right).
Within each panel: grouped bars by config (A, B2, C, D2); S2=solid, S1=hatched.

Note: v2 only has k=7 data; the 3-column (k=1/7/14) grid from v1 cannot be reproduced.
This simplified figure shows the single available condition.

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

# ── Load v2 k=7 CSVs ─────────────────────────────────────────────────────────

CSV_MAP = {
    ("S2", 7): (os.path.join(CALIB, "comparison_structure_metrics_v2_s2ms.csv"),
                "CMA-ES-S2MS", "TabICL-MS-v2"),
    ("S1", 7): (os.path.join(CALIB, "comparison_structure_metrics_v2_s1ms.csv"),
                "CMA-ES-S1MS", "TabICL-MS-v2"),
}

CONFIGS = ["A", "B2", "C", "D2"]
METHODS = ["CMA-ES", "TabICL-MS"]
SCENS   = ["S2", "S1"]

CFG_COLORS = {
    "A":  "#2166ac",
    "B2": "#4dac26",
    "C":  "#d01c8b",
    "D2": "#f1a340",
}

HATCH = {"S2": "", "S1": "///"}

# ── Compute per-config means ──────────────────────────────────────────────────

data = {}  # (scen, k=7, method, config) → mean ss_angle

for (scen, k), (path, cmae_label, tab_label) in CSV_MAP.items():
    df = pd.read_csv(path)
    for method_key, label in [("CMA-ES", cmae_label), ("TabICL-MS", tab_label)]:
        sub = df[df["method"] == label]
        for cfg in CONFIGS:
            vals = sub[sub["config"] == cfg]["ss_angle"].dropna()
            data[(scen, k, method_key, cfg)] = vals.mean() if len(vals) > 0 else float("nan")

# ── Plot: 1 row × 2 cols (method) ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.18,
                    wspace=0.12)

N_CONFIGS = len(CONFIGS)
x_base    = np.arange(N_CONFIGS)
bar_w     = 0.35
offsets   = {"S2": -bar_w / 2, "S1": bar_w / 2}

for col_idx, method in enumerate(METHODS):
    ax = axes[col_idx]

    for scen in SCENS:
        xpos  = x_base + offsets[scen]
        yvals = [data.get((scen, 7, method, cfg), float("nan")) for cfg in CONFIGS]
        colors = [CFG_COLORS[c] for c in CONFIGS]

        ax.bar(xpos, yvals, width=bar_w * 0.88,
               color=colors,
               hatch=HATCH[scen],
               edgecolor="0.25", linewidth=0.6,
               alpha=0.85 if scen == "S2" else 0.60,
               zorder=3)

    ax.axhline(0, color="0.5", lw=0.8, ls=":", zorder=2)
    ax.set_xticks(x_base)
    ax.set_xticklabels(CONFIGS, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5, color="0.75")

    method_label = "CMA-ES" if method == "CMA-ES" else "TabICL-MS"
    ax.set_title(f"{method_label}  ($k$=7,  skin depth = 0.2 cm)",
                 fontsize=10, fontweight="bold", pad=5)

    if col_idx == 0:
        ax.set_ylabel("SS$_{\\mathrm{angle}}$", fontsize=10.5)

# ── Legend ────────────────────────────────────────────────────────────────────

cfg_patches = [mpatches.Patch(facecolor=CFG_COLORS[c], edgecolor="0.25",
                               linewidth=0.5, label=f"Config {c}")
               for c in CONFIGS]
s2_patch = mpatches.Patch(facecolor="0.6", edgecolor="0.25", linewidth=0.5,
                           hatch="", label="S2 — elong. known (solid)")
s1_patch = mpatches.Patch(facecolor="0.6", edgecolor="0.25", linewidth=0.5,
                           hatch="///", label="S1 — all unknown (hatched)")

fig.legend(handles=cfg_patches + [s2_patch, s1_patch],
           loc="lower center", ncol=6, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.52, 0.01), handlelength=1.5, handleheight=1.2)

fig.suptitle(
    "Figure S2 — Per-configuration SS$_{angle}$  (k=7, skin depth = 0.2 cm)",
    fontsize=11, y=0.97
)

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
