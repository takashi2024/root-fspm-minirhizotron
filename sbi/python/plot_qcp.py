"""
plot_qcp.py

WeedMap-style Quantile Calibration Plot (QCP) for TabICL predictions.
Mirrors main_5_QCP_NNDM_TabICL.R visual style:
  - x-axis: Target quantile (0-100 %)
  - y-axis: QCP / empirical coverage (0-100 %)
  - One coloured line per parameter
  - Dashed diagonal = perfect calibration
  - MACE annotation per parameter
  - Two panels: Scen2 (elong known) | Scen1 (all unknown)

Usage (from repo root):
    python sbi/python/plot_qcp.py
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO = Path(__file__).parent.parent.parent
CAL  = REPO / "sbi/data/combined/calibration"
OUT  = CAL / "tabicl_QCP_plot.png"

QUANTILES    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COV_COLS     = [f"cov_q{q:.1f}" for q in QUANTILES]
TARGET_PCT   = [q * 100 for q in QUANTILES]

# ── Parameter display names and colors ────────────────────────────────────────
PARAM_META = {
    "branch_angle_order1":      {"label": "branch angle",    "color": "#1B5E9E"},
    "g_axial":                  {"label": "g_axial",         "color": "#E05C2A"},
    "branching_prob_order1":    {"label": "bp order 1",      "color": "#2E8B57"},
    "branching_prob_order2":    {"label": "bp order 2",      "color": "#9C27B0"},
    "elongation_rate_min":      {"label": "elong min",       "color": "#B8860B"},
    "elongation_rate_max":      {"label": "elong max",       "color": "#C62828"},
    "elongation_logistic_k":    {"label": "elong k",         "color": "#00838F"},
    "elongation_logistic_t50":  {"label": "elong t50",       "color": "#6D4C41"},
}

def draw_panel(ax, df, title):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # Perfect calibration diagonal
    ax.plot([0, 100], [0, 100], linestyle="--", color="grey", linewidth=1.0,
            zorder=1, label="_nolegend_")

    # Light shading ±5pp band around diagonal
    ax.fill_between([0, 100], [-5, 95], [5, 105], color="grey", alpha=0.08,
                    zorder=0)

    legend_handles = []
    for _, row in df.iterrows():
        pname = row["parameter"]
        meta  = PARAM_META.get(pname, {"label": pname, "color": "#555555"})
        cov   = [row[c] * 100 for c in COV_COLS]
        mace  = row["MACE"]

        ax.plot(TARGET_PCT, cov,
                color=meta["color"], linewidth=2.0, alpha=0.9, zorder=2)
        ax.scatter(TARGET_PCT, cov,
                   color=meta["color"], s=22, zorder=3, clip_on=True)

        handle = mlines.Line2D([], [],
                               color=meta["color"], linewidth=2.0,
                               marker="o", markersize=4,
                               label=f"{meta['label']}  (MACE={mace:.3f})")
        legend_handles.append(handle)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=13, direction="out", length=4)
    ax.set_xlabel("Target quantile (%)", fontsize=15, labelpad=6)
    ax.set_ylabel("QCP (%)",             fontsize=15, labelpad=6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.grid(True, which="major", color="#CCCCCC", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=legend_handles,
              fontsize=10, frameon=True, framealpha=0.9,
              edgecolor="#AAAAAA", loc="upper left",
              handlelength=2.0, handletextpad=0.5)


def main():
    s2 = pd.read_csv(CAL / "tabicl_calibration_scen2.csv")
    s1 = pd.read_csv(CAL / "tabicl_calibration_scen1.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             facecolor="white", constrained_layout=True)

    draw_panel(axes[0], s2,
               "Scen 2 — elongation known\n(4 arch params, 10-fold CV, N≈1 123)")
    draw_panel(axes[1], s1,
               "Scen 1 — all unknown\n(8 params, 10-fold CV, N≈1 123)")

    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
