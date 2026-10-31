"""
plot_figS4_equifinality.py

Figure S3 — Equifinality: CMA-ES final objective vs 3D structural recovery.
3×4 grid: rows = k (1, 7, 14); cols = S1 SS_RLD, S1 SS_angle, S2 SS_RLD, S2 SS_angle.
Shows that low minirhizotron loss does NOT guarantee good structural recovery.

Run from project root:
  python3 publication/scripts/plot_figS4_equifinality.py
Output: publication/figures/figS4_equifinality.png
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CALIB   = os.path.join(ROOT, "calibration")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "figS4_equifinality.png")

CFG_LABEL  = {"A": "A", "B2": "B", "C": "C", "D2": "D"}
CFG_COLORS = {"A": "#2166ac", "B2": "#4dac26", "C": "#d01c8b", "D2": "#f1a340"}


def load_J(suffix):
    exp_root = os.path.join(CALIB, "experiments")
    rows = []
    for p in glob.glob(os.path.join(exp_root, f"tc_v2_*_{suffix}", "run_summary.json")):
        with open(p) as f:
            d = json.load(f)
        tag   = os.path.basename(os.path.dirname(p))
        inner = tag.replace("tc_v2_", "").replace(f"_{suffix}", "")
        tokens    = inner.split("_")
        sample_id = int(tokens[-1])
        config    = "_".join(tokens[:-1])
        rows.append({"config": config, "sample_id": sample_id,
                     "best_J": d.get("best_J")})
    return pd.DataFrame(rows)


def load_struct(csv_name, method_name):
    df = pd.read_csv(os.path.join(CALIB, csv_name))
    return df[df["method"] == method_name].copy()


def build_dataset(suffix, csv_name, method_name):
    J_df   = load_J(suffix)
    struct = load_struct(csv_name, method_name)
    merged = pd.merge(J_df, struct, on=["config", "sample_id"])
    merged = merged.dropna(subset=["best_J", "ss_rld", "ss_angle"])
    print(f"  {suffix}: {len(merged)} cases")
    return merged


# ── Load all k × scenario combinations ───────────────────────────────────────

K_CONFIGS = [
    {"k": 1,
     "s1": build_dataset("s1",       "comparison_structure_metrics_v2_s1_k1.csv",   "CMA-ES-S1"),
     "s2": build_dataset("s2",       "comparison_structure_metrics_v2_s2_k1.csv",   "CMA-ES-S2")},
    {"k": 7,
     "s1": build_dataset("s1ms",     "comparison_structure_metrics_v2_s1ms.csv",    "CMA-ES-S1MS"),
     "s2": build_dataset("s2ms",     "comparison_structure_metrics_v2_s2ms.csv",    "CMA-ES-S2MS")},
    {"k": 14,
     "s1": build_dataset("s1ms_k14", "comparison_structure_metrics_v2_s1ms_k14.csv","CMA-ES-S1MS-k14"),
     "s2": build_dataset("s2ms_k14", "comparison_structure_metrics_v2_s2ms_k14.csv","CMA-ES-S2MS-k14")},
]

# Cols: S1 SS_RLD, S1 SS_angle, S2 SS_RLD, S2 SS_angle
COLS = [
    ("s1", "ss_rld",   "S1  SS$_{\\mathrm{RLD}}$"),
    ("s1", "ss_angle", "S1  SS$_{\\mathrm{angle}}$"),
    ("s2", "ss_rld",   "S2  SS$_{\\mathrm{RLD}}$"),
    ("s2", "ss_angle", "S2  SS$_{\\mathrm{angle}}$"),
]

# ── Plot 3×4 ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.11,
                    hspace=0.48, wspace=0.38)

for row_idx, kcfg in enumerate(K_CONFIGS):
    k = kcfg["k"]
    for col_idx, (scen_key, met, col_title) in enumerate(COLS):
        ax  = axes[row_idx, col_idx]
        df  = kcfg[scen_key]
        colors = [CFG_COLORS.get(c, "#888888") for c in df["config"]]

        x = df["best_J"].values
        y = df[met].values

        ax.scatter(x, y, c=colors, edgecolors="0.25", linewidths=0.6,
                   s=55, zorder=3, alpha=0.9)
        ax.axhline(0, color="0.5", lw=0.8, ls=":", zorder=1)

        r, pval = stats.pearsonr(x, y)
        p_str = "p<0.05" if pval < 0.05 else f"p={pval:.2f}"
        ax.text(0.97, 0.05, f"r = {r:+.2f}  ({p_str})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8))

        m_fit, b_fit = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m_fit * xfit + b_fit, color="0.6", lw=1.2, ls="--", zorder=2)

        ax.set_xlabel("CMA-ES final objective ($J_{best}$)", fontsize=8.5)
        ax.set_ylabel(col_title.split("  ")[1] if col_idx > 0 else col_title.split("  ")[1],
                      fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", lw=0.35, alpha=0.45, color="0.75")

        title = col_title if row_idx == 0 else col_title.split("  ")[1]
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

    # Row label: k value
    axes[row_idx, 0].annotate(
        f"$k$ = {k}",
        xy=(-0.28, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", ha="center", va="center",
        rotation=90,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef2f7", ec="0.7", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────

cfg_patches = [mpatches.Patch(facecolor=CFG_COLORS[c], edgecolor="0.25",
                               linewidth=0.5, label=f"Config {CFG_LABEL[c]}")
               for c in ["A", "B2", "C", "D2"]]
fig.legend(handles=cfg_patches, loc="lower center", ncol=4, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.01), handlelength=1.2)

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
