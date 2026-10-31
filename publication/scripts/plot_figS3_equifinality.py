"""
plot_figS3_equifinality.py

Figure S3 (2 mm skin depth) — Equifinality: CMA-ES final objective vs 3D structural recovery.
2x2 grid: rows = Scenario 2 (S2MS) / Scenario 1 (S1MS), cols = SS_rld / SS_angle.
Shows that low minirhizotron loss does NOT guarantee good structural recovery in either scenario.

Run from project root:
  python3 publication/scripts/plot_figS3_equifinality.py
Output: publication/figures/figS3_equifinality.png
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
FIG_OUT = os.path.join(ROOT, "publication", "figures", "figS3_equifinality.png")

# Internal code → paper config label
CFG_LABEL  = {"A": "A", "B2": "B", "C": "C", "D2": "D"}
CFG_COLORS = {"A": "#2166ac", "B2": "#4dac26", "C": "#d01c8b", "D2": "#f1a340"}


def load_J(suffix):
    """Load best_J from run_summary.json for all experiments matching suffix."""
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
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} {suffix} run summaries")
    return df


def load_struct(csv_name, method_name):
    """Load CMA-ES structural metrics from a comparison CSV."""
    df = pd.read_csv(os.path.join(CALIB, csv_name))
    return df[df["method"] == method_name].copy()


def build_dataset(suffix, csv_name, method_name):
    J_df     = load_J(suffix)
    struct   = load_struct(csv_name, method_name)
    merged   = pd.merge(J_df, struct, on=["config", "sample_id"])
    merged   = merged.dropna(subset=["best_J", "ss_rld", "ss_angle"])
    print(f"  Merged: {len(merged)} cases")
    return merged


# ── Load both scenarios ───────────────────────────────────────────────────────

print("Loading Scenario 2 (S2MS):")
s2 = build_dataset("s2ms", "comparison_structure_metrics_v2_s2ms.csv", "CMA-ES-S2MS")

print("Loading Scenario 1 (S1MS):")
s1 = build_dataset("s1ms", "comparison_structure_metrics_v2_s1ms.csv", "CMA-ES-S1MS")

SCENARIOS = [
    (s1, "Scenario 1 (all parameters unknown)"),
    (s2, "Scenario 2 (elongation fixed)"),
]
METRICS = [
    ("ss_rld",   "SS$_{\\mathrm{RLD}}$"),
    ("ss_angle", "SS$_{\\mathrm{angle}}$"),
]

# ── Plot 2×2 ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12,
                    hspace=0.42, wspace=0.38)

for row_idx, (df, scen_label) in enumerate(SCENARIOS):
    colors = [CFG_COLORS.get(c, "#888888") for c in df["config"]]
    for col_idx, (met, ylabel) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        x  = df["best_J"].values
        y  = df[met].values

        ax.scatter(x, y, c=colors, edgecolors="0.25", linewidths=0.6,
                   s=60, zorder=3, alpha=0.9)
        ax.axhline(0, color="0.5", lw=0.8, ls=":", zorder=1)

        r, pval = stats.pearsonr(x, y)
        p_str = "p<0.05" if pval < 0.05 else f"p={pval:.2f}"
        ax.text(0.97, 0.05, f"r = {r:+.2f}  ({p_str})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8))

        m_fit, b_fit = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m_fit * xfit + b_fit, color="0.6", lw=1.2, ls="--", zorder=2)

        ax.set_xlabel("CMA-ES final objective ($J_{best}$)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", lw=0.35, alpha=0.45, color="0.75")

        if col_idx == 0:
            ax.set_title(f"{scen_label}\n{ylabel}", fontsize=9.5,
                         fontweight="bold", pad=5)
        else:
            ax.set_title(ylabel, fontsize=9.5, fontweight="bold", pad=5)

# ── Legend ────────────────────────────────────────────────────────────────────

cfg_patches = [mpatches.Patch(facecolor=CFG_COLORS[c], edgecolor="0.25",
                               linewidth=0.5,
                               label=f"Config {CFG_LABEL[c]}")
               for c in ["A", "B2", "C", "D2"]]
fig.legend(handles=cfg_patches, loc="lower center", ncol=4, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.01), handlelength=1.2)


os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
