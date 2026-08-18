"""
plot_figS3_equifinality.py

Figure S3 (2 mm skin depth) — Equifinality: CMA-ES final objective vs 3D structural recovery.
Scatter of best-J (minirhizotron loss) vs SS_rld and SS_angle for 20 S2MS k=7 cases.
Shows that low minirhizotron loss does NOT guarantee good structural recovery.

Note: v1 used S2 k=1 for this figure. v2 uses S2MS k=7 (the only available v2 CMA-ES S2 run).
The equifinality effect is expected to remain: minirhizotron loss and structural SS are
poorly correlated regardless of k.

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

# ── Load CMA-ES S2MS v2 final objective values ────────────────────────────────

exp_root = os.path.join(CALIB, "experiments")
J_rows = []
for p in glob.glob(os.path.join(exp_root, "tc_v2_*_s2ms", "run_summary.json")):
    with open(p) as f:
        d = json.load(f)
    tag = os.path.basename(os.path.dirname(p))
    # tag format: tc_v2_{config}_{sid}_s2ms
    inner = tag.replace("tc_v2_", "").replace("_s2ms", "")
    tokens = inner.split("_")
    sample_id = int(tokens[-1])
    config = "_".join(tokens[:-1])
    J_rows.append({"config": config, "sample_id": sample_id,
                   "best_J": d.get("best_J"),
                   "runtime_h": d.get("runtime_seconds", 0) / 3600})

J_df = pd.DataFrame(J_rows)
print(f"Loaded {len(J_df)} CMA-ES S2MS v2 run summaries")

# ── Load structural metrics (S2MS k=7 v2, CMA-ES) ────────────────────────────

struct_df = pd.read_csv(os.path.join(CALIB, "comparison_structure_metrics_v2_s2ms.csv"))
struct_df = struct_df[struct_df["method"] == "CMA-ES-S2MS"].copy()

# ── Merge ─────────────────────────────────────────────────────────────────────

merged = pd.merge(J_df, struct_df, on=["config", "sample_id"])
merged = merged.dropna(subset=["best_J", "ss_rld", "ss_angle"])
print(f"Merged: {len(merged)} cases")

CFG_COLORS = {"A": "#2166ac", "B2": "#4dac26", "C": "#d01c8b", "D2": "#f1a340"}
colors = [CFG_COLORS.get(c, "#888888") for c in merged["config"]]

# ── Plot ──────────────────────────────────────────────────────────────────────

PANELS = [
    ("ss_rld",   "SS$_{\\mathrm{RLD}}$",  "Root length density skill"),
    ("ss_angle", "SS$_{\\mathrm{angle}}$", "Crossing angle skill"),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.18, wspace=0.40)

for ax, (met, ylabel, panel_title) in zip(axes, PANELS):
    x = merged["best_J"].values
    y = merged[met].values

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

    ax.set_xlabel("CMA-ES final objective ($J_{best}$,  0.2 cm skin depth)", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(panel_title, fontsize=9.5, fontweight="bold", pad=5)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", lw=0.35, alpha=0.45, color="0.75")

# ── Legend ────────────────────────────────────────────────────────────────────

cfg_patches = [mpatches.Patch(facecolor=CFG_COLORS[c], edgecolor="0.25",
                               linewidth=0.5, label=f"Config {c}")
               for c in ["A", "B2", "C", "D2"]]
fig.legend(handles=cfg_patches, loc="lower center", ncol=4, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.52, 0.01), handlelength=1.2)

fig.suptitle(
    "Figure S3 — Equifinality: minirhizotron fit quality vs structural recovery  "
    "(CMA-ES S2MS k=7, skin depth = 0.2 cm, N=20 test cases)",
    fontsize=10.5, y=0.95
)

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
