"""plot_paper_figures.py

Publication figures for the CMA-ES vs SBI (TabICL) calibration paper.

Outputs (in calibration/figures/):
  fig1_structural_skill_scores.pdf  — SS_rld & SS_angle, CMA-ES vs TabICL (v2)
  fig2_skin_depth_sensitivity.pdf   — Skill scores at 1.0 cm vs 0.2 cm skin depth
  fig3_parameter_recovery.pdf       — % error per parameter (v2 S2MS)
  fig4_obs_vs_pred_depth.pdf        — Depth profile obs vs pred (selected cases)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":     42,   # embed fonts as TrueType
    "svg.fonttype":     "none",
})

CALIB_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(CALIB_DIR, "figures")
EXPS_DIR   = os.path.join(CALIB_DIR, "experiments")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    "CMA-ES-S2MS": "#2171B5",   # steel blue
    "CMA-ES-S1MS": "#6BAED6",   # light blue
    "TabICL-MS":   "#E6550D",   # burnt orange
}
SD_COLORS = {1.0: "#BDBDBD", 0.2: "#636363"}   # grey scale for sensitivity fig

CONFIG_ORDER = ["A", "B2", "C", "D2"]
CONFIG_COLORS = {"A": "#4DAF4A", "B2": "#984EA3", "C": "#FF7F00", "D2": "#A65628"}
PARAM_LABELS = {
    "branch_angle_order1":   "Branch angle\n(order 1, °)",
    "g_axial":               "Axial\nconductance",
    "branching_prob_order1": "Branching prob.\n(order 1)",
    "branching_prob_order2": "Branching prob.\n(order 2)",
}

# ── Data ───────────────────────────────────────────────────────────────────────
struct_s2  = pd.read_csv(os.path.join(CALIB_DIR, "comparison_structure_metrics_combined_s2ms.csv"))
struct_s1  = pd.read_csv(os.path.join(CALIB_DIR, "comparison_structure_metrics_combined_s1ms.csv"))
param_v2   = pd.read_csv(os.path.join(CALIB_DIR, "test_case_results_v2_s2ms.csv"))
param_comb = pd.read_csv(os.path.join(CALIB_DIR, "test_case_results_combined_s2ms.csv"))

# ── Helpers ────────────────────────────────────────────────────────────────────

def jitter(n, width=0.08, rng=None):
    rng = rng or np.random.default_rng(42)
    return rng.uniform(-width, width, n)

def strip_mean(ax, data, x, color, marker="o", size=4, mean_lw=2, alpha=0.55, jw=0.12):
    rng = np.random.default_rng(hash(str(x)) % (2**31))
    vals = np.asarray(data.dropna())
    xs   = x + jitter(len(vals), jw, rng)
    ax.scatter(xs, vals, s=size**2, color=color, alpha=alpha,
               marker=marker, linewidths=0, zorder=3)
    ax.plot([x - 0.22, x + 0.22], [vals.mean(), vals.mean()],
            color=color, lw=mean_lw, solid_capstyle="round", zorder=4)
    return vals.mean()

def add_null_line(ax):
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4, zorder=1)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"  → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Structural skill scores (v2, 0.2 cm skin depth)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)

for ax, metric, ylabel in zip(
        axes,
        ["ss_rld",   "ss_angle"],
        ["Skill score (RLD)", "Skill score (angle)"]):

    ax.set_ylabel(ylabel)
    add_null_line(ax)

    for panel_name, mode_key, methods, x_positions in [
        ("S2 (arch params fixed elong.)",
         "s2", ["CMA-ES-S2MS", "TabICL-MS"], [0.8, 1.2]),
        ("S1 (all params free)",
         "s1", ["CMA-ES-S1MS", "TabICL-MS"], [2.0, 2.4]),
    ]:
        src = struct_s2 if mode_key == "s2" else struct_s1
        df  = src[src.skin_depth_cm == 0.2]
        for method, xp in zip(methods, x_positions):
            c    = COLORS[method]
            data = df[df.method == method][metric]
            strip_mean(ax, data, xp, c)

    ax.set_xlim(0.4, 2.8)
    ax.set_xticks([1.0, 2.2])
    ax.set_xticklabels(["Scenario 2\n(elong. known)", "Scenario 1\n(all unknown)"])
    ax.set_ylim(-1.5, 1.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

legend_els = [
    Line2D([0],[0], color=COLORS["CMA-ES-S2MS"], lw=0, marker="o", ms=6, label="CMA-ES"),
    Line2D([0],[0], color=COLORS["TabICL-MS"],   lw=0, marker="o", ms=6, label="TabICL"),
    Line2D([0],[0], color="black", lw=0, marker="_", ms=10, mew=2, label="Mean"),
]
axes[0].legend(handles=legend_els, frameon=False, loc="upper left")

axes[0].set_title("Root length density", pad=4)
axes[1].set_title("Root angle distribution", pad=4)
fig.suptitle("Structural similarity skill scores  (skin depth = 0.2 cm)", y=1.01, fontsize=10)
fig.tight_layout()
save(fig, "fig1_structural_skill_scores.pdf")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Skin depth sensitivity (S2MS only: 1.0 vs 0.2 cm)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

df_s2 = struct_s2.copy()
METHODS_S2 = ["CMA-ES-S2MS", "TabICL-MS"]
method_labels = ["CMA-ES", "TabICL"]
x_base = [1, 2]
width  = 0.25

for ax, metric, ylabel in zip(
        axes,
        ["ss_rld", "ss_angle"],
        ["Skill score (RLD)", "Skill score (angle)"]):

    add_null_line(ax)

    for i, (method, label) in enumerate(zip(METHODS_S2, method_labels)):
        for j, (sd, hatch) in enumerate([(1.0, "///"), (0.2, "")]):
            sub  = df_s2[(df_s2.method == method) & (df_s2.skin_depth_cm == sd)]
            vals = sub[metric].dropna()
            xp   = x_base[i] + (j - 0.5) * width
            mean = vals.mean()
            se   = vals.std() / np.sqrt(len(vals))
            bar  = ax.bar(xp, mean, width=width*0.9,
                          color=COLORS[method], alpha=(0.55 if sd == 1.0 else 0.95),
                          hatch=hatch, edgecolor="white" if not hatch else COLORS[method],
                          linewidth=0.5, zorder=3)
            ax.errorbar(xp, mean, yerr=se, fmt="none",
                        color="black", lw=1.0, capsize=3, zorder=4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 2.7)

legend_els = [
    mpl.patches.Patch(facecolor="#888888", alpha=0.55, hatch="///", label="1.0 cm skin depth"),
    mpl.patches.Patch(facecolor="#888888", alpha=0.95, label="0.2 cm skin depth"),
]
axes[0].legend(handles=legend_els, frameon=False)
axes[0].set_title("Root length density", pad=4)
axes[1].set_title("Root angle distribution", pad=4)
fig.suptitle("Effect of skin depth on structural similarity  (S2 — arch params)", y=1.01, fontsize=10)
fig.tight_layout()
save(fig, "fig2_skin_depth_sensitivity.pdf")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Parameter recovery (v2 S2MS)
# ══════════════════════════════════════════════════════════════════════════════

S2MS_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

fig, axes = plt.subplots(1, 4, figsize=(9.0, 3.2))

for ax, p in zip(axes, S2MS_PARAMS):
    col = f"pct_err_{p}"
    for ci, cfg in enumerate(CONFIG_ORDER):
        sub  = param_v2[param_v2.config == cfg][col].dropna()
        # cap display at 500% for readability
        sub  = sub.clip(upper=500)
        xpos = ci + 1
        strip_mean(ax, sub, xpos, CONFIG_COLORS[cfg], size=5, jw=0.18, alpha=0.65)

    ax.axhline(50, color="black", lw=0.6, ls=":", alpha=0.4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(CONFIG_ORDER)
    ax.set_xlabel("Config")
    ax.set_title(PARAM_LABELS[p], pad=4)
    ax.set_ylim(-30, 520)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(100))

axes[0].set_ylabel("Absolute % error  (capped at 500%)")
for ax in axes[1:]:
    ax.set_ylabel("")

legend_els = [Line2D([0],[0], color=CONFIG_COLORS[c], lw=0, marker="o", ms=6, label=c)
              for c in CONFIG_ORDER]
legend_els.append(Line2D([0],[0], color="black", lw=2, label="Mean"))
axes[-1].legend(handles=legend_els, frameon=False, loc="upper right", ncol=1)

fig.suptitle("Parameter recovery — CMA-ES S2MS  (skin depth = 0.2 cm)", y=1.01, fontsize=10)
fig.tight_layout()
save(fig, "fig3_parameter_recovery.pdf")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Depth profile obs vs pred (selected cases)
# ══════════════════════════════════════════════════════════════════════════════
# Select cases with positive R²_all for a representative look.
good_cases = param_v2[param_v2.r2_all > 0.3][["config","sample_id","r2_all"]].sort_values("r2_all", ascending=False)
selected = good_cases.head(4)[["config","sample_id"]].values.tolist()

if len(selected) == 0:
    # fallback to highest r2
    selected = param_v2.nlargest(4, "r2_all")[["config","sample_id"]].values.tolist()

fig, axes = plt.subplots(1, len(selected), figsize=(3.0*len(selected), 3.8), sharey=True)
if len(selected) == 1: axes = [axes]

FINAL_DAY = 84
DEPTH_BINS = np.arange(0, 55, 5)

for ax, (cfg, sid) in zip(axes, selected):
    tag  = f"tc_v2_{cfg}_{sid}_s2ms"
    path = os.path.join(EXPS_DIR, tag, "best_obs_pred_stations.csv")
    if not os.path.isfile(path):
        ax.set_visible(False); continue

    df  = pd.read_csv(path)
    end = df[df.day == FINAL_DAY]

    # Bin by depth and sum root length
    end = end.copy()
    end["depth_bin"] = pd.cut(end.depth_cm, bins=DEPTH_BINS, labels=DEPTH_BINS[:-1]+2.5)
    grp = end.groupby("depth_bin")[["obs_cm","pred_cm"]].sum()

    depths = grp.index.astype(float)
    ax.barh(depths, grp.obs_cm,  height=4.5, color="#4393C3", alpha=0.6, label="Observed")
    ax.barh(depths, -grp.pred_cm, height=4.5, color="#D6604D", alpha=0.6, label="Predicted (CMA-ES)")
    ax.axvline(0, color="black", lw=0.8)

    r2_val = param_v2[(param_v2.config==cfg)&(param_v2.sample_id==sid)]["r2_all"].values[0]
    ax.set_title(f"{cfg} #{sid}\n(R² = {r2_val:+.2f})", pad=4)
    ax.set_xlabel("Root length (cm)")
    ax.set_xlim(-max(grp.pred_cm.max(), grp.obs_cm.max())*1.2,
                 max(grp.pred_cm.max(), grp.obs_cm.max())*1.2)
    # Clean x-ticks: show absolute values
    mx = round(max(grp.pred_cm.max(), grp.obs_cm.max()) * 1.1 / 2) * 2
    xt = np.arange(-mx, mx+1, max(2, round(mx/3)))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(abs(int(v))) for v in xt])

axes[0].set_ylabel("Depth (cm)")
axes[0].invert_yaxis()
axes[0].yaxis.set_major_locator(mticker.MultipleLocator(10))

legend_els = [
    mpl.patches.Patch(color="#4393C3", alpha=0.7, label="Observed"),
    mpl.patches.Patch(color="#D6604D", alpha=0.7, label="CMA-ES"),
]
axes[-1].legend(handles=legend_els, frameon=False, loc="lower right")

fig.suptitle(f"Root length depth profile — day {FINAL_DAY}  (skin depth = 0.2 cm)", y=1.02, fontsize=10)
fig.tight_layout()
save(fig, "fig4_obs_vs_pred_depth.pdf")
plt.close(fig)

print(f"\nAll figures saved to: {FIG_DIR}/")
