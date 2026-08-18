"""
plot_all_results_comparison.py

Comprehensive visual comparison of all CMA-ES and TabICL results:
  - Top row: parameter recovery MAE (4 arch params)
  - Bottom row: structural similarity (SS_rld, SS_angle, Chamfer)

Output: calibration/figures/all_results_comparison.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CALIB = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(CALIB, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ARCH = ["branch_angle_order1", "g_axial",
        "branching_prob_order1", "branching_prob_order2"]
PARAM_SHORT = ["Angle\norder1 (°)", "g_axial", "BP\norder1", "BP\norder2"]
EXCLUDED = {("A", 497), ("C", 100)}

tc = pd.read_csv(os.path.join(CALIB, "test_cases_selected.csv"))

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "CMA-ES S1":      "#1f4e8c",
    "CMA-ES S2":      "#2e86de",
    "CMA-ES S2W":     "#74b9ff",
    "CMA-ES S2MS":    "#00cec9",
    "TabICL S1":      "#d35400",
    "TabICL S2":      "#f39c12",
    "TabICL-MS S1":   "#27ae60",
    "TabICL-MS S2":   "#2ecc71",
}

METHOD_ORDER = ["CMA-ES S1", "CMA-ES S2", "CMA-ES S2W", "CMA-ES S2MS",
                "TabICL S1", "TabICL S2", "TabICL-MS S1", "TabICL-MS S2"]

# ── Load parameter recovery per-case data ──────────────────────────────────────

def load_cmaes_errors(tag_suffix, exclude=False):
    rows = []
    for _, r in tc.iterrows():
        cfg, sid = r["config"], int(r["sample_id"])
        if exclude and (cfg, sid) in EXCLUDED:
            continue
        tag = f"tc_{cfg}_{sid}_{tag_suffix}"
        bp_path = os.path.join(CALIB, "experiments", tag, "best_params.json")
        if not os.path.isfile(bp_path):
            continue
        with open(bp_path) as f:
            bp = json.load(f)
        rows.append({p: abs(float(bp.get(p, np.nan)) - float(r[p])) for p in ARCH})
    return pd.DataFrame(rows)

def load_tabicl_errors(csv_path, scenario):
    df = pd.read_csv(csv_path)
    sub = df[df["scenario"] == scenario]
    rows = {}
    for _, r in sub.iterrows():
        key = (r["config"], r["sample_id"])
        if key not in rows:
            rows[key] = {}
        if r["parameter"] in ARCH:
            rows[key][r["parameter"]] = r["abs_error"]
    return pd.DataFrame(list(rows.values()))

# Load all
param_data = {
    "CMA-ES S1":    load_cmaes_errors("s1",   exclude=False),
    "CMA-ES S2":    load_cmaes_errors("s2",   exclude=False),
    "CMA-ES S2W":   load_cmaes_errors("s2w",  exclude=False),
    "CMA-ES S2MS":  load_cmaes_errors("s2ms", exclude=True),
    "TabICL S1":    load_tabicl_errors(
        os.path.join(os.path.dirname(CALIB), "sbi", "data", "combined",
                     "test_case_eval", "results_tabicl.csv"),
        "Scen1: all unknown"),
    "TabICL S2":    load_tabicl_errors(
        os.path.join(os.path.dirname(CALIB), "sbi", "data", "combined",
                     "test_case_eval", "results_tabicl.csv"),
        "Scen2: elong known"),
    "TabICL-MS S1": load_tabicl_errors(
        os.path.join(os.path.dirname(CALIB), "sbi", "data", "combined",
                     "test_case_eval_multiseed", "results_tabicl_multiseed.csv"),
        "Scen1: all unknown"),
    "TabICL-MS S2": load_tabicl_errors(
        os.path.join(os.path.dirname(CALIB), "sbi", "data", "combined",
                     "test_case_eval_multiseed", "results_tabicl_multiseed.csv"),
        "Scen2: elong known"),
}

# ── Load structural metrics per-case data ──────────────────────────────────────

def load_struct(csv_path, method_name, rename=None):
    df = pd.read_csv(csv_path)
    sub = df[df["method"] == method_name].copy()
    if rename:
        sub = sub.copy()
        sub["method"] = rename
    return sub[["ss_rld", "ss_angle", "chamfer_cm"]]

struct_data = {
    "CMA-ES S1":    load_struct(os.path.join(CALIB, "comparison_structure_metrics_s1.csv"),  "CMA-ES"),
    "CMA-ES S2":    load_struct(os.path.join(CALIB, "comparison_structure_metrics.csv"),     "CMA-ES"),
    "CMA-ES S2W":   load_struct(os.path.join(CALIB, "comparison_structure_metrics_s2w.csv"), "CMA-ES-S2W"),
    "CMA-ES S2MS":  load_struct(os.path.join(CALIB, "comparison_structure_metrics_s2ms.csv"),"CMA-ES-S2MS"),
    "TabICL S1":    load_struct(os.path.join(CALIB, "comparison_structure_metrics_s1.csv"),  "TabICL"),
    "TabICL S2":    load_struct(os.path.join(CALIB, "comparison_structure_metrics.csv"),     "TabICL"),
    # TabICL-MS S2 only (no S1 structural eval)
    "TabICL-MS S2": load_struct(os.path.join(CALIB, "comparison_structure_metrics_s2ms.csv"),"TabICL-MS"),
}
# TabICL-MS S1: not available → placeholder
struct_data["TabICL-MS S1"] = None

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(20, 9),
                         gridspec_kw={"hspace": 0.45, "wspace": 0.35})

x = np.arange(len(METHOD_ORDER))
bar_w = 0.6
jitter = 0.18

def add_bars_and_dots(ax, values_dict, metric, ylabel, title,
                      higher_is_better=True, ref_line=None):
    means, errs_lo, errs_hi, colors_list = [], [], [], []
    for m in METHOD_ORDER:
        df = values_dict.get(m)
        if df is None or metric not in df.columns or df[metric].dropna().empty:
            means.append(np.nan); errs_lo.append(0); errs_hi.append(0)
        else:
            vals = df[metric].dropna().values
            means.append(np.nanmean(vals))
            errs_lo.append(np.nanmean(vals) - np.nanpercentile(vals, 25))
            errs_hi.append(np.nanpercentile(vals, 75) - np.nanmean(vals))
        colors_list.append(COLORS[m])

    for i, (m, mean, c) in enumerate(zip(METHOD_ORDER, means, colors_list)):
        if np.isnan(mean):
            continue
        ax.bar(i, mean, bar_w, color=c, alpha=0.85, zorder=2,
               edgecolor="0.3", linewidth=0.5)
        # Scatter individual cases
        df = values_dict.get(m)
        if df is not None and metric in df.columns:
            vals = df[metric].dropna().values
            jx = np.random.default_rng(hash(m) % (2**31)).uniform(-jitter, jitter, len(vals))
            ax.scatter(i + jx, vals, s=18, color=c, edgecolors="0.2",
                       linewidths=0.4, zorder=3, alpha=0.7)

    if ref_line is not None:
        ax.axhline(ref_line, color="0.4", lw=0.9, ls="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["S1", "S2", "S2W", "S2MS", "S1", "S2", "MS-S1", "MS-S2"],
        fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=7.5)

    # Group dividers
    ax.axvline(3.5, color="0.6", lw=0.8, ls=":")

# ── Top row: Parameter MAE ────────────────────────────────────────────────────

param_labels = [
    ("branch_angle_order1", "MAE (°)",    "branch_angle_order1"),
    ("g_axial",             "MAE",        "g_axial"),
    ("branching_prob_order1","MAE",       "branching_prob_order1"),
    ("branching_prob_order2","MAE",       "branching_prob_order2"),
]

for col, (param, ylabel, title) in enumerate(param_labels):
    ax = axes[0, col]
    add_bars_and_dots(ax, param_data, param, ylabel, title,
                      higher_is_better=False, ref_line=None)
    ax.set_ylim(bottom=0)

# ── Bottom row: Structural metrics ────────────────────────────────────────────

struct_specs = [
    ("ss_rld",    "Skill score",   "SS_rld\n(RLD, higher=better)",  0.0),
    ("ss_angle",  "Skill score",   "SS_angle\n(angle, higher=better)", 0.0),
    ("chamfer_cm","Distance (cm)", "Chamfer distance\n(lower=better)",  None),
]

for col, (metric, ylabel, title, ref) in enumerate(struct_specs):
    ax = axes[1, col]
    add_bars_and_dots(ax, struct_data, metric, ylabel, title,
                      higher_is_better=(metric != "chamfer_cm"),
                      ref_line=ref)

# Hide the unused 4th panel in the bottom row
axes[1, 3].set_visible(False)

# ── Group labels ──────────────────────────────────────────────────────────────

for row in range(2):
    ax = axes[row, 0]
    ax.text(1.75, -0.18, "← CMA-ES →", transform=ax.transAxes,
            ha="center", fontsize=8, color="0.3", style="italic")

# ── Legend ────────────────────────────────────────────────────────────────────

legend_handles = []
group_info = [
    ("CMA-ES", ["CMA-ES S1","CMA-ES S2","CMA-ES S2W","CMA-ES S2MS"],
               ["S1","S2","S2W","S2MS"]),
    ("TabICL (single-seed)", ["TabICL S1","TabICL S2"],
               ["S1","S2"]),
    ("TabICL-MS (K=7 mean)", ["TabICL-MS S1","TabICL-MS S2"],
               ["MS-S1","MS-S2"]),
]
for group_label, methods, short_labels in group_info:
    legend_handles.append(mpatches.Patch(color="none", label=f"$\\bf{{{group_label}}}$"))
    for m, sl in zip(methods, short_labels):
        legend_handles.append(mpatches.Patch(color=COLORS[m], label=f"  {sl}", alpha=0.85))

# Place legend to the right of the hidden bottom-right panel
ax_leg = axes[1, 3]
ax_leg.set_visible(True)
ax_leg.axis("off")
ax_leg.legend(handles=legend_handles, loc="center left",
              fontsize=8, framealpha=0.9,
              title="Methods", title_fontsize=9)

# Notes
fig.text(0.5, 0.01,
         "Top: lower MAE = better parameter recovery.  "
         "Bottom: SS > 0 = better than mean prediction; lower Chamfer = closer 3D structure.\n"
         "TabICL-MS S1 structural eval not available.  "
         "CMA-ES S2MS and TabICL-MS evaluated on N=18 (A_497, C_100 excluded); all others N=20.",
         ha="center", fontsize=7.5, color="0.4")

fig.suptitle("CMA-ES vs TabICL — Parameter Recovery and Structural Similarity\n"
             "All Scenarios: S1 (all unknown) | S2 (elong known) | S2W (warm-start) | S2MS (K=7 seed mean)",
             fontsize=11, fontweight="bold", y=0.99)

out = os.path.join(FIG_DIR, "all_results_comparison.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
