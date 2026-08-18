"""
plot_structural_vs_param_recovery.py

Reads the voxel_rld_mode_comparison.log and wasserstein_mode_comparison.log,
then plots and computes correlations between structural recovery metrics
(SS_rld, SS_angle, SS_depth, Chamfer) and parameter recovery error (mean |% err|).

Outputs:
  calibration/figures/structural_vs_param_recovery.png
"""

import re
import os
import math

# ── Try to import plotting/stats libs; fall back to text-only ─────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib not available — printing correlation table only")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(CALIB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Parse voxel log ────────────────────────────────────────────────────────────
# Table columns: Objective  RMSE_rld  SS_rld  RMSE_angle  SS_angle  Chamfer  Mean|%err|

def parse_voxel_log(path):
    records = []
    if not os.path.isfile(path):
        print(f"Not found: {path}")
        return records

    current_group = None
    in_table = False

    with open(path) as f:
        for line in f:
            # Group header
            m = re.match(r'\s+([A-C]:[^\n]+)', line)
            if m and '═' not in line:
                current_group = m.group(1).strip()

            # Separator toggles table region
            if '─' * 10 in line:
                in_table = not in_table
                continue

            if not in_table or current_group is None:
                continue

            # Skip header row
            if 'Objective' in line or 'RMSE_rld' in line:
                continue

            # Data row: name (40 chars) then 6 numbers
            # Use regex to extract: text field + 6 floats (last has %)
            pat = r'^\s{2}(.{40})\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)%'
            m = re.match(pat, line)
            if m:
                obj      = m.group(1).strip()
                rmse_rld = float(m.group(2))
                ss_rld   = float(m.group(3))
                rmse_ang = float(m.group(4))
                ss_ang   = float(m.group(5))
                chamfer  = float(m.group(6))
                pct_err  = float(m.group(7))
                records.append(dict(
                    group=current_group, obj=obj,
                    rmse_rld=rmse_rld, ss_rld=ss_rld,
                    rmse_angle=rmse_ang, ss_angle=ss_ang,
                    chamfer=chamfer, pct_err=pct_err,
                    source="voxel"
                ))
    return records

# ── Parse Wasserstein log ──────────────────────────────────────────────────────
# Table columns: Objective  R²(ms)  RMSE(ms)  W_depth  SS_depth  W_angle  SS_angle  Mean|%err|

def parse_wasserstein_log(path):
    records = []
    if not os.path.isfile(path):
        print(f"Not found: {path}")
        return records

    current_group = None
    in_table = False

    with open(path) as f:
        for line in f:
            m = re.match(r'\s+([A-C]:[^\n]+)', line)
            if m and '═' not in line:
                current_group = m.group(1).strip()

            if '─' * 10 in line:
                in_table = not in_table
                continue

            if not in_table or current_group is None:
                continue
            if 'Objective' in line or 'R²' in line:
                continue

            # cols: name(40) R2 RMSE W_depth SS_depth W_angle SS_angle pct_err
            # W_angle and SS_angle may be "  —" when missing
            pat = r'^\s{2}(.{40})\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)'
            m = re.match(pat, line)
            if not m:
                continue
            obj      = m.group(1).strip()
            ss_depth = float(m.group(5))
            # angle fields may be "—"
            try:
                w_angle  = float(m.group(6))
                ss_wangle = float(m.group(7))
            except ValueError:
                w_angle = math.nan
                ss_wangle = math.nan
            try:
                pct_str = m.group(8).replace('%','')
                pct_err = float(pct_str)
            except ValueError:
                continue

            records.append(dict(
                group=current_group, obj=obj,
                ss_depth=ss_depth, ss_wangle=ss_wangle,
                pct_err=pct_err,
                source="wasserstein"
            ))
    return records

# ── Load data ──────────────────────────────────────────────────────────────────

voxel_log = os.path.join(CALIB_DIR, "pipeline_logs", "voxel_rld_mode_comparison.log")
wass_log  = os.path.join(CALIB_DIR, "pipeline_logs", "wasserstein_mode_comparison.log")

voxel_recs = parse_voxel_log(voxel_log)
wass_recs  = parse_wasserstein_log(wass_log)

print(f"Voxel records loaded: {len(voxel_recs)}")
print(f"Wasserstein records loaded: {len(wass_recs)}")

# ── Correlation helper ─────────────────────────────────────────────────────────

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return math.nan, math.nan
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    if den < 1e-12:
        return math.nan, math.nan
    r = num / den
    if HAS_SCIPY:
        _, p = sp_stats.pearsonr(xs, ys)
    else:
        # two-tailed t approximation
        t = r * math.sqrt((n-2) / max(1 - r**2, 1e-12))
        p = math.nan  # skip without scipy
    return r, p

def spearman(xs, ys):
    if not HAS_SCIPY:
        return math.nan, math.nan
    r, p = sp_stats.spearmanr(xs, ys)
    return r, p

# ── Print correlation table ────────────────────────────────────────────────────

GROUP_COLORS = {
    "A: 200cm_length_only":       "#e67e22",
    "B: 100cm_pbc+angle_hist":    "#2980b9",
    "C: 200cm+angle_hist":        "#27ae60",
}

print("\n" + "="*70)
print("Correlations: structural metric vs mean |% parameter error|")
print("="*70)

# Voxel metrics vs pct_err
for metric_key, label in [("ss_angle", "SS_angle (voxel)"),
                            ("ss_rld",   "SS_rld  (voxel)"),
                            ("chamfer",  "Chamfer (voxel)")]:
    print(f"\n{label}:")
    print(f"  {'Group':<30}  {'N':>3}  {'Pearson r':>10}  {'p':>8}  {'Spearman r':>10}  {'p':>8}")
    print("  " + "-"*68)
    for grp in sorted(set(r["group"] for r in voxel_recs)):
        sub = [r for r in voxel_recs if r["group"] == grp]
        xs = [r[metric_key] for r in sub]
        ys = [r["pct_err"]  for r in sub]
        pr, pp = pearson(xs, ys)
        sr, sp = spearman(xs, ys)
        print(f"  {grp:<30}  {len(sub):>3}  {pr:>+10.3f}  {pp:>8.3f}  {sr:>+10.3f}  {sp:>8.3f}")
    # All groups combined
    xs = [r[metric_key] for r in voxel_recs]
    ys = [r["pct_err"]  for r in voxel_recs]
    pr, pp = pearson(xs, ys)
    sr, sp = spearman(xs, ys)
    print(f"  {'All groups':<30}  {len(xs):>3}  {pr:>+10.3f}  {pp:>8.3f}  {sr:>+10.3f}  {sp:>8.3f}")

# Wasserstein SS_depth vs pct_err
if wass_recs:
    print(f"\nSS_depth (Wasserstein, Level 2a):")
    print(f"  {'Group':<30}  {'N':>3}  {'Pearson r':>10}  {'p':>8}  {'Spearman r':>10}  {'p':>8}")
    print("  " + "-"*68)
    for grp in sorted(set(r["group"] for r in wass_recs)):
        sub = [r for r in wass_recs if r["group"] == grp]
        xs = [r["ss_depth"] for r in sub]
        ys = [r["pct_err"]  for r in sub]
        pr, pp = pearson(xs, ys)
        sr, sp = spearman(xs, ys)
        print(f"  {grp:<30}  {len(sub):>3}  {pr:>+10.3f}  {pp:>8.3f}  {sr:>+10.3f}  {sp:>8.3f}")

print()

# ── Scatter plots ──────────────────────────────────────────────────────────────

if not HAS_PLOT:
    print("Skipping plots (matplotlib not available).")
else:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Structural Recovery vs Parameter Recovery\n"
        "Each point = one calibration mode (one experiment)",
        fontsize=13, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    def scatter_panel(ax, records, x_key, y_key, xlabel, ylabel, title):
        for grp, col in GROUP_COLORS.items():
            sub = [r for r in records if r["group"] == grp]
            if not sub:
                continue
            xs = [r[x_key] for r in sub]
            ys = [r[y_key] for r in sub]
            ax.scatter(xs, ys, color=col, s=60, label=grp.split(":")[0].strip(), zorder=3)
            # label each point
            for r in sub:
                short = r["obj"][:18]
                ax.annotate(short, (r[x_key], r[y_key]),
                            fontsize=5.5, ha="left", va="bottom",
                            xytext=(3, 2), textcoords="offset points")

        # reference line at x=0 (null baseline)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)

        # Pearson r (all groups)
        valid = [(r[x_key], r[y_key]) for r in records
                 if not math.isnan(r[x_key]) and not math.isnan(r[y_key])]
        if len(valid) >= 3:
            xs_all = [v[0] for v in valid]
            ys_all = [v[1] for v in valid]
            pr, pp = pearson(xs_all, ys_all)
            ax.set_title(f"{title}\nr={pr:+.2f}, p={pp:.3f}", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1: SS_angle, SS_rld, Chamfer — all groups
    scatter_panel(fig.add_subplot(gs[0, 0]), voxel_recs,
                  "ss_angle", "pct_err",
                  "SS_angle (voxel zenith angle)", "Mean |% param error|",
                  "SS_angle vs Parameter Error")

    scatter_panel(fig.add_subplot(gs[0, 1]), voxel_recs,
                  "ss_rld", "pct_err",
                  "SS_rld (voxel root density)", "Mean |% param error|",
                  "SS_rld vs Parameter Error")

    scatter_panel(fig.add_subplot(gs[0, 2]), voxel_recs,
                  "chamfer", "pct_err",
                  "Chamfer distance (cm)", "Mean |% param error|",
                  "Chamfer vs Parameter Error")

    # Row 2: Group B only (most informative group)
    grp_b = [r for r in voxel_recs if "100cm_pbc" in r["group"]]

    scatter_panel(fig.add_subplot(gs[1, 0]), grp_b,
                  "ss_angle", "pct_err",
                  "SS_angle", "Mean |% param error|",
                  "Group B only: SS_angle vs Error")

    scatter_panel(fig.add_subplot(gs[1, 1]), grp_b,
                  "ss_rld", "pct_err",
                  "SS_rld", "Mean |% param error|",
                  "Group B only: SS_rld vs Error")

    # Wasserstein SS_depth if available
    if wass_recs:
        wass_b = [r for r in wass_recs if "100cm_pbc" in r["group"]]
        scatter_panel(fig.add_subplot(gs[1, 2]), wass_b,
                      "ss_depth", "pct_err",
                      "SS_depth (Wasserstein, Level 2a)", "Mean |% param error|",
                      "Group B only: SS_depth vs Error")
    else:
        fig.add_subplot(gs[1, 2]).axis("off")

    # Legend
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=col, markersize=8, label=grp.split(":")[0].strip())
               for grp, col in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    out = os.path.join(FIG_DIR, "structural_vs_param_recovery.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)
