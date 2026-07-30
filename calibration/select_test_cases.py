"""
select_test_cases.py

Selects 20 representative test cases from the N=2000 synthetic dataset using
a stratified design that ensures:
  1. Adequate observation signal for CMA-ES (>= 10 occupied viewing windows)
  2. Coverage of the gravitropism identifiability spectrum (g_axial quintiles)
  3. Balance across all 4 planting configurations (A, B2, C, D2)

Algorithm (per config):
  1. Compute windows_with_roots from X_fov.npz tensors (binary image stack).
  2. Filter: feasible=1 AND windows_with_roots >= 10.
  3. Divide the filtered samples into 5 equal g_axial quintiles (Q1=lowest, Q5=highest).
  4. Select 1 case from each quintile using constrained greedy maximin in 3D
     parameter space {branch_angle_order1, branching_prob_order1, branching_prob_order2}:
       Q1: sample closest to the centroid of the config's filtered set (first pick; no prior).
       Q2-Q5: sample within that quintile maximizing min-distance to all prior picks.
     If a quintile is empty, an unconstrained greedy step fills the slot.

Configs: A, B2, C, D2 — these are the configs used in the SBI combined training set
(B2/D2 have tube repositioned to 12.5 cm from plant column, matching A/C geometry).

Usage:
  python calibration/select_test_cases.py

Output:
  calibration/test_cases_selected.csv  — 20 selected cases with all metadata
  calibration/test_cases_summary.txt   — human-readable summary
"""

import csv
import math
import os

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────

CONFIGS       = ["A", "B2", "C", "D2"]
N_PER_CONFIG  = 5             # 5 × 4 configs = 20 total
MIN_WINDOWS   = 10            # minimum occupied viewing windows (out of 60)
N_QUINTILES   = 5

# Greedy maximin uses this 3D subspace (g_axial is handled by quintile stratification)
DIST_PARAMS   = ["branch_angle_order1", "branching_prob_order1", "branching_prob_order2"]
ALL_PARAMS    = ["branch_angle_order1", "g_axial", "branching_prob_order1", "branching_prob_order2"]

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(CALIB_DIR, "..", "sbi", "data")
OUT_CSV   = os.path.join(CALIB_DIR, "test_cases_selected.csv")
OUT_TXT   = os.path.join(CALIB_DIR, "test_cases_summary.txt")

# ── Utility ───────────────────────────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def _load_npz(path):
    """Load first array from an NPZ file regardless of key name."""
    data = np.load(path)
    return data[data.files[0]]

# ── Step 1: compute windows_with_roots from binary image tensors ──────────────

def compute_windows_with_roots(config):
    """
    Returns dict sample_id -> int (count of viewing windows where any root pixel
    is nonzero on any of the 11 snapshot days).
    """
    tensor_dir = os.path.join(DATA_DIR, f"config_{config}", "tensors")
    X_fov   = _load_npz(os.path.join(tensor_dir, "X_fov.npz"))        # (N_win, 11, 64, 64)
    win_ids = _load_npz(os.path.join(tensor_dir, "X_fov_window_ids.npz"))  # (N_win, 3)

    # A window has roots if any pixel is nonzero across any day
    has_root   = X_fov.reshape(len(X_fov), -1).sum(axis=1) > 0        # (N_win,) bool
    sample_ids = win_ids[:, 0]                                          # 1-indexed sample ids

    counts = {}
    for sid, hr in zip(sample_ids.tolist(), has_root.tolist()):
        counts[int(sid)] = counts.get(int(sid), 0) + int(hr)
    return counts

# ── Step 2: load samples CSV + join window counts + filter ────────────────────

def load_samples(config):
    win_counts = compute_windows_with_roots(config)
    path = os.path.join(DATA_DIR, f"config_{config}", "samples.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("feasible", 1)) != 1:
                continue
            sid = int(row["sample_id"])
            n_win = win_counts.get(sid, 0)
            if n_win < MIN_WINDOWS:
                continue
            row["windows_with_roots"] = n_win
            rows.append(row)
    return rows

print("Loading samples and computing window counts...")
all_samples = {}
for cfg in CONFIGS:
    all_samples[cfg] = load_samples(cfg)
    n_total = sum(1 for _ in open(
        os.path.join(DATA_DIR, f"config_{cfg}", "samples.csv"))) - 1
    print(f"  Config {cfg}: {len(all_samples[cfg])} / {n_total} pass filter "
          f"(feasible + windows_with_roots >= {MIN_WINDOWS})")

# ── Step 3: global normalization bounds (pooled filtered set across all configs) ──

global_min = {p: math.inf  for p in ALL_PARAMS}
global_max = {p: -math.inf for p in ALL_PARAMS}

for cfg in CONFIGS:
    for row in all_samples[cfg]:
        for p in ALL_PARAMS:
            v = float(row[p])
            global_min[p] = min(global_min[p], v)
            global_max[p] = max(global_max[p], v)

print("\nNormalization bounds (pooled filtered set):")
for p in ALL_PARAMS:
    print(f"  {p}: [{global_min[p]:.4f}, {global_max[p]:.4f}]")

def norm3d(row):
    """Normalize 3D greedy params to [0,1] using pooled filtered-set range."""
    return [
        (float(row[p]) - global_min[p]) / max(global_max[p] - global_min[p], 1e-12)
        for p in DIST_PARAMS
    ]

def norm4d(row):
    return [
        (float(row[p]) - global_min[p]) / max(global_max[p] - global_min[p], 1e-12)
        for p in ALL_PARAMS
    ]

# ── Step 4: constrained greedy maximin within quintiles ───────────────────────

def assign_quintiles(rows):
    """Return list of quintile labels (0..4) for each row, based on g_axial rank."""
    n = len(rows)
    order = sorted(range(n), key=lambda i: float(rows[i]["g_axial"]))
    labels = [0] * n
    for rank, idx in enumerate(order):
        labels[idx] = min(N_QUINTILES - 1, rank * N_QUINTILES // n)
    return labels   # 0=lowest g_axial (Q1), 4=highest (Q5)

def select_for_config(rows):
    """
    Select N_PER_CONFIG cases from rows using g_axial quintile stratification
    with constrained greedy maximin in 3D parameter space.
    Returns (selected_indices, quintile_labels).
    """
    quintile_labels = assign_quintiles(rows)
    X3d = [norm3d(r) for r in rows]
    n   = len(rows)
    centroid = [sum(X3d[i][j] for i in range(n)) / n for j in range(3)]

    selected_indices = []   # indices into `rows`
    selected_X3d     = []   # their 3D coordinates

    for q in range(N_QUINTILES):
        q_idxs = [i for i in range(n) if quintile_labels[i] == q]

        if not q_idxs:
            # Empty quintile — unconstrained fill from remaining samples
            remaining = [i for i in range(n) if i not in selected_indices]
            if not remaining:
                break
            if not selected_X3d:
                best = min(remaining, key=lambda i: euclidean(X3d[i], centroid))
            else:
                best = max(remaining,
                           key=lambda i: min(euclidean(X3d[i], s) for s in selected_X3d))
        elif not selected_X3d:
            # Q1: centroid-closest within the quintile
            best = min(q_idxs, key=lambda i: euclidean(X3d[i], centroid))
        else:
            # Q2-Q5: max min-distance to all prior picks
            best = max(q_idxs,
                       key=lambda i: min(euclidean(X3d[i], s) for s in selected_X3d))

        selected_indices.append(best)
        selected_X3d.append(X3d[best])

    return selected_indices, quintile_labels

# ── Run selection ─────────────────────────────────────────────────────────────

selected_rows = []

for cfg in CONFIGS:
    rows = all_samples[cfg]
    sel_idxs, q_labels = select_for_config(rows)

    print(f"\nConfig {cfg} — selected {len(sel_idxs)} cases:")
    for rank, idx in enumerate(sel_idxs, 1):
        r   = rows[idx]
        qno = q_labels[idx] + 1   # 1-indexed label
        print(f"  [Q{qno}] sample_id={r['sample_id']:>4s}  "
              f"ba={float(r['branch_angle_order1']):.1f}°  "
              f"g={float(r['g_axial']):.4f}  "
              f"bp1={float(r['branching_prob_order1']):.3f}  "
              f"bp2={float(r['branching_prob_order2']):.3f}  "
              f"wins={r['windows_with_roots']}")
        row_out = dict(r)
        row_out["config"]           = cfg
        row_out["selection_rank"]   = rank
        row_out["g_axial_quintile"] = f"Q{qno}"
        x3 = norm3d(r)
        row_out["x_norm_" + DIST_PARAMS[0]] = round(x3[0], 4)
        row_out["x_norm_" + DIST_PARAMS[1]] = round(x3[1], 4)
        row_out["x_norm_" + DIST_PARAMS[2]] = round(x3[2], 4)
        x4 = norm4d(r)
        row_out["x_norm_g_axial"] = round(x4[1], 4)
        selected_rows.append(row_out)

# ── Write CSV ─────────────────────────────────────────────────────────────────

fieldnames = list(selected_rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(selected_rows)

print(f"\nWrote {len(selected_rows)} selected cases → {OUT_CSV}")

# ── Write summary ─────────────────────────────────────────────────────────────

def pct_of_range(val, p):
    return 100.0 * (val - global_min[p]) / max(global_max[p] - global_min[p], 1e-12)

lines = []
lines.append("=" * 80)
lines.append("Test Case Selection — Stratified Greedy Maximin Design")
lines.append("=" * 80)
lines.append(f"Total selected : {len(selected_rows)} cases ({N_PER_CONFIG} per config × {len(CONFIGS)} configs)")
lines.append(f"Configs        : {', '.join(CONFIGS)}")
lines.append(f"Algorithm      : constrained greedy maximin in 3D space {{{', '.join(DIST_PARAMS)}}}")
lines.append(f"                 stratified by g_axial quintile (1 case per quintile)")
lines.append(f"Filter         : feasible=1, windows_with_roots >= {MIN_WINDOWS}")
lines.append(f"Normalization  : pooled filtered set across all {sum(len(v) for v in all_samples.values())} samples")
lines.append("")
lines.append("Parameter ranges (pooled filtered set min → max):")
for p in ALL_PARAMS:
    lines.append(f"  {p:<30s}: [{global_min[p]:.4f}, {global_max[p]:.4f}]")
lines.append("")
lines.append("Selected cases:")
lines.append(f"  {'#':<3} {'Config':<7} {'Quintile':<9} {'sample_id':<10} "
             f"{'ba(°)':<8} {'g_axial':<9} {'bp_o1':<8} {'bp_o2':<8} {'wins':>5}")
lines.append("-" * 80)
for i, r in enumerate(selected_rows, 1):
    ba  = float(r["branch_angle_order1"])
    g   = float(r["g_axial"])
    bp1 = float(r["branching_prob_order1"])
    bp2 = float(r["branching_prob_order2"])
    lines.append(
        f"  {i:<3} {r['config']:<7} {r['g_axial_quintile']:<9} {r['sample_id']:<10} "
        f"{ba:<8.1f} {g:<9.4f} {bp1:<8.3f} {bp2:<8.3f} {r['windows_with_roots']:>5}"
    )
lines.append("-" * 80)
lines.append("")

# Coverage stats
lines.append("Coverage statistics (selected 20 vs pooled filtered set):")
for p in ALL_PARAMS:
    vals_sel = [float(r[p]) for r in selected_rows]
    vals_all = [float(r[p]) for cfg in CONFIGS for r in all_samples[cfg]]
    rng_sel  = max(vals_sel) - min(vals_sel)
    rng_all  = max(vals_all) - min(vals_all)
    lines.append(f"  {p:<30s}: range covered = {100*rng_sel/rng_all:.1f}%  "
                 f"mean selected={sum(vals_sel)/len(vals_sel):.4f}  "
                 f"mean all={sum(vals_all)/len(vals_all):.4f}")
lines.append("")

# g_axial range per quintile across configs
lines.append("g_axial range per quintile (across all configs):")
for q in range(1, N_QUINTILES + 1):
    q_rows = [r for r in selected_rows if r["g_axial_quintile"] == f"Q{q}"]
    if not q_rows:
        continue
    g_vals = [float(r["g_axial"]) for r in q_rows]
    lines.append(f"  Q{q}: {min(g_vals):.4f} – {max(g_vals):.4f}  "
                 f"(mean wins: {sum(int(r['windows_with_roots']) for r in q_rows)//len(q_rows)})")
lines.append("")

# Min pairwise distance
def min_pairwise(rows_subset):
    X = [norm4d(r) for r in rows_subset]
    best = math.inf
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            d = euclidean(X[i], X[j])
            if d < best:
                best = d
    return best

lines.append("Min pairwise distance (normalized 4D space, for separation check):")
for cfg in CONFIGS:
    cfg_rows = [r for r in selected_rows if r["config"] == cfg]
    lines.append(f"  Config {cfg}: {min_pairwise(cfg_rows):.4f}")
lines.append(f"  All 20    : {min_pairwise(selected_rows):.4f}")
lines.append("")
lines.append(f"All {len(selected_rows)} selected cases have windows_with_roots >= {MIN_WINDOWS}: "
             f"{all(int(r['windows_with_roots']) >= MIN_WINDOWS for r in selected_rows)}")
lines.append("")
lines.append("Reference: true parameter values used in current calibration experiments:")
lines.append("  branch_angle_order1=80.0°  g_axial=0.05  bp_o1=0.20  bp_o2=0.30")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote summary → {OUT_TXT}")
print("\n--- Summary ---")
print("\n".join(lines))
