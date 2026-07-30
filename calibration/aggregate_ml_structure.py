"""aggregate_ml_structure.py

Combines per-case CSVs from calibration/ml_struct_tmp/ into a single
comparison_structure_metrics.csv and prints a summary table.

Run after run_ml_structure_parallel.sh completes:
    python3 calibration/aggregate_ml_structure.py
"""

import os
import glob
import pandas as pd
import numpy as np

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR   = os.path.join(CALIB_DIR, "ml_struct_tmp")
OUT_CSV   = os.path.join(CALIB_DIR, "comparison_structure_metrics.csv")
TC_CSV    = os.path.join(CALIB_DIR, "test_cases_selected.csv")

# ── Collect per-case CSVs ─────────────────────────────────────────────────────

per_case_files = sorted(glob.glob(os.path.join(TMP_DIR, "tc_*.csv")))
if not per_case_files:
    print(f"No per-case CSVs found in {TMP_DIR}/")
    raise SystemExit(1)

dfs = [pd.read_csv(f) for f in per_case_files]
df  = pd.concat(dfs, ignore_index=True)

# Check completeness
tc_df   = pd.read_csv(TC_CSV)
n_cases = len(tc_df)
n_found = df[["config", "sample_id"]].drop_duplicates().shape[0]
print(f"Cases found: {n_found} / {n_cases}")
if n_found < n_cases:
    missing = set(zip(tc_df.config, tc_df.sample_id)) - \
              set(zip(df.config, df.sample_id))
    for cfg, sid in sorted(missing):
        print(f"  MISSING: {cfg} {sid}")

# ── Write combined CSV ────────────────────────────────────────────────────────

df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(df)} rows → {OUT_CSV}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("Structural Similarity — CMA-ES vs XGBoost (Scenario 2, N=20)")
print("=" * 72)
print(f"  {'Method':<8}  {'config':<6}  {'#':>4}  {'Q':>2}  "
      f"{'SS_rld':>8}  {'SS_angle':>8}  {'Chamfer':>8}")
print("-" * 72)

for _, r in df.sort_values(["config", "sample_id", "method"]).iterrows():
    print(f"  {r['method']:<8}  {r['config']:<6}  {int(r['sample_id']):>4}  "
          f"{r['g_axial_quintile']:>2}  "
          f"{r['ss_rld']:>+8.3f}  {r['ss_angle']:>+8.3f}  {r['chamfer_cm']:>8.3f}")

print("-" * 72)
print("\nMethod means:")
for method in ["CMA-ES", "TabICL"]:
    sub = df[df["method"] == method]
    print(f"  {method:<8}  SS_rld={sub['ss_rld'].mean():+.3f}  "
          f"SS_angle={sub['ss_angle'].mean():+.3f}  "
          f"Chamfer={sub['chamfer_cm'].mean():.3f} cm  (N={len(sub)})")

# ── Per-quintile breakdown ────────────────────────────────────────────────────

print("\nBy g_axial quintile:")
print(f"  {'Method':<8}  {'Q':<3}  {'SS_rld':>8}  {'SS_angle':>8}  {'Chamfer':>8}  N")
for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    for method in ["CMA-ES", "TabICL"]:
        sub = df[(df["method"] == method) & (df["g_axial_quintile"] == q)]
        if len(sub) == 0:
            continue
        print(f"  {method:<8}  {q:<3}  "
              f"{sub['ss_rld'].mean():>+8.3f}  "
              f"{sub['ss_angle'].mean():>+8.3f}  "
              f"{sub['chamfer_cm'].mean():>8.3f}  {len(sub)}")
