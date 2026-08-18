"""aggregate_ml_structure_s1.py

Combines per-case CSVs from calibration/ml_struct_tmp_s1/ into
comparison_structure_metrics_s1.csv and prints a summary table.
"""

import os
import glob
import pandas as pd
import numpy as np

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR   = os.path.join(CALIB_DIR, "ml_struct_tmp_s1")
OUT_CSV   = os.path.join(CALIB_DIR, "comparison_structure_metrics_s1.csv")
TC_CSV    = os.path.join(CALIB_DIR, "test_cases_selected.csv")

per_case_files = sorted(glob.glob(os.path.join(TMP_DIR, "tc_*_s1.csv")))
if not per_case_files:
    print(f"No per-case CSVs found in {TMP_DIR}/")
    raise SystemExit(1)

dfs = [pd.read_csv(f) for f in per_case_files]
df  = pd.concat(dfs, ignore_index=True)

tc_df   = pd.read_csv(TC_CSV)
n_cases = len(tc_df)
n_found = df[["config", "sample_id"]].drop_duplicates().shape[0]
print(f"Cases found: {n_found} / {n_cases}")
if n_found < n_cases:
    missing = set(zip(tc_df.config, tc_df.sample_id)) - \
              set(zip(df.config, df.sample_id))
    for cfg, sid in sorted(missing):
        print(f"  MISSING: {cfg} {sid}")

df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(df)} rows → {OUT_CSV}")

print("\n" + "=" * 72)
print("Structural Similarity — CMA-ES vs TabICL (Scenario 1, N=20)")
print("=" * 72)

print("\nMethod means:")
for method in ["CMA-ES", "TabICL"]:
    sub = df[df["method"] == method]
    print(f"  {method:<8}  SS_rld={sub['ss_rld'].mean():+.3f}  "
          f"SS_angle={sub['ss_angle'].mean():+.3f}  "
          f"Chamfer={sub['chamfer_cm'].mean():.3f} cm  (N={len(sub)})")

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
