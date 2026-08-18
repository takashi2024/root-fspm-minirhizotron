"""aggregate_ml_structure_s2ms_k14.py

Combines per-case CSVs from calibration/ml_struct_tmp_s2ms_k14/ into a single
comparison_structure_metrics_s2ms_k14.csv and prints a summary table.

Run after evaluate_ml_structure_s2ms_k14.jl completes for all 18 cases:
    python3 calibration/aggregate_ml_structure_s2ms_k14.py
"""

import os
import glob
import pandas as pd
import numpy as np

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR   = os.path.join(CALIB_DIR, "ml_struct_tmp_s2ms_k14")
OUT_CSV   = os.path.join(CALIB_DIR, "comparison_structure_metrics_s2ms_k14.csv")
TC_CSV    = os.path.join(CALIB_DIR, "test_cases_selected.csv")

# ── Collect per-case CSVs ─────────────────────────────────────────────────────

per_case_files = sorted(glob.glob(os.path.join(TMP_DIR, "tc_*_s2ms.csv")))
if not per_case_files:
    print(f"No per-case CSVs found in {TMP_DIR}/")
    raise SystemExit(1)

dfs = [pd.read_csv(f) for f in per_case_files]
df  = pd.concat(dfs, ignore_index=True)

# Completeness check (expect 18 cases × 1 method = 18 rows)
n_cases = df.groupby(["config", "sample_id"]).ngroups
print(f"Loaded {n_cases} cases × 1 method = {len(df)} rows from {len(per_case_files)} files")
if n_cases < 18:
    print(f"  WARNING: expected 18 cases, got {n_cases}")

df.to_csv(OUT_CSV, index=False)
print(f"Wrote → {OUT_CSV}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("Structural Similarity — TabICL-MS-k14 (Scen2, N=18)")
print("="*72)
print(f"{'Method':<14}  {'cfg':<4}  {'#':>4}  {'Q':>2}  {'SS_rld':>8}  {'SS_ang':>8}  {'Chamfer':>8}")
print("-"*72)
for _, r in df.sort_values(["method", "config", "sample_id"]).iterrows():
    print(f"{r['method']:<14}  {r['config']:<4}  {int(r['sample_id']):>4}  "
          f"{r['g_axial_quintile']:>2}  {r['ss_rld']:>+8.3f}  "
          f"{r['ss_angle']:>+8.3f}  {r['chamfer_cm']:>8.3f}")
print("-"*72)

print("\nMethod means (N=18):")
for method in ["TabICL-MS-k14"]:
    sub = df[df["method"] == method]
    if sub.empty:
        continue
    ss_r = sub["ss_rld"].dropna().mean()
    ss_a = sub["ss_angle"].dropna().mean()
    chf  = sub["chamfer_cm"].dropna().mean()
    n    = len(sub)
    print(f"  {method:<14}  SS_rld={ss_r:+.3f}  SS_angle={ss_a:+.3f}  Chamfer={chf:.3f} cm  (N={n})")

print("\nBy config:")
for method in ["TabICL-MS-k14"]:
    print(f"  {method}:")
    for cfg in ["A", "B2", "C", "D2"]:
        sub = df[(df["method"] == method) & (df["config"] == cfg)]
        if sub.empty:
            continue
        ss_r = sub["ss_rld"].dropna().mean()
        ss_a = sub["ss_angle"].dropna().mean()
        chf  = sub["chamfer_cm"].dropna().mean()
        print(f"    {cfg:<3}  SS_rld={ss_r:+.3f}  SS_angle={ss_a:+.3f}  Chamfer={chf:.3f} cm  (N={len(sub)})")
