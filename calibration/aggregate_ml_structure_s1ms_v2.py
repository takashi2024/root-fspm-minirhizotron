"""aggregate_ml_structure_s1ms_v2.py

Combines per-case CSVs from ml_struct_tmp_v2_s1ms/ into
comparison_structure_metrics_v2_s1ms.csv and prints a summary.
"""

import os, glob
import pandas as pd
import numpy as np

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR   = os.path.join(CALIB_DIR, "ml_struct_tmp_v2_s1ms")
OUT_CSV   = os.path.join(CALIB_DIR, "comparison_structure_metrics_v2_s1ms.csv")

per_case_files = sorted(glob.glob(os.path.join(TMP_DIR, "tc_v2_*_s1ms.csv")))
if not per_case_files:
    print(f"No per-case CSVs found in {TMP_DIR}/"); raise SystemExit(1)

dfs = [pd.read_csv(f) for f in per_case_files]
df  = pd.concat(dfs, ignore_index=True)

n_cases = df.groupby(["config", "sample_id"]).ngroups
print(f"Loaded {n_cases} cases × 2 methods = {len(df)} rows from {len(per_case_files)} files")
if n_cases < 18:
    print(f"  WARNING: expected 18 cases, got {n_cases}")

df.to_csv(OUT_CSV, index=False)
print(f"Wrote → {OUT_CSV}")

print("\n" + "="*72)
print("Structural Similarity v2 — CMA-ES-S1MS vs TabICL-MS-v2 (Scen1)")
print("="*72)
print(f"{'Method':<16}  {'cfg':<4}  {'#':>4}  {'Q':>2}  {'SS_rld':>8}  {'SS_ang':>8}")
print("-"*72)
for _, r in df.sort_values(["method", "config", "sample_id"]).iterrows():
    print(f"{r['method']:<16}  {r['config']:<4}  {int(r['sample_id']):>4}  "
          f"{r['g_axial_quintile']:>2}  {r['ss_rld']:>+8.3f}  {r['ss_angle']:>+8.3f}")
print("-"*72)

print("\nMethod means:")
for method in df["method"].unique():
    sub = df[df["method"] == method]
    ss_r = sub["ss_rld"].dropna().mean()
    ss_a = sub["ss_angle"].dropna().mean()
    print(f"  {method:<16}  SS_rld={ss_r:+.3f}  SS_angle={ss_a:+.3f}  (N={len(sub)})")

print("\nBy config:")
for method in df["method"].unique():
    print(f"  {method}:")
    for cfg in ["A", "B2", "C", "D2"]:
        sub = df[(df["method"] == method) & (df["config"] == cfg)]
        if sub.empty: continue
        ss_r = sub["ss_rld"].dropna().mean()
        ss_a = sub["ss_angle"].dropna().mean()
        print(f"    {cfg:<3}  SS_rld={ss_r:+.3f}  SS_angle={ss_a:+.3f}  (N={len(sub)})")
