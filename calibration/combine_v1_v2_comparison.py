"""combine_v1_v2_comparison.py

Merges v1 (skin_depth=1.0 cm) and v2 (skin_depth=0.2 cm) results into
combined CSVs suitable for skin-depth sensitivity figures.

Adds `skin_depth_cm` column (1.0 or 0.2) and normalises method labels:
  CMA-ES-S2MS  →  CMA-ES-S2MS   (unchanged; same across versions)
  TabICL-MS    →  TabICL-MS     (v1 name kept; v2 "TabICL-MS-v2" mapped here)
  CMA-ES-S1MS  →  CMA-ES-S1MS
  TabICL-MS-k1 →  TabICL-MS-k1

Outputs
-------
  calibration/comparison_structure_metrics_combined_s2ms.csv
  calibration/comparison_structure_metrics_combined_s1ms.csv
  calibration/test_case_results_combined_s2ms.csv
  calibration/test_case_results_combined_s2ms_summary.txt
"""

import os
import pandas as pd

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Method label normalisation ────────────────────────────────────────────────
# Map v2-specific labels back to canonical names so rows are comparable across versions.
METHOD_NORM = {
    "TabICL-MS-v2": "TabICL-MS",
    "CMA-ES-S2MS":  "CMA-ES-S2MS",
    "CMA-ES-S1MS":  "CMA-ES-S1MS",
    "TabICL-MS":    "TabICL-MS",
}

def load_struct(path, skin_depth):
    df = pd.read_csv(path)
    df["skin_depth_cm"] = skin_depth
    df["method"] = df["method"].map(lambda m: METHOD_NORM.get(m, m))
    return df

def merge_struct(v1_path, v2_path, out_path, label):
    print(f"\n{'='*60}")
    print(f"Structural — {label}")
    print(f"{'='*60}")
    v1 = load_struct(v1_path, 1.0)
    v2 = load_struct(v2_path, 0.2)
    # Align columns: v1 may have chamfer_cm that v2 lacks; fill with NaN
    combined = pd.concat([v1, v2], ignore_index=True, sort=False)
    # Move skin_depth_cm to front
    cols = ["skin_depth_cm"] + [c for c in combined.columns if c != "skin_depth_cm"]
    combined = combined[cols]
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows → {out_path}")

    print(f"\n{'Method':<14}  {'skin_depth':>10}  {'N':>4}  {'SS_rld':>8}  {'SS_angle':>8}")
    print("-" * 54)
    for (method, sd), grp in combined.groupby(["method", "skin_depth_cm"]):
        n  = len(grp)
        sr = grp["ss_rld"].mean()
        sa = grp["ss_angle"].mean()
        print(f"{method:<14}  {sd:>10.1f}  {n:>4}  {sr:>+8.3f}  {sa:>+8.3f}")

def merge_param_recovery(v1_path, v2_path, out_path, out_txt, params):
    print(f"\n{'='*60}")
    print(f"Parameter recovery — S2MS (4 arch params)")
    print(f"{'='*60}")
    v1 = pd.read_csv(v1_path); v1["skin_depth_cm"] = 1.0
    v2 = pd.read_csv(v2_path); v2["skin_depth_cm"] = 0.2
    combined = pd.concat([v1, v2], ignore_index=True, sort=False)
    cols = ["skin_depth_cm"] + [c for c in combined.columns if c != "skin_depth_cm"]
    combined = combined[cols]
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows → {out_path}")

    lines = []
    lines.append("=" * 64)
    lines.append("Parameter recovery: skin_depth 1.0 cm  vs  0.2 cm  (S2MS)")
    lines.append("=" * 64)
    lines.append(f"{'Param':<35}  {'1.0 cm':>8}  {'0.2 cm':>8}  {'delta':>8}")
    lines.append("-" * 64)
    for p in params:
        col = f"pct_err_{p}"
        if col not in combined.columns:
            continue
        e1 = combined[combined.skin_depth_cm == 1.0][col].dropna().mean()
        e2 = combined[combined.skin_depth_cm == 0.2][col].dropna().mean()
        lines.append(f"  {p:<33}  {e1:>7.1f}%  {e2:>7.1f}%  {e2-e1:>+7.1f}%")
    lines.append("-" * 64)
    col = "mean_pct_err"
    e1 = combined[combined.skin_depth_cm == 1.0][col].dropna().mean()
    e2 = combined[combined.skin_depth_cm == 0.2][col].dropna().mean()
    lines.append(f"  {'Overall mean':33}  {e1:>7.1f}%  {e2:>7.1f}%  {e2-e1:>+7.1f}%")

    with open(out_txt, "w") as f: f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote → {out_txt}")


# ── S2MS structural ───────────────────────────────────────────────────────────
merge_struct(
    v1_path  = os.path.join(CALIB_DIR, "comparison_structure_metrics_s2ms.csv"),
    v2_path  = os.path.join(CALIB_DIR, "comparison_structure_metrics_v2_s2ms.csv"),
    out_path = os.path.join(CALIB_DIR, "comparison_structure_metrics_combined_s2ms.csv"),
    label    = "S2MS (4 arch params)",
)

# ── S1MS structural ───────────────────────────────────────────────────────────
merge_struct(
    v1_path  = os.path.join(CALIB_DIR, "comparison_structure_metrics_s1ms.csv"),
    v2_path  = os.path.join(CALIB_DIR, "comparison_structure_metrics_v2_s1ms.csv"),
    out_path = os.path.join(CALIB_DIR, "comparison_structure_metrics_combined_s1ms.csv"),
    label    = "S1MS (8 params)",
)

# ── S2MS parameter recovery ───────────────────────────────────────────────────
S2MS_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]
merge_param_recovery(
    v1_path  = os.path.join(CALIB_DIR, "test_case_results.csv"),
    v2_path  = os.path.join(CALIB_DIR, "test_case_results_v2_s2ms.csv"),
    out_path = os.path.join(CALIB_DIR, "test_case_results_combined_s2ms.csv"),
    out_txt  = os.path.join(CALIB_DIR, "test_case_results_combined_s2ms_summary.txt"),
    params   = S2MS_PARAMS,
)
