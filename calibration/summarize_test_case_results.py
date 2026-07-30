"""
summarize_test_case_results.py

Aggregates results from all 20 Scenario 2 CMA-ES calibrations.
Requires calibrations to have completed (run_test_case_calibrations.sh).

For each test case, reads:
  - best_params.json         → recovered parameter values
  - best_obs_pred_stations.csv → obs vs pred root length at each station/day/azimuth

Computes:
  (a) Parameter recovery:  mean |% error| per parameter and overall
  (b) Obs similarity:      R² and RMSE (root_length_cm, obs vs pred)

Output:
  calibration/test_case_results.csv   — one row per test case (full results)
  calibration/test_case_summary.txt   — grouped summary tables
"""

import csv
import json
import math
import os

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR    = os.path.join(CALIB_DIR, "experiments")
TEST_CASES_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
OUT_CSV     = os.path.join(CALIB_DIR, "test_case_results.csv")
OUT_TXT     = os.path.join(CALIB_DIR, "test_case_summary.txt")

CALIB_PARAMS = ["branch_angle_order1", "g_axial", "branching_prob_order1", "branching_prob_order2"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def r_squared(obs, pred):
    n = len(obs)
    if n < 2:
        return float("nan")
    mean_o = sum(obs) / n
    ss_tot = sum((o - mean_o) ** 2 for o in obs)
    ss_res = sum((o - p) ** 2 for o, p in zip(obs, pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

def rmse(obs, pred):
    n = len(obs)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(obs, pred)) / n)

# ── Load true params ───────────────────────────────────────────────────────────

with open(TEST_CASES_CSV, newline="") as f:
    test_cases = {
        (row["config"], int(row["sample_id"])): row
        for row in csv.DictReader(f)
    }

# ── Collect results per test case ──────────────────────────────────────────────

results = []
missing = []

for (cfg, sid), tc in sorted(test_cases.items()):
    tag     = f"tc_{cfg}_{sid}_s2"
    exp_dir = os.path.join(EXPS_DIR, tag)

    best_params_path = os.path.join(exp_dir, "best_params.json")
    obs_pred_path    = os.path.join(exp_dir, "best_obs_pred_stations.csv")

    if not os.path.isfile(best_params_path):
        print(f"  MISSING best_params.json: {tag}")
        missing.append(tag)
        continue

    with open(best_params_path) as f:
        best = json.load(f)

    # ── (a) Parameter recovery ────────────────────────────────────────────────
    param_errors = {}
    for p in CALIB_PARAMS:
        true_val  = float(tc[p])
        calib_val = float(best.get(p, float("nan")))
        if true_val > 1e-12:
            param_errors[p] = abs(calib_val - true_val) / true_val * 100.0
        else:
            param_errors[p] = float("nan")

    mean_pct_err = sum(v for v in param_errors.values() if not math.isnan(v)) / \
                   sum(1 for v in param_errors.values() if not math.isnan(v))

    # ── (b) Obs similarity ────────────────────────────────────────────────────
    r2_all = r2_nz = rmse_all = rmse_nz = float("nan")
    n_all  = n_nz  = 0

    if os.path.isfile(obs_pred_path):
        obs_vals = []; pred_vals = []
        obs_nz = []; pred_nz = []
        with open(obs_pred_path, newline="") as f:
            for row in csv.DictReader(f):
                o = float(row["obs_cm"])
                p = float(row["pred_cm"])
                obs_vals.append(o); pred_vals.append(p)
                if o > 0:
                    obs_nz.append(o); pred_nz.append(p)
        r2_all   = r_squared(obs_vals, pred_vals)
        rmse_all = rmse(obs_vals, pred_vals)
        r2_nz    = r_squared(obs_nz, pred_nz)
        rmse_nz  = rmse(obs_nz, pred_nz)
        n_all    = len(obs_vals)
        n_nz     = len(obs_nz)

    row_out = {
        "config":           cfg,
        "sample_id":        sid,
        "g_axial_quintile": tc["g_axial_quintile"],
        "windows_with_roots": int(tc["windows_with_roots"]),
    }
    # True params
    for p in CALIB_PARAMS:
        row_out[f"true_{p}"] = float(tc[p])
    # Calibrated params
    for p in CALIB_PARAMS:
        row_out[f"calib_{p}"] = float(best.get(p, float("nan")))
    # Parameter errors
    for p in CALIB_PARAMS:
        row_out[f"pct_err_{p}"] = param_errors[p]
    row_out["mean_pct_err"] = mean_pct_err
    # Obs similarity
    row_out["r2_all"]   = r2_all
    row_out["rmse_all"] = rmse_all
    row_out["r2_nz"]    = r2_nz
    row_out["rmse_nz"]  = rmse_nz
    row_out["n_obs"]    = n_all
    row_out["n_obs_nz"] = n_nz

    results.append(row_out)
    print(f"  [{cfg} #{sid:>4}] {tc['g_axial_quintile']}  "
          f"mean|%err|={mean_pct_err:5.1f}%  "
          f"R²_all={r2_all:+.3f}  R²_nz={r2_nz:+.3f}")

if not results:
    print(f"No completed calibrations found in {EXPS_DIR}.")
    print(f"Missing: {len(missing)} / {len(test_cases)} calibrations")
    raise SystemExit(1)

# ── Write CSV ──────────────────────────────────────────────────────────────────

fieldnames = list(results[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

print(f"\nWrote {len(results)} rows → {OUT_CSV}")

# ── Summary ────────────────────────────────────────────────────────────────────

lines = []
lines.append("=" * 72)
lines.append("Test Case Calibration Results — Scenario 2 (4-param)")
lines.append("=" * 72)
lines.append(f"Calibrations completed: {len(results)} / {len(test_cases)}")
if missing:
    lines.append(f"Missing: {', '.join(missing)}")
lines.append("")

# Header row
hdr = f"  {'Config':<7} {'#':>4} {'Q':>2}  {'wins':>5}  "
hdr += "  ".join(f"{'err_'+p[:7]:>10}" for p in CALIB_PARAMS)
hdr += f"  {'mean%err':>9}  {'R²_all':>7}  {'R²_nz':>7}"
lines.append(hdr)
lines.append("-" * len(hdr))

for r in results:
    line = f"  {r['config']:<7} {r['sample_id']:>4} {r['g_axial_quintile']:>2}  "
    line += f"{r['windows_with_roots']:>5}  "
    for p in CALIB_PARAMS:
        v = r[f"pct_err_{p}"]
        line += f"  {v:>9.1f}%" if not math.isnan(v) else f"  {'—':>10}"
    v_mean = r["mean_pct_err"]
    line += f"  {v_mean:>8.1f}%"
    r2a = r["r2_all"]
    r2n = r["r2_nz"]
    line += f"  {r2a:>+7.3f}  {r2n:>+7.3f}"
    lines.append(line)

lines.append("-" * len(hdr))

# Summary stats per quintile
lines.append("")
lines.append("Per g_axial quintile:")
for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    sub = [r for r in results if r["g_axial_quintile"] == q]
    if not sub:
        continue
    mean_e = sum(r["mean_pct_err"] for r in sub) / len(sub)
    mean_r2 = sum(r["r2_all"] for r in sub if not math.isnan(r["r2_all"])) / max(1, sum(1 for r in sub if not math.isnan(r["r2_all"])))
    lines.append(f"  {q}: n={len(sub):>2}  mean_pct_err={mean_e:.1f}%  mean_R²_all={mean_r2:+.3f}")

lines.append("")
overall_err  = sum(r["mean_pct_err"] for r in results) / len(results)
r2_vals      = [r["r2_all"] for r in results if not math.isnan(r["r2_all"])]
overall_r2   = sum(r2_vals) / len(r2_vals) if r2_vals else float("nan")
lines.append(f"Overall mean |%error|: {overall_err:.1f}%")
lines.append(f"Overall mean R²_all:   {overall_r2:+.3f}")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n--- Summary ---")
print("\n".join(lines))
print(f"\nWrote summary → {OUT_TXT}")
