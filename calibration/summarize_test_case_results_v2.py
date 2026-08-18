"""summarize_test_case_results_v2.py

Parameter recovery and obs-similarity summary for v2 CMA-ES calibrations.
Reads tc_v2_*_s2ms and tc_v2_*_s1ms experiment dirs.

Outputs:
  calibration/test_case_results_v2_s2ms.csv
  calibration/test_case_summary_v2_s2ms.txt
  calibration/test_case_results_v2_s1ms.csv
  calibration/test_case_summary_v2_s1ms.txt
"""

import csv, json, math, os

CALIB_DIR      = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR       = os.path.join(CALIB_DIR, "experiments")
TEST_CASES_CSV = os.path.join(CALIB_DIR, "test_cases_selected_v2.csv")

S2MS_PARAMS = ["branch_angle_order1", "g_axial", "branching_prob_order1", "branching_prob_order2"]
S1MS_PARAMS = ["elongation_rate_min", "elongation_rate_max", "elongation_logistic_k",
               "elongation_logistic_t50", "branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

def r_squared(obs, pred):
    n = len(obs)
    if n < 2: return float("nan")
    mean_o = sum(obs) / n
    ss_tot = sum((o - mean_o)**2 for o in obs)
    ss_res = sum((o - p)**2 for o, p in zip(obs, pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

def rmse(obs, pred):
    n = len(obs)
    if n == 0: return float("nan")
    return math.sqrt(sum((o-p)**2 for o, p in zip(obs, pred)) / n)

with open(TEST_CASES_CSV, newline="") as f:
    test_cases = {(row["config"], int(row["sample_id"])): row for row in csv.DictReader(f)}

def process_mode(mode_suffix, calib_params, out_csv_path, out_txt_path, expected_n):
    results, missing = [], []
    print(f"\n{'='*60}")
    print(f"Mode: {mode_suffix}  ({len(calib_params)} params)")
    print(f"{'='*60}")

    for (cfg, sid), tc in sorted(test_cases.items()):
        tag     = f"tc_v2_{cfg}_{sid}{mode_suffix}"
        exp_dir = os.path.join(EXPS_DIR, tag)

        best_params_path = os.path.join(exp_dir, "best_params.json")
        obs_pred_path    = os.path.join(exp_dir, "best_obs_pred_stations.csv")

        if not os.path.isfile(best_params_path):
            print(f"  MISSING: {tag}")
            missing.append(tag)
            continue

        with open(best_params_path) as f:
            best = json.load(f)

        param_errors = {}
        for p in calib_params:
            true_val  = float(tc[p])
            calib_val = float(best.get(p, float("nan")))
            if true_val > 1e-12:
                param_errors[p] = abs(calib_val - true_val) / true_val * 100.0
            else:
                param_errors[p] = float("nan")

        valid_errs = [v for v in param_errors.values() if not math.isnan(v)]
        mean_pct_err = sum(valid_errs) / len(valid_errs) if valid_errs else float("nan")

        r2_all = r2_nz = rmse_all = rmse_nz = float("nan")
        n_all = n_nz = 0
        if os.path.isfile(obs_pred_path):
            obs_vals, pred_vals, obs_nz, pred_nz = [], [], [], []
            with open(obs_pred_path, newline="") as f:
                for row in csv.DictReader(f):
                    o = float(row["obs_cm"]); p = float(row["pred_cm"])
                    obs_vals.append(o); pred_vals.append(p)
                    if o > 0: obs_nz.append(o); pred_nz.append(p)
            r2_all   = r_squared(obs_vals, pred_vals)
            rmse_all = rmse(obs_vals, pred_vals)
            r2_nz    = r_squared(obs_nz, pred_nz)
            rmse_nz  = rmse(obs_nz, pred_nz)
            n_all = len(obs_vals); n_nz = len(obs_nz)

        row_out = {"config": cfg, "sample_id": sid,
                   "g_axial_quintile": tc["g_axial_quintile"],
                   "windows_with_roots": int(tc["windows_with_roots"])}
        for p in calib_params:
            row_out[f"true_{p}"] = float(tc[p])
        for p in calib_params:
            row_out[f"calib_{p}"] = float(best.get(p, float("nan")))
        for p in calib_params:
            row_out[f"pct_err_{p}"] = param_errors[p]
        row_out.update({"mean_pct_err": mean_pct_err,
                        "r2_all": r2_all, "rmse_all": rmse_all,
                        "r2_nz":  r2_nz,  "rmse_nz":  rmse_nz,
                        "n_obs": n_all, "n_obs_nz": n_nz})
        results.append(row_out)
        print(f"  [{cfg} #{sid:>4}] {tc['g_axial_quintile']}  "
              f"mean|%err|={mean_pct_err:5.1f}%  R²_all={r2_all:+.3f}  R²_nz={r2_nz:+.3f}")

    if not results:
        print(f"No results found."); return

    fieldnames = list(results[0].keys())
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(results)
    print(f"\nWrote {len(results)} rows → {out_csv_path}")

    lines = []
    lines.append("=" * 72)
    lines.append(f"Test Case Calibration v2 — {mode_suffix}  ({len(calib_params)} params)")
    lines.append("=" * 72)
    lines.append(f"Completed: {len(results)} / {expected_n}  Missing: {len(missing)}")
    if missing: lines.append(f"  {', '.join(missing)}")
    lines.append("")

    param_labels = [p[:8] for p in calib_params]
    hdr = f"  {'Config':<7} {'#':>4} {'Q':>2}  {'wins':>5}  "
    hdr += "  ".join(f"{'%e_'+lb:>10}" for lb in param_labels)
    hdr += f"  {'mean%err':>9}  {'R²_all':>7}  {'R²_nz':>7}"
    lines.append(hdr); lines.append("-" * len(hdr))

    for r in results:
        line = f"  {r['config']:<7} {r['sample_id']:>4} {r['g_axial_quintile']:>2}  "
        line += f"{r['windows_with_roots']:>5}  "
        for p in calib_params:
            v = r[f"pct_err_{p}"]
            line += f"  {v:>9.1f}%" if not math.isnan(v) else f"  {'—':>10}"
        v_mean = r["mean_pct_err"]
        line += f"  {v_mean:>8.1f}%  {r['r2_all']:>+7.3f}  {r['r2_nz']:>+7.3f}"
        lines.append(line)

    lines.append("-" * len(hdr)); lines.append("")
    overall_err = sum(r["mean_pct_err"] for r in results) / len(results)
    r2_vals = [r["r2_all"] for r in results if not math.isnan(r["r2_all"])]
    overall_r2 = sum(r2_vals) / len(r2_vals) if r2_vals else float("nan")
    lines.append(f"Overall mean |%error|: {overall_err:.1f}%")
    lines.append(f"Overall mean R²_all:   {overall_r2:+.3f}")
    lines.append("")
    lines.append("Per parameter mean |%error|:")
    for p in calib_params:
        errs = [r[f"pct_err_{p}"] for r in results if not math.isnan(r[f"pct_err_{p}"])]
        mean_e = sum(errs) / len(errs) if errs else float("nan")
        lines.append(f"  {p:<35}  {mean_e:>6.1f}%")

    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n--- Summary ---"); print("\n".join(lines))
    print(f"\nWrote summary → {out_txt_path}")

# S2MS: 20 cases (includes A_497 + C_100)
process_mode("_s2ms", S2MS_PARAMS,
             os.path.join(CALIB_DIR, "test_case_results_v2_s2ms.csv"),
             os.path.join(CALIB_DIR, "test_case_summary_v2_s2ms.txt"),
             expected_n=20)

# S1MS: 18 cases (A_497 + C_100 excluded by design)
process_mode("_s1ms", S1MS_PARAMS,
             os.path.join(CALIB_DIR, "test_case_results_v2_s1ms.csv"),
             os.path.join(CALIB_DIR, "test_case_summary_v2_s1ms.txt"),
             expected_n=18)
