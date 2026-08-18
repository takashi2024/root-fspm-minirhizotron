"""
generate_test_case_configs_s1.py

Creates one CMA-ES calibration config JSON per test case (Scenario 1):
  - All 8 parameters calibrated (elongation NOT fixed)
  - warm_start_json = template JSON (CMA-ES starts from template defaults)
  - Objective: replicate_seed_average with w_angle_mean=0.3
  - experiment_tag: tc_<config>_<sample_id>_s1

Output:
  calibration/configs/test_cases/tc_<config>_<sample_id>_s1.json  (20 files)
"""

import csv
import json
import os

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)

TEST_CASES_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
TEMPLATE_JSON  = os.path.join(CALIB_DIR, "Parameter_VD_template.json")
OUT_DIR        = os.path.join(CALIB_DIR, "configs", "test_cases")

os.makedirs(OUT_DIR, exist_ok=True)

CALIBRATION_PARAMS_S1 = [
    "elongation_rate_min",
    "elongation_rate_max",
    "elongation_logistic_k",
    "elongation_logistic_t50",
    "branch_angle_order1",
    "g_axial",
    "branching_prob_order1",
    "branching_prob_order2",
]

CALIBRATION_SEEDS = [11, 22, 99, 88, 33, 55, 77, 7, 13, 42, 66, 91]

with open(TEST_CASES_CSV, newline="") as f:
    test_cases = list(csv.DictReader(f))

for tc in test_cases:
    cfg       = tc["config"]
    sample_id = int(tc["sample_id"])

    tag = f"tc_{cfg}_{sample_id}_s1"

    config = {
        "tube_setup":            "1tube",
        "calibration_seeds":     CALIBRATION_SEEDS,
        "maxiter":               150,
        "sigma0":                0.1,
        "grid_n_axial":          [1],
        "grid_max_order1":       [20],
        "calibration_objective": "replicate_seed_average",
        "calibration_params":    CALIBRATION_PARAMS_S1,
        "objective_config": {
            "w_angle_mean":        0.3,
            "angle_normalize_deg": 45.0,
        },
        "env_json": os.path.join(
            PROJECT_DIR, "sbi", "configs", f"Parameter_SBI_env_{cfg}.json"
        ),
        "obs_csv_1tube": os.path.join(
            CALIB_DIR, "data", "test_cases", f"obs_{cfg}_{sample_id}.csv"
        ),
        "warm_start_json": TEMPLATE_JSON,
        "experiment_tag":        tag,
    }

    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  {tag}.json  [{cfg} g_axial={tc['g_axial'][:7]}  Q{tc['g_axial_quintile'][1]}]")

print(f"\nDone — {len(test_cases)} Scenario 1 config JSONs written to {OUT_DIR}")
