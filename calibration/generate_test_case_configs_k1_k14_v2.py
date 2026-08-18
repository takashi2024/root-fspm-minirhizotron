"""
generate_test_case_configs_k1_k14_v2.py

Generates v2 CMA-ES config files for k=1 and k=14 seed count variants.

  k=1  →  tc_v2_{cfg}_{sid}_s2.json   (S2: 4 params, obs_k1)
           tc_v2_{cfg}_{sid}_s1.json   (S1: 8 params, obs_k1)
  k=14 →  tc_v2_{cfg}_{sid}_s2ms_k14.json  (obs_k14)
           tc_v2_{cfg}_{sid}_s1ms_k14.json  (obs_k14)

Prerequisites for k=14:
  calibration/data_v2/test_cases/obs_k14_*.csv must exist.
  Run extract_test_case_obs_k14_v2.py after the dataset extension.
"""

import json
import os

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)
OUT_DIR     = os.path.join(CALIB_DIR, "configs", "test_cases_v2")
DATA_DIR    = os.path.join(CALIB_DIR, "data_v2", "test_cases")
ENV_DIR     = os.path.join(PROJECT_DIR, "sbi", "configs_v2")
WS_DIR      = os.path.join(CALIB_DIR, "data", "test_cases")

os.makedirs(OUT_DIR, exist_ok=True)

CALIBRATION_SEEDS = [11, 22, 99, 88, 33, 55, 77, 7, 13, 42, 66, 91]

PARAMS_S2 = [
    "branch_angle_order1",
    "g_axial",
    "branching_prob_order1",
    "branching_prob_order2",
]

PARAMS_S1 = [
    "elongation_rate_min",
    "elongation_rate_max",
    "elongation_logistic_k",
    "elongation_logistic_t50",
    "branch_angle_order1",
    "g_axial",
    "branching_prob_order1",
    "branching_prob_order2",
]

OBJECTIVE_CONFIG = {"w_angle_mean": 0.3, "angle_normalize_deg": 45.0}

CONFIG_MAP = {
    "A":  "A_v2",
    "B2": "B2_v2",
    "C":  "C_v2",
    "D2": "D2_v2",
}

EXCLUDED = {("A", 497), ("C", 100)}

import pandas as pd
tc_df = pd.read_csv(os.path.join(CALIB_DIR, "test_cases_selected_v2.csv"))

written = 0

for _, tc in tc_df.iterrows():
    cfg       = tc["config"]
    sample_id = int(tc["sample_id"])

    env_json  = os.path.join(ENV_DIR, f"Parameter_SBI_env_{CONFIG_MAP[cfg]}.json")
    ws_known  = os.path.join(WS_DIR,  f"warmstart_{cfg}_{sample_id}.json")
    ws_blind  = os.path.join(CALIB_DIR, "Parameter_VD_template.json")

    obs_k1  = os.path.join(DATA_DIR, f"obs_k1_{cfg}_{sample_id}.csv")
    obs_k14 = os.path.join(DATA_DIR, f"obs_k14_{cfg}_{sample_id}.csv")

    is_slow = (cfg, sample_id) in EXCLUDED

    base = {
        "tube_setup": "1tube",
        "calibration_seeds": CALIBRATION_SEEDS,
        "maxiter": 100,
        "sigma0": 0.1,
        "grid_n_axial": [1],
        "grid_max_order1": [20],
        "calibration_objective": "replicate_seed_average",
        "objective_config": OBJECTIVE_CONFIG,
        "env_json": env_json,
    }

    variants = [
        # (obs_csv, params, warm_start, suffix, include_slow)
        (obs_k1,  PARAMS_S2, ws_known, "s2",       True),
        (obs_k1,  PARAMS_S1, ws_blind, "s1",       True),
        (obs_k14, PARAMS_S2, ws_known, "s2ms_k14", True),
        (obs_k14, PARAMS_S1, ws_blind, "s1ms_k14", True),
    ]

    for obs_csv, params, warm_start, suffix, include_slow in variants:
        tag = f"tc_v2_{cfg}_{sample_id}_{suffix}"
        cfg_dict = {
            **base,
            "calibration_params": params,
            "obs_csv_1tube": obs_csv,
            "warm_start_json": warm_start,
            "experiment_tag": tag,
        }
        out_path = os.path.join(OUT_DIR, f"{tag}.json")
        with open(out_path, "w") as f:
            json.dump(cfg_dict, f, indent=2)
        written += 1

print(f"Written {written} config files to {OUT_DIR}")
