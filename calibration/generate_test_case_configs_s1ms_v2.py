"""
generate_test_case_configs_s1ms_v2.py

Generates CMA-ES calibration config JSONs for Scenario 1 Multi-Seed v2 (S1MS v2):
  - Observations: K=7 multi-seed mean from sbi/data/multi_seed_v2 (skin_depth=0.2 cm)
  - Calibrated params: all 8 (elongation + architecture)
  - sigma0: 0.10 (cold-start from template)
  - maxiter: 100
  - env_json: sbi/configs_v2/Parameter_SBI_env_*_v2.json (skin_depth=0.2 cm)
  - tag: tc_v2_<config>_<sample_id>_s1ms

Excludes A_497 and C_100 from Phase 1 (run separately with slow runner).

Output: calibration/configs/test_cases_v2/tc_v2_*_s1ms.json  (18 files)
"""

import json
import os
import pandas as pd

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)
OUT_DIR     = os.path.join(CALIB_DIR, "configs", "test_cases_v2")
DATA_V2     = os.path.join(CALIB_DIR, "data_v2", "test_cases")
SBI_V2      = os.path.join(PROJECT_DIR, "sbi", "configs_v2")
TEMPLATE    = os.path.join(CALIB_DIR, "Parameter_VD_template.json")

os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDED = {("A", 497), ("C", 100)}

CALIBRATION_SEEDS = [11, 22, 99, 88, 33, 55, 77, 7, 13, 42, 66, 91]
ALL_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
    "branch_angle_order1", "g_axial",
    "branching_prob_order1", "branching_prob_order2",
]

ENV_JSON_V2 = {
    "A":  os.path.join(SBI_V2, "Parameter_SBI_env_A_v2.json"),
    "B2": os.path.join(SBI_V2, "Parameter_SBI_env_B2_v2.json"),
    "C":  os.path.join(SBI_V2, "Parameter_SBI_env_C_v2.json"),
    "D2": os.path.join(SBI_V2, "Parameter_SBI_env_D2_v2.json"),
}

tc_df = pd.read_csv(os.path.join(CALIB_DIR, "test_cases_selected_v2.csv"))
tc_18 = tc_df[~tc_df.apply(lambda r: (r["config"], r["sample_id"]) in EXCLUDED, axis=1)]
print(f"Generating {len(tc_18)} S1MS v2 config JSONs (excluded Phase 1: {EXCLUDED})")

generated = 0
for _, tc in tc_18.iterrows():
    cfg = tc["config"]
    sid = int(tc["sample_id"])
    tag = f"tc_v2_{cfg}_{sid}_s1ms"

    obs_csv = os.path.join(DATA_V2, f"obs_multiseed_{cfg}_{sid}.csv")
    if not os.path.exists(obs_csv):
        print(f"  WARNING: v2 obs CSV missing for {cfg} #{sid} — skipping")
        continue

    config = {
        "tube_setup":            "1tube",
        "calibration_seeds":     CALIBRATION_SEEDS,
        "maxiter":               100,
        "sigma0":                0.1,
        "grid_n_axial":          [1],
        "grid_max_order1":       [20],
        "calibration_objective": "replicate_seed_average",
        "calibration_params":    ALL_PARAMS,
        "objective_config": {
            "w_angle_mean":        0.3,
            "angle_normalize_deg": 45.0,
        },
        "env_json":      ENV_JSON_V2[cfg],
        "obs_csv_1tube": obs_csv,
        "warm_start_json": TEMPLATE,
        "experiment_tag": tag,
    }

    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  {tag}.json")
    generated += 1

print(f"\nDone — {generated} S1MS v2 configs in {OUT_DIR}")
