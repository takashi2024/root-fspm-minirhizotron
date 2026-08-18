"""
generate_test_case_configs_s2ms.py

Generates CMA-ES calibration config JSONs for Scenario 2 Multi-Seed (S2MS):
  - Observations: K=7 multi-seed mean (obs_multiseed_<config>_<sample_id>.csv)
  - Calibrated params: 4 arch params (elongation fixed at true values)
  - sigma0: 0.10 (cold-start, same as S2 — fair comparison)
  - maxiter: 150
  - tag: tc_<config>_<sample_id>_s2ms

Excludes A_497 and C_100 (prohibitive runtime).
Warm-start JSONs are reused from S2 (true elongation values already written).

Output: calibration/configs/test_cases/tc_*_s2ms.json  (18 files)
"""

import json
import os
import pandas as pd

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)
OUT_DIR     = os.path.join(CALIB_DIR, "configs", "test_cases")
DATA_DIR    = os.path.join(CALIB_DIR, "data", "test_cases")
SBI_CONFIGS = os.path.join(PROJECT_DIR, "sbi", "configs")

os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDED = {("A", 497), ("C", 100)}

CALIBRATION_SEEDS = [11, 22, 99, 88, 33, 55, 77, 7, 13, 42, 66, 91]
ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

ENV_JSON = {
    "A":  os.path.join(SBI_CONFIGS, "Parameter_SBI_env_A.json"),
    "B2": os.path.join(SBI_CONFIGS, "Parameter_SBI_env_B2.json"),
    "C":  os.path.join(SBI_CONFIGS, "Parameter_SBI_env_C.json"),
    "D2": os.path.join(SBI_CONFIGS, "Parameter_SBI_env_D2.json"),
}

tc_df = pd.read_csv(os.path.join(CALIB_DIR, "test_cases_selected.csv"))
tc_18 = tc_df[~tc_df.apply(lambda r: (r["config"], r["sample_id"]) in EXCLUDED, axis=1)]
print(f"Generating {len(tc_18)} config JSONs (excluded: {EXCLUDED})")

for _, tc in tc_18.iterrows():
    cfg       = tc["config"]
    sid       = int(tc["sample_id"])
    tag       = f"tc_{cfg}_{sid}_s2ms"

    obs_csv      = os.path.join(DATA_DIR, f"obs_multiseed_{cfg}_{sid}.csv")
    warmstart    = os.path.join(DATA_DIR, f"warmstart_{cfg}_{sid}.json")

    if not os.path.exists(obs_csv):
        print(f"  WARNING: obs CSV missing for {cfg} #{sid} — skipping")
        continue
    if not os.path.exists(warmstart):
        print(f"  WARNING: warmstart JSON missing for {cfg} #{sid} — skipping")
        continue

    config = {
        "tube_setup":             "1tube",
        "calibration_seeds":      CALIBRATION_SEEDS,
        "maxiter":                150,
        "sigma0":                 0.1,
        "grid_n_axial":           [1],
        "grid_max_order1":        [20],
        "calibration_objective":  "replicate_seed_average",
        "calibration_params":     ARCH_PARAMS,
        "objective_config": {
            "w_angle_mean":       0.3,
            "angle_normalize_deg": 45.0,
        },
        "env_json":               ENV_JSON[cfg],
        "obs_csv_1tube":          obs_csv,
        "warm_start_json":        warmstart,
        "experiment_tag":         tag,
    }

    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  {tag}.json")

print(f"\nDone — configs in {OUT_DIR}")
