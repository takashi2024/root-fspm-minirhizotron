"""
extract_test_case_obs_multiseed_v2.py

Extracts K=7 multi-seed mean minirhizotron observations for the 18 selected
test cases (excluding A_497 and C_100) from v2 X_mean.npz tensors.

Source: sbi/data/multi_seed_v2/config_<X>/tensors/X_mean.npz
Output: calibration/data_v2/test_cases/obs_multiseed_<config>_<sample_id>.csv
"""

import csv
import math
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

SNAPSHOT_DAYS    = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
AZIMUTHS_DEG     = [0.0, 90.0, 180.0, 270.0]
SOWING_DATE      = date(2025, 4, 1)
TUBE_ANGLE_DEG   = 45.0
SIN45            = math.sin(math.radians(TUBE_ANGLE_DEG))
STATION_ALONG_CM = [2.5 + i * 5.0 for i in range(15)]

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)
OUT_DIR     = os.path.join(CALIB_DIR, "data_v2", "test_cases")
MS_DIR      = os.path.join(PROJECT_DIR, "sbi", "data", "multi_seed_v2")

os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDED = {("A", 497), ("C", 100)}

tc_df = pd.read_csv(os.path.join(CALIB_DIR, "test_cases_selected_v2.csv"))
tc_18 = tc_df[~tc_df.apply(lambda r: (r["config"], r["sample_id"]) in EXCLUDED, axis=1)]
print(f"Test cases: {len(tc_df)} total → {len(tc_18)} after excluding {EXCLUDED}")

x_mean       = {}
sample_index = {}

for cfg in ["A", "B2", "C", "D2"]:
    td   = os.path.join(MS_DIR, f"config_{cfg}", "tensors")
    df_s = pd.read_csv(os.path.join(MS_DIR, f"config_{cfg}", "samples_filtered.csv"))
    x_mean[cfg]       = np.load(os.path.join(td, "X_mean.npz"))["X"]
    sample_index[cfg] = {int(r.sample_id): i for i, r in df_s.iterrows()}
    print(f"  Config {cfg}: X_mean shape={x_mean[cfg].shape}  N_filtered={len(df_s)}")

for _, tc in tc_18.iterrows():
    cfg       = tc["config"]
    sample_id = int(tc["sample_id"])

    if sample_id not in sample_index[cfg]:
        print(f"  WARNING: {cfg} #{sample_id} not in v2 multi-seed filtered samples — skipping")
        continue

    row_idx = sample_index[cfg][sample_id]
    data    = x_mean[cfg][row_idx]   # (11, 15, 4, 3)

    rows = []
    for t_idx, sim_day in enumerate(SNAPSHOT_DAYS):
        obs_date = (SOWING_DATE + timedelta(days=sim_day - 1)).strftime("%Y-%m-%d")
        for s_idx, along_tube_cm in enumerate(STATION_ALONG_CM):
            station_idx = s_idx + 1
            depth_cm    = along_tube_cm * SIN45
            for a_idx, azimuth_deg in enumerate(AZIMUTHS_DEG):
                length_cm      = float(data[t_idx, s_idx, a_idx, 0])
                has_root_frac  = float(data[t_idx, s_idx, a_idx, 1])
                angle_sum_norm = float(data[t_idx, s_idx, a_idx, 2])

                has_root = has_root_frac > 0.0
                if has_root:
                    mean_angle = (angle_sum_norm / has_root_frac) * 90.0
                else:
                    mean_angle = float("nan")

                rows.append([
                    obs_date,
                    station_idx,
                    round(along_tube_cm, 4),
                    round(depth_cm, 4),
                    azimuth_deg,
                    round(length_cm, 6),
                    "" if math.isnan(mean_angle) else round(mean_angle, 4),
                ])

    out_path = os.path.join(OUT_DIR, f"obs_multiseed_{cfg}_{sample_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "station_idx", "along_tube_cm", "depth_cm",
                    "azimuth_deg", "root_length_cm", "mean_crossing_angle_deg"])
        w.writerows(rows)

    n_nonzero = sum(1 for r in rows if r[5] > 0)
    print(f"  [{cfg} #{sample_id:>4}] {len(rows)} rows  "
          f"{n_nonzero} non-zero ({100*n_nonzero/len(rows):.1f}%)  "
          f"→ {os.path.basename(out_path)}")

print(f"\nDone — {len(tc_18)} obs CSVs written to {OUT_DIR}")
