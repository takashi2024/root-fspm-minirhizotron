"""
extract_test_case_obs.py

Extracts minirhizotron observations for the 20 selected test cases directly from the
pre-computed SBI X_summary.npz tensors. No simulation re-run is needed.

X_summary.npz shape: (500, 11, 15, 4, 3)
  dim 0: 500 samples (0-indexed; sample_id N → index N-1)
  dim 1: 11 snapshot days: [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
  dim 2: 15 stations (index 0 → station_idx=1, ..., index 14 → station_idx=15)
  dim 3: 4 azimuths (0→0°, 1→90°, 2→180°, 3→270°)
  dim 4: 3 channels [root_length_cm, has_root_binary, crossing_angle/90.0]

Date mapping: SOWING_DATE = 2025-04-01
  obs_date = SOWING_DATE + (sim_day - 1) days
  (matches calibration script: _date_to_simday = days_since_sowing + 1)

Output:
  calibration/data/test_cases/obs_<config>_<sample_id>.csv
  Columns: date, station_idx, along_tube_cm, depth_cm, azimuth_deg,
           root_length_cm, mean_crossing_angle_deg
"""

import csv
import math
import os
import numpy as np
from datetime import date, timedelta

# ── Constants ──────────────────────────────────────────────────────────────────

SNAPSHOT_DAYS  = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
AZIMUTHS_DEG   = [0.0, 90.0, 180.0, 270.0]
SOWING_DATE    = date(2025, 4, 1)
TUBE_ANGLE_DEG = 45.0          # tube inclination from horizontal
SIN45          = math.sin(math.radians(TUBE_ANGLE_DEG))

# Station positions along tube (cm) — from env JSON station_positions_cm
STATION_ALONG_CM = [2.5 + i * 5.0 for i in range(15)]   # 2.5, 7.5, ..., 72.5

CALIB_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CALIB_DIR)
TEST_CASES_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
OUT_DIR     = os.path.join(CALIB_DIR, "data", "test_cases")
SBI_DATA    = os.path.join(PROJECT_DIR, "sbi", "data")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load test cases ────────────────────────────────────────────────────────────

with open(TEST_CASES_CSV, newline="") as f:
    test_cases = list(csv.DictReader(f))

print(f"Loaded {len(test_cases)} test cases from {TEST_CASES_CSV}")

# ── Load X_summary tensors (one per config) ───────────────────────────────────

tensors = {}
for cfg in ["A", "B2", "C", "D2"]:
    npz_path = os.path.join(SBI_DATA, f"config_{cfg}", "tensors", "X_summary.npz")
    d = np.load(npz_path)
    tensors[cfg] = d[d.files[0]]   # (500, 11, 15, 4, 3)
    print(f"  Config {cfg}: shape {tensors[cfg].shape}")

# ── Extract per test case ──────────────────────────────────────────────────────

for tc in test_cases:
    cfg       = tc["config"]
    sample_id = int(tc["sample_id"])
    idx       = sample_id - 1      # 0-indexed

    data = tensors[cfg][idx]       # (11, 15, 4, 3)

    rows = []
    for t_idx, sim_day in enumerate(SNAPSHOT_DAYS):
        obs_date = (SOWING_DATE + timedelta(days=sim_day - 1)).strftime("%Y-%m-%d")
        for s_idx, along_tube_cm in enumerate(STATION_ALONG_CM):
            station_idx = s_idx + 1
            depth_cm    = along_tube_cm * SIN45
            for a_idx, azimuth_deg in enumerate(AZIMUTHS_DEG):
                length_cm  = float(data[t_idx, s_idx, a_idx, 0])
                has_root   = float(data[t_idx, s_idx, a_idx, 1]) > 0.5
                angle_raw  = float(data[t_idx, s_idx, a_idx, 2])
                # channel 2 is 0.0 when no roots (SimRunner stores NaN as 0)
                mean_angle = (angle_raw * 90.0) if has_root else float("nan")
                rows.append([
                    obs_date,
                    station_idx,
                    round(along_tube_cm, 4),
                    round(depth_cm, 4),
                    azimuth_deg,
                    round(length_cm, 6),
                    "" if math.isnan(mean_angle) else round(mean_angle, 4),
                ])

    out_path = os.path.join(OUT_DIR, f"obs_{cfg}_{sample_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "station_idx", "along_tube_cm", "depth_cm",
                    "azimuth_deg", "root_length_cm", "mean_crossing_angle_deg"])
        w.writerows(rows)

    # Quick stats
    n_total    = len(rows)
    n_nonzero  = sum(1 for r in rows if r[5] > 0)
    n_w_angle  = sum(1 for r in rows if r[6] != "")
    print(f"  [{cfg} #{sample_id:>4}] {n_total} rows, {n_nonzero} non-zero length "
          f"({100*n_nonzero/n_total:.1f}%), {n_w_angle} with angle → {out_path}")

print(f"\nDone — {len(test_cases)} obs CSVs written to {OUT_DIR}")
