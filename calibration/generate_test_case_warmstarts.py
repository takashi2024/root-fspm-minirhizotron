"""
generate_test_case_warmstarts.py

Creates per-case species JSON files for Scenario 2: elongation parameters are set to
their true values (as if obtained from destructive sampling at end of growing season).
All other parameters are inherited from the template species JSON.

Output:
  calibration/data/test_cases/warmstart_<config>_<sample_id>.json  (20 files)
"""

import csv
import json
import os

CALIB_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(CALIB_DIR)
TEST_CASES_CSV = os.path.join(CALIB_DIR, "test_cases_selected.csv")
TEMPLATE_JSON  = os.path.join(CALIB_DIR, "Parameter_VD_template.json")
SBI_DATA       = os.path.join(PROJECT_DIR, "sbi", "data")
OUT_DIR        = os.path.join(CALIB_DIR, "data", "test_cases")

ELONGATION_PARAMS = [
    "elongation_rate_min",
    "elongation_rate_max",
    "elongation_logistic_k",
    "elongation_logistic_t50",
]

os.makedirs(OUT_DIR, exist_ok=True)

with open(TEMPLATE_JSON) as f:
    template = json.load(f)

with open(TEST_CASES_CSV, newline="") as f:
    test_cases = list(csv.DictReader(f))

for tc in test_cases:
    cfg       = tc["config"]
    sample_id = int(tc["sample_id"])

    ws = dict(template)
    for p in ELONGATION_PARAMS:
        ws[p] = float(tc[p])

    out_path = os.path.join(OUT_DIR, f"warmstart_{cfg}_{sample_id}.json")
    with open(out_path, "w") as f:
        json.dump(ws, f, indent=2)

    vals = {p: ws[p] for p in ELONGATION_PARAMS}
    print(f"  [{cfg} #{sample_id:>4}] elongation: "
          f"min={vals['elongation_rate_min']:.4f}  "
          f"max={vals['elongation_rate_max']:.4f}  "
          f"k={vals['elongation_logistic_k']:.4f}  "
          f"t50={vals['elongation_logistic_t50']:.2f}  → {out_path}")

print(f"\nDone — {len(test_cases)} warmstart JSONs written to {OUT_DIR}")
