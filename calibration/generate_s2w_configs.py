"""
generate_s2w_configs.py

Scenario 2 warm-start (S2W): CMA-ES calibration of the 4 architectural params
with TabICL S2 predictions as the initial guess (warm-start point).

For each of the 20 test cases:
  1. Load existing warmstart_<config>_<sample_id>.json (true elongation params)
  2. Overwrite the 4 arch params with TabICL Scen2 predictions
  3. Save as data/test_cases/warmstart_tabicl_<config>_<sample_id>.json
  4. Write calibration config JSON → configs/test_cases/tc_<config>_<sample_id>_s2w.json

Key differences from S2 cold-start:
  - warm_start_json → TabICL-based warmstart (arch params set to TabICL predictions)
  - sigma0 = 0.05  (tighter initial search, trusting TabICL is already in the right ballpark)
  - experiment_tag → tc_<config>_<sample_id>_s2w
"""

import os, json
import pandas as pd

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(CALIB_DIR)

TC_CSV    = os.path.join(CALIB_DIR, "test_cases_selected.csv")
TAB_CSV   = os.path.join(REPO_ROOT, "sbi", "data", "combined",
                          "test_case_eval", "results_tabicl.csv")
WS_DIR    = os.path.join(CALIB_DIR, "data", "test_cases")
CFG_DIR   = os.path.join(CALIB_DIR, "configs", "test_cases")

tc_df  = pd.read_csv(TC_CSV)
tab_df = pd.read_csv(TAB_CSV)
tab_s2 = tab_df[tab_df["scenario"] == "Scen2: elong known"]

ARCH_PARAMS = ["branch_angle_order1", "g_axial",
               "branching_prob_order1", "branching_prob_order2"]

os.makedirs(WS_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

print(f"Generating S2W configs for {len(tc_df)} test cases\n")

for _, row in tc_df.iterrows():
    cfg = row["config"]
    sid = int(row["sample_id"])
    tag = f"tc_{cfg}_{sid}_s2w"

    # ── 1. Load TabICL S2 predictions for this case ───────────────────────────
    pred_rows = tab_s2[(tab_s2["config"] == cfg) & (tab_s2["sample_id"] == sid)]
    if len(pred_rows) == 0:
        print(f"  SKIP {tag} — no TabICL S2 prediction found")
        continue

    tabicl_arch = {}
    for p in ARCH_PARAMS:
        pr = pred_rows[pred_rows["parameter"] == p]
        if len(pr) == 0:
            print(f"  SKIP {tag} — missing TabICL prediction for {p}")
            break
        tabicl_arch[p] = float(pr.iloc[0]["pred_value"])
    else:
        pass  # all params found

    if len(tabicl_arch) < 4:
        continue

    # ── 2. Build TabICL warm-start JSON ──────────────────────────────────────
    base_ws_path = os.path.join(WS_DIR, f"warmstart_{cfg}_{sid}.json")
    if not os.path.isfile(base_ws_path):
        print(f"  SKIP {tag} — base warmstart not found: {base_ws_path}")
        continue

    with open(base_ws_path) as f:
        ws = json.load(f)

    true_vals = {p: tabicl_arch[p] for p in ARCH_PARAMS}
    for p in ARCH_PARAMS:
        ws[p] = tabicl_arch[p]

    tabicl_ws_path = os.path.join(WS_DIR, f"warmstart_tabicl_{cfg}_{sid}.json")
    with open(tabicl_ws_path, "w") as f:
        json.dump(ws, f, indent=2)

    # ── 3. Build calibration config JSON ─────────────────────────────────────
    s2_cfg_path = os.path.join(CFG_DIR, f"tc_{cfg}_{sid}_s2.json")
    with open(s2_cfg_path) as f:
        s2_cfg = json.load(f)

    s2w_cfg = dict(s2_cfg)
    s2w_cfg["warm_start_json"] = tabicl_ws_path
    s2w_cfg["sigma0"]          = 0.05
    s2w_cfg["experiment_tag"]  = tag

    out_cfg_path = os.path.join(CFG_DIR, f"{tag}.json")
    with open(out_cfg_path, "w") as f:
        json.dump(s2w_cfg, f, indent=2)

    print(f"  {tag}")
    for p in ARCH_PARAMS:
        true_v = float(row[p])
        pred_v = tabicl_arch[p]
        print(f"    {p:<30} true={true_v:8.4f}  tabicl={pred_v:8.4f}")

print(f"\nDone. Warmstart JSONs → {WS_DIR}")
print(f"      Config JSONs    → {CFG_DIR}")
