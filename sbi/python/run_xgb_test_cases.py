"""run_xgb_test_cases.py — XGBoost predictions for 20 held-out test cases.

Runs Scenario 2 (elongation params known) and Scenario 1 (all unknown) for XGBoost only.
Saves a predictions CSV immediately without waiting for TabICL.

Output:
  sbi/data/combined/test_case_eval/xgb_predictions.csv
  columns: config, sample_id, scenario, <param_name>_pred, <param_name>_true  (wide)

Usage (from repo root):
    python sbi/python/run_xgb_test_cases.py
"""

import sys
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler

REPO   = Path(__file__).parent.parent.parent
TENSORS_DIR = REPO / "sbi/data/combined/tensors"
TEST_CSV    = REPO / "calibration/test_cases_selected.csv"
OUT_DIR     = REPO / "sbi/data/combined/test_case_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = ["A", "B2", "C", "D2"]
ELONGATION_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
]


def make_xgb():
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        ), n_jobs=1)


def load_tensors():
    X_raw   = np.load(TENSORS_DIR / "X_summary.npz")["X"]
    y_raw   = np.load(TENSORS_DIR / "y.npz")["y"]
    cfg_ids = np.load(TENSORS_DIR / "config_ids.npy")
    layout  = np.load(TENSORS_DIR / "layout_features.npy")
    with open(TENSORS_DIR / "param_names.txt") as f:
        param_names = [l.strip() for l in f if l.strip()]
    N      = len(X_raw)
    X_flat = X_raw.reshape(N, -1).astype(np.float64)
    onehot = np.eye(int(cfg_ids.max()) + 1, dtype=np.float64)[cfg_ids]
    X_base = np.hstack([X_flat, layout.astype(np.float64), onehot])
    return X_base, y_raw.astype(np.float64), param_names


def run_xgb_scenario(X_tr, X_te, y_tr, y_te, param_names,
                     target_params, known_params, scenario_label):
    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    X_tr_s = np.hstack([X_tr, y_tr[:, k_idx]]) if known_params else X_tr
    X_te_s = np.hstack([X_te, y_te[:, k_idx]]) if known_params else X_te

    mu, sig   = X_tr_s.mean(0), X_tr_s.std(0) + 1e-8
    X_tr_sn   = (X_tr_s - mu) / sig
    X_te_sn   = (X_te_s - mu) / sig

    scaler   = ParamScaler(target_params)
    y_tr_n   = scaler.normalize(y_tr[:, t_idx])

    print(f"  [{scenario_label}] Fitting XGBoost ({X_tr_sn.shape[1]} features, {X_tr_sn.shape[0]} train)…", flush=True)
    model = make_xgb()
    model.fit(X_tr_sn, y_tr_n)
    y_pred_n = np.clip(model.predict(X_te_sn), 0.0, 1.0)
    y_pred   = scaler.denormalize(y_pred_n)
    y_true   = y_te[:, t_idx]

    for i, p in enumerate(target_params):
        mae    = np.abs(y_true[:, i] - y_pred[:, i]).mean()
        pct_err = np.abs((y_true[:, i] - y_pred[:, i]) / (np.abs(y_true[:, i]) + 1e-8)).mean() * 100
        print(f"    {p:30s}  MAE={mae:.4f}  mean|%err|={pct_err:.1f}%")

    return y_true, y_pred, target_params


def main():
    print("Loading tensors…")
    X_base, y_all, param_names = load_tensors()
    print(f"  Shape: X={X_base.shape}, y={y_all.shape}")

    test_df = pd.read_csv(TEST_CSV)
    print(f"  Test cases: {len(test_df)}")

    test_idx = np.array([
        CONFIG_ORDER.index(r["config"]) * 500 + int(r["sample_id"])
        for _, r in test_df.iterrows()
    ])
    mask = np.ones(len(X_base), dtype=bool)
    mask[test_idx] = False
    X_tr = X_base[mask];    y_tr = y_all[mask]
    X_te = X_base[test_idx]; y_te = y_all[test_idx]
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]
    rows = []

    for scenario_label, target_params, known_params in [
        ("Scen1_all_unknown", param_names, []),
        ("Scen2_elong_known", arch_params, ELONGATION_PARAMS),
    ]:
        print(f"\n── {scenario_label} ──")
        y_true, y_pred, tparams = run_xgb_scenario(
            X_tr, X_te, y_tr, y_te, param_names,
            target_params, known_params, scenario_label)

        for j, (_, row) in enumerate(test_df.iterrows()):
            rec = {
                "model": "XGBoost",
                "scenario": scenario_label,
                "config": row["config"],
                "sample_id": int(row["sample_id"]),
                "g_axial_quintile": row["g_axial_quintile"],
                "windows_with_roots": int(row["windows_with_roots"]),
            }
            for i, p in enumerate(tparams):
                rec[f"{p}_true"] = float(y_true[j, i])
                rec[f"{p}_pred"] = float(y_pred[j, i])
            rows.append(rec)

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "xgb_predictions.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} rows → {out_path}")

    # Convenience: also save a flat format for structural evaluation
    # One row per test case for Scenario 2 only, with predicted arch params
    s2 = df_out[df_out["scenario"] == "Scen2_elong_known"].copy()
    print("\n── Scenario 2 summary (mean |%error| per param) ──")
    for p in arch_params:
        if f"{p}_true" in s2.columns:
            pct = np.abs((s2[f"{p}_true"] - s2[f"{p}_pred"]) / (s2[f"{p}_true"].abs() + 1e-8)).mean() * 100
            print(f"  {p:30s}  {pct:.1f}%")
    overall = np.mean([
        np.abs((s2[f"{p}_true"] - s2[f"{p}_pred"]) / (s2[f"{p}_true"].abs() + 1e-8)).mean() * 100
        for p in arch_params if f"{p}_true" in s2.columns
    ])
    print(f"\n  Overall mean |%error|: {overall:.1f}%")


if __name__ == "__main__":
    main()
