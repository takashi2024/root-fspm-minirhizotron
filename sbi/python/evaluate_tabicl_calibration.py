"""
evaluate_tabicl_calibration.py

Quantile Calibration Plot (QCP) assessment for TabICL predictions on root
architecture parameters. Mirrors the WeedMap calibration analysis.

For each scenario (Scen2: elong known, Scen1: all unknown), performs k-fold
cross-validation on the training set (1980 samples, test cases excluded) and
computes:
  - Per-quantile actual coverage vs. nominal quantile (reliability diagram)
  - MACE (Mean Absolute Calibration Error) per parameter
  - 80% interval width distribution per parameter

Output:
  sbi/data/combined/calibration/
    tabicl_calibration_scen2.csv   — per-quantile coverage table
    tabicl_calibration_scen1.csv
    tabicl_calibration_scen2.png   — reliability diagram (all params)
    tabicl_calibration_scen1.png

Usage (from repo root):
    python sbi/python/evaluate_tabicl_calibration.py
"""

import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler

REPO    = Path(__file__).parent.parent.parent
OUT     = REPO / "sbi/data/combined/calibration"
OUT.mkdir(parents=True, exist_ok=True)

ELONGATION_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
]
CONFIG_ORDER = ["A", "B2", "C", "D2"]
MIN_WINDOWS  = 10
N_ESTIMATORS = 4
N_FOLDS      = 10

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all():
    td      = REPO / "sbi/data/combined/tensors"
    X_raw   = np.load(td / "X_summary.npz")["X"]
    y_raw   = np.load(td / "y.npz")["y"]
    cfg_ids = np.load(td / "config_ids.npy")
    layout  = np.load(td / "layout_features.npy")
    with open(td / "param_names.txt") as f:
        param_names = [l.strip() for l in f if l.strip()]
    N      = len(X_raw)
    X_flat = X_raw.reshape(N, -1).astype(np.float64)
    onehot = np.eye(int(cfg_ids.max()) + 1, dtype=np.float64)[cfg_ids]
    X_base = np.hstack([X_flat, layout.astype(np.float64), onehot])
    return X_base, y_raw.astype(np.float64), param_names, X_raw


# ── Calibration metric ─────────────────────────────────────────────────────────

def qcp_row(y_true: np.ndarray, q_mat: np.ndarray, param_name: str) -> dict:
    """Compute per-quantile actual coverage and MACE for one parameter."""
    row = {"parameter": param_name}
    abs_errors = []
    for j, q in enumerate(QUANTILES):
        actual_cov = float(np.mean(y_true <= q_mat[:, j]))
        row[f"cov_q{q:.1f}"] = round(actual_cov, 4)
        abs_errors.append(abs(actual_cov - q))
    row["MACE"] = round(float(np.mean(abs_errors)), 4)
    # 80% interval width (Q10 to Q90) in original scale
    idx_q10 = QUANTILES.index(0.1)
    idx_q90 = QUANTILES.index(0.9)
    widths = q_mat[:, idx_q90] - q_mat[:, idx_q10]
    row["interval_width_mean"] = round(float(widths.mean()), 4)
    row["interval_width_std"]  = round(float(widths.std()),  4)
    return row


# ── Reliability diagram ────────────────────────────────────────────────────────

def plot_reliability(cov_df: pd.DataFrame, scenario: str, out_path: Path):
    params = cov_df["parameter"].tolist()
    n      = len(params)
    ncol   = min(4, n)
    nrow   = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow),
                             squeeze=False)
    fig.suptitle(f"TabICL Quantile Calibration [{scenario}]  (10-fold CV, N≈1980)",
                 fontsize=12, fontweight="bold")

    cov_cols = [f"cov_q{q:.1f}" for q in QUANTILES]

    for idx, row in cov_df.iterrows():
        ax = axes[idx // ncol][idx % ncol]
        actual = [row[c] for c in cov_cols]
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Perfect")
        ax.plot(QUANTILES, actual, "o-", color="#1B5E9E", lw=1.8, ms=5,
                label=f"TabICL  MACE={row['MACE']:.3f}")
        ax.fill_between(QUANTILES, QUANTILES, actual,
                        alpha=0.15, color="#1B5E9E")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal quantile", fontsize=8)
        ax.set_ylabel("Actual coverage", fontsize=8)
        ax.set_title(f"{row['parameter']}\nMAC={row['MACE']:.3f}  "
                     f"80%width={row['interval_width_mean']:.3g}±{row['interval_width_std']:.3g}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Cross-validated calibration assessment ─────────────────────────────────────

def run_calibration(X_train, y_train, X_raw_train,
                    target_params, known_params, param_names, scenario_label):
    from tabicl import TabICLRegressor

    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    # Append known params as extra covariates
    X = np.hstack([X_train, y_train[:, k_idx]]) if known_params else X_train

    # Informative sample filter (apply globally, then CV within filtered set)
    presence  = X_raw_train[:, :, :, :, 1]
    win_count = (presence > 0).reshape(len(X_raw_train), -1).sum(1)
    keep      = win_count >= MIN_WINDOWS
    X_f = X[keep]; y_f = y_train[keep]
    print(f"  Informative filter: {keep.sum()}/{len(X)} samples", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Accumulators: per parameter, collect (true, q_mat) across folds
    all_true   = {p: [] for p in target_params}
    all_q_mats = {p: [] for p in target_params}

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_f)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(val_idx)})",
              flush=True)
        Xtr, Xval = X_f[tr_idx], X_f[val_idx]
        ytr, yval = y_f[tr_idx], y_f[val_idx]

        mu, sig = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s  = (Xtr  - mu) / sig
        Xval_s = (Xval - mu) / sig

        for i, pname in enumerate(target_params):
            param_scaler = ParamScaler([pname])   # single-param scaler for this output
            ytr_p = param_scaler.normalize(ytr[:, t_idx[i:i+1]])[:, 0]

            m = TabICLRegressor(n_estimators=N_ESTIMATORS,
                                random_state=42, device="cpu")
            m.fit(Xtr_s, ytr_p)

            result = m.predict(Xval_s,
                               output_type=["mean", "quantiles"],
                               alphas=QUANTILES)
            q_norm = np.clip(result["quantiles"], 0.0, 1.0)   # (n_val, n_quantiles)

            # Denormalize each quantile column independently
            q_denorm = np.column_stack([
                param_scaler.denormalize(q_norm[:, j:j+1])[:, 0]
                for j in range(len(QUANTILES))
            ])

            all_true[pname].append(yval[:, t_idx[i]])
            all_q_mats[pname].append(q_denorm)

    # Concatenate across folds and compute calibration
    cov_rows = []
    for pname in target_params:
        y_concat = np.concatenate(all_true[pname])
        q_concat = np.vstack(all_q_mats[pname])
        cov_rows.append(qcp_row(y_concat, q_concat, pname))

    return pd.DataFrame(cov_rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data…", flush=True)
    X_base, y_all, param_names, X_raw_all = load_all()
    test_df = pd.read_csv(REPO / "calibration/test_cases_selected.csv")

    # Exclude the 20 test cases from training data
    test_idx = np.array([CONFIG_ORDER.index(r["config"]) * 500 + int(r["sample_id"])
                         for _, r in test_df.iterrows()])
    mask     = np.ones(len(X_base), dtype=bool)
    mask[test_idx] = False
    X_tr     = X_base[mask]
    y_tr     = y_all[mask]
    X_tr_raw = X_raw_all[mask]
    print(f"Training set: {X_tr.shape[0]} samples\n", flush=True)

    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]

    for scenario_label, target_params, known_params in [
        ("Scen2: elong known", arch_params,  ELONGATION_PARAMS),
        ("Scen1: all unknown", param_names,  []),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"TabICL Calibration — {scenario_label}", flush=True)
        print(f"{'='*60}", flush=True)

        cov_df = run_calibration(X_tr, y_tr, X_tr_raw,
                                 target_params, known_params,
                                 param_names, scenario_label)

        tag = "scen2" if "2" in scenario_label else "scen1"
        csv_path = OUT / f"tabicl_calibration_{tag}.csv"
        cov_df.to_csv(csv_path, index=False)
        print(f"\n  Wrote: {csv_path.name}")

        print("\n  MACE per parameter:")
        for _, row in cov_df.iterrows():
            print(f"    {row['parameter']:<30} MACE={row['MACE']:.4f}  "
                  f"80%width={row['interval_width_mean']:.4f}")

        plot_reliability(cov_df, scenario_label,
                         OUT / f"tabicl_calibration_{tag}.png")

    print("\nDone. Outputs in:", OUT)


if __name__ == "__main__":
    main()
