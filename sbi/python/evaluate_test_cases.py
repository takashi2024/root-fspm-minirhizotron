"""evaluate_test_cases.py — Evaluate XGBoost + TabICL on 20 held-out test cases.

Two scenarios:
  Scenario 1: all 8 parameters predicted from minirhizotron obs only
  Scenario 2: elongation params provided as known covariates (destructive sampling);
              only 4 architectural params predicted

Usage (from repo root):
    python sbi/python/evaluate_test_cases.py \
        --test_csv calibration/test_cases_selected.csv \
        --tensors_dir sbi/data/combined/tensors \
        --out_dir sbi/data/combined/test_case_eval
"""

import argparse
import sys
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler

ELONGATION_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
]
CONFIG_ORDER = ["A", "B2", "C", "D2"]
QUINTILE_COLORS = {
    "Q1": "#1a9850", "Q2": "#91cf60", "Q3": "#fee08b",
    "Q4": "#fc8d59", "Q5": "#d73027",
}
QUINTILE_LABELS = {
    "Q1": "Q1 (low g)", "Q2": "Q2", "Q3": "Q3",
    "Q4": "Q4", "Q5": "Q5 (high g)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv",    default="calibration/test_cases_selected.csv")
    p.add_argument("--tensors_dir", default="sbi/data/combined/tensors")
    p.add_argument("--out_dir",     default="sbi/data/combined/test_case_eval")
    return p.parse_args()


def make_xgb():
    from xgboost import XGBRegressor
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        ), n_jobs=1)


MIN_WINDOWS = 10   # only train on informative samples (same threshold as test case selection)

def make_tabicl():
    from tabicl import TabICLRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    # MPS hangs in TabICL 2.1.1 under MultiOutputRegressor; CPU is stable (matches WeedMap usage)
    device = "cpu"
    print(f"  TabICL device: {device}", flush=True)
    return MultiOutputRegressor(
        Pipeline([
            ("pca",   PCA(n_components=64)),
            ("model", TabICLRegressor(random_state=42, device=device)),
        ]), n_jobs=1)


def load_all(tensors_dir):
    X_raw   = np.load(tensors_dir / "X_summary.npz")["X"]   # (2000,11,15,4,3) — kept for window filter
    y_raw   = np.load(tensors_dir / "y.npz")["y"]
    cfg_ids = np.load(tensors_dir / "config_ids.npy")
    layout  = np.load(tensors_dir / "layout_features.npy")
    with open(tensors_dir / "param_names.txt") as f:
        param_names = [l.strip() for l in f if l.strip()]
    N      = len(X_raw)
    X_flat = X_raw.reshape(N, -1).astype(np.float64)
    onehot = np.eye(int(cfg_ids.max()) + 1, dtype=np.float64)[cfg_ids]
    X_base = np.hstack([X_flat, layout.astype(np.float64), onehot])
    return X_base, y_raw.astype(np.float64), param_names, X_raw


def plot_actual_vs_pred(y_true, y_pred, param_names, quintiles, title, out_path):
    n = len(param_names)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow))
    axf = np.array(axes).flatten()
    for i, name in enumerate(param_names):
        ax = axf[i]
        for q, color in QUINTILE_COLORS.items():
            sel = [j for j, qq in enumerate(quintiles) if qq == q]
            if sel:
                ax.scatter(y_true[sel, i], y_pred[sel, i], color=color, s=90,
                           edgecolors="white", linewidths=0.5,
                           label=QUINTILE_LABELS[q], zorder=3)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        pad = (hi - lo) * 0.08 + 1e-6
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlabel("True", fontsize=9); ax.set_ylabel("Predicted", fontsize=9)
        mae = np.abs(y_true[:, i] - y_pred[:, i]).mean()
        ax.set_title(f"{name}\nMAE={mae:.3g}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, title="g_axial", title_fontsize=7, markerscale=1.2)
    for ax in axf[n:]: ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_gaxial_vs_error(y_true, y_pred, param_names, g_axial,
                         quintiles, windows, title, out_path):
    n = len(param_names)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow))
    axf = np.array(axes).flatten()
    abs_err = np.abs(y_true - y_pred)
    for i, name in enumerate(param_names):
        ax = axf[i]
        sc = ax.scatter(g_axial, abs_err[:, i], c=windows, cmap="viridis",
                        s=90, edgecolors="white", linewidths=0.5,
                        vmin=0, vmax=max(windows), zorder=3)
        for q, color in QUINTILE_COLORS.items():
            sel = [j for j, qq in enumerate(quintiles) if qq == q]
            if sel:
                ax.scatter(g_axial[sel], abs_err[sel, i], facecolors="none",
                           edgecolors=color, s=110, linewidths=1.8, zorder=4,
                           label=QUINTILE_LABELS[q])
        ax.set_xscale("log")
        ax.set_xlabel("True g_axial (log)", fontsize=9)
        ax.set_ylabel("|Error|", fontsize=9)
        ax.set_title(f"{name}\nmean={abs_err[:,i].mean():.3g}", fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6.5, title="Quintile", title_fontsize=7, markerscale=1.0)
    for ax in axf[n:]: ax.set_visible(False)
    fig.colorbar(sc, ax=axf[min(n, ncol)-1], label="obs windows", shrink=0.8)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_comparison(results_by_model_scen, arch_params, quintiles, out_path):
    """4-panel: actual-vs-pred and quintile-error for both scenarios, both models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    scen_pairs = [
        ("Scen1: all unknown", "Scen2: elong known"),
        ("Scen1: all unknown", "Scen2: elong known"),
    ]
    model_rows = ["XGBoost", "TabICL"]
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    w = 0.35

    for row, model_name in enumerate(model_rows):
        # Left col: MAE per arch param, scen1 vs scen2
        ax = axes[row, 0]
        x = np.arange(len(arch_params))
        for k, (scen_label, color) in enumerate([
            ("Scen1: all unknown", "#2196F3"),
            ("Scen2: elong known", "#FF9800"),
        ]):
            res = results_by_model_scen[model_name][scen_label]
            idxs = [res["param_names"].index(p) for p in arch_params]
            mae  = np.abs(res["y_true"][:, idxs] - res["y_pred"][:, idxs]).mean(axis=0)
            ax.bar(x + k*w, mae, w, label=scen_label, color=color, alpha=0.85)
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(arch_params, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAE (raw units)"); ax.set_title(f"{model_name}: MAE per arch param")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

        # Right col: per-quintile normalized error
        ax2 = axes[row, 1]
        for k, (scen_label, color) in enumerate([
            ("Scen1: all unknown", "#2196F3"),
            ("Scen2: elong known", "#FF9800"),
        ]):
            res  = results_by_model_scen[model_name][scen_label]
            idxs = [res["param_names"].index(p) for p in arch_params]
            y_t  = res["y_true"][:, idxs]; y_p = res["y_pred"][:, idxs]
            rng  = y_t.max(0) - y_t.min(0) + 1e-8
            nerr = np.abs(y_t - y_p) / rng
            q_means = [nerr[[j for j,qq in enumerate(quintiles) if qq==q]].mean()
                       if any(qq==q for qq in quintiles) else 0 for q in q_labels]
            ax2.bar(np.arange(len(q_labels)) + k*w, q_means, w,
                    label=scen_label, color=color, alpha=0.85)
        ax2.set_xticks(np.arange(len(q_labels)) + w/2)
        ax2.set_xticklabels(["Q1\n(low g)", "Q2", "Q3", "Q4", "Q5\n(high g)"], fontsize=8)
        ax2.set_ylabel("Norm. |error|")
        ax2.set_title(f"{model_name}: recovery by g_axial quintile")
        ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ML evaluation on 20 test cases — XGBoost vs TabICL, Scenario 1 vs 2",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_scenario(X_tr_base, X_te_base, y_tr, y_te, param_names,
                 target_params, known_params, g_axial, quintiles,
                 windows, scen_label, model_name, model, out_dir,
                 X_tr_raw=None):
    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    X_tr = np.hstack([X_tr_base, y_tr[:, k_idx]]) if known_params else X_tr_base
    X_te = np.hstack([X_te_base, y_te[:, k_idx]]) if known_params else X_te_base

    mu, sig  = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_s   = (X_tr - mu) / sig
    X_te_s   = (X_te - mu) / sig
    X_tr_raw = X_tr_raw  # passed through for TabICL window filter

    scaler   = ParamScaler(target_params)
    y_tr_n   = scaler.normalize(y_tr[:, t_idx])

    # For TabICL: filter to informative training samples only
    if X_tr_raw is not None:
        presence  = X_tr_raw[:, :, :, :, 1]
        win_count = (presence > 0).reshape(len(X_tr_raw), -1).sum(1)
        info_mask = win_count >= MIN_WINDOWS
        X_tr_s_fit = X_tr_s[info_mask]; y_tr_n_fit = y_tr_n[info_mask]
        print(f"  Informative filter: {info_mask.sum()}/{len(X_tr_s)} samples (≥{MIN_WINDOWS} windows)", flush=True)
    else:
        X_tr_s_fit, y_tr_n_fit = X_tr_s, y_tr_n

    print(f"  Fitting ({X_tr_s_fit.shape[1]} features, N={len(X_tr_s_fit)})…", flush=True)
    model.fit(X_tr_s_fit, y_tr_n_fit)
    y_pred_n = np.clip(model.predict(X_te_s), 0.0, 1.0)
    y_pred   = scaler.denormalize(y_pred_n)
    y_true   = y_te[:, t_idx]

    print(f"  Mean MAE: {np.abs(y_true - y_pred).mean():.4f}")
    for i, p in enumerate(target_params):
        print(f"  {p:30s}  MAE={np.abs(y_true[:,i]-y_pred[:,i]).mean():.4f}")

    tag = f"{model_name.lower()}_{scen_label.replace(' ','_').replace(':','')}"
    plot_actual_vs_pred(
        y_true, y_pred, target_params, quintiles,
        f"{model_name} [{scen_label}] — True vs Predicted (N=20)",
        out_dir / f"actual_vs_pred_{tag}.png")
    plot_gaxial_vs_error(
        y_true, y_pred, target_params, g_axial, quintiles, windows,
        f"{model_name} [{scen_label}] — g_axial vs Recovery Error",
        out_dir / f"gaxial_vs_error_{tag}.png")

    return {"y_true": y_true, "y_pred": y_pred, "param_names": target_params}


def main():
    args    = parse_args()
    root    = Path(__file__).parent.parent.parent
    out_dir = root / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(root / args.test_csv)

    X_base, y_all, param_names, X_raw_all = load_all(root / args.tensors_dir)

    test_idx = np.array([CONFIG_ORDER.index(r["config"]) * 500 + int(r["sample_id"])
                         for _, r in test_df.iterrows()])
    mask = np.ones(len(X_base), dtype=bool); mask[test_idx] = False
    X_tr     = X_base[mask];     y_tr     = y_all[mask]
    X_te     = X_base[test_idx]; y_te     = y_all[test_idx]
    X_tr_raw = X_raw_all[mask]   # raw (N,11,15,4,3) for window count filter
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    g_axial   = test_df["g_axial"].values
    quintiles = test_df["g_axial_quintile"].tolist()
    windows   = test_df["windows_with_roots"].values
    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]

    models = [("XGBoost", make_xgb()), ("TabICL", make_tabicl())]
    results_all = {m: {} for m, _ in models}
    rows = []

    for model_name, model_obj in models:
        print(f"\n══ {model_name} ══")
        for scen_label, target_params, known_params in [
            ("Scen1: all unknown", param_names,   []),
            ("Scen2: elong known", arch_params,   ELONGATION_PARAMS),
        ]:
            print(f"\n  ── {scen_label} ──")
            from sklearn.base import clone
            from sklearn.base import clone
            model_inst = clone(model_obj)
            res = run_scenario(
                X_tr, X_te, y_tr, y_te, param_names,
                target_params, known_params,
                g_axial, quintiles, windows,
                scen_label, model_name, model_inst, out_dir,
                X_tr_raw=X_tr_raw if model_name == "TabICL" else None)
            results_all[model_name][scen_label] = res

            for j, (_, row) in enumerate(test_df.iterrows()):
                for i, p in enumerate(res["param_names"]):
                    rows.append({
                        "model": model_name, "scenario": scen_label,
                        "sample_id": row["sample_id"], "config": row["config"],
                        "g_axial_true": row["g_axial"],
                        "g_axial_quintile": row["g_axial_quintile"],
                        "windows": row["windows_with_roots"],
                        "parameter": p,
                        "true_value": res["y_true"][j, i],
                        "pred_value": res["y_pred"][j, i],
                        "abs_error":  abs(res["y_true"][j,i] - res["y_pred"][j,i]),
                    })

    print("\n── Comparison plot ──")
    plot_comparison(results_all, arch_params, quintiles,
                    out_dir / "scenario_comparison.png")

    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
