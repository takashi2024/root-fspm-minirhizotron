"""evaluate_tabicl_multiseed.py — TabICL evaluation on 20 test cases using
K=1 multi-seed observations (mean only as features).

Counterpart to evaluate_tabicl_only.py (single-seed baseline).
The original script is NOT modified — this is a standalone comparison.

Feature construction:
  X_mean (N, T, ns, na, 3) flattened → (N, T*ns*na*3 = 1980).
  Layout features (8 per config) and config one-hot (4) are appended → 1992 total.
  X_std is intentionally excluded: with N~1200 and p~4000 the ratio p/N≈3.3 is
  unfavorable; using mean only matches the original single-seed feature dimension
  while benefiting from the better (K=7-averaged) mean estimate.

Data:
  Training: sbi/data/multi_seed/config_*/tensors/  (1270 filtered samples)
  Test:     20 cases from calibration/test_cases_selected.csv

Output:
  sbi/data/combined/test_case_eval_multiseed/
    results_tabicl_multiseed_k1.csv
    actual_vs_pred_tabicl_multiseed_k1_*.png
    gaxial_vs_error_tabicl_multiseed_k1_*.png

Usage (from repo root):
    python sbi/python/evaluate_tabicl_multiseed.py
"""

import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler

ELONGATION_PARAMS = [
    "elongation_rate_min", "elongation_rate_max",
    "elongation_logistic_k", "elongation_logistic_t50",
]
CONFIG_ORDER = ["A", "B2", "C", "D2"]
MIN_WINDOWS  = 10
N_ESTIMATORS = 4
N_JOBS       = 6   # PyTorch CPU threads during TabICL predict (M3: 8 physical cores)

QUINTILE_COLORS = {
    "Q1": "#1a9850", "Q2": "#91cf60", "Q3": "#fee08b",
    "Q4": "#fc8d59", "Q5": "#d73027",
}
QUINTILE_LABELS = {
    "Q1": "Q1 (low g)", "Q2": "Q2", "Q3": "Q3",
    "Q4": "Q4", "Q5": "Q5 (high g)",
}

REPO = Path(__file__).parent.parent.parent
OUT  = REPO / "sbi/data/combined/test_case_eval_multiseed_k1"
OUT.mkdir(parents=True, exist_ok=True)

# Per-config layout features from the combined tensor (8 values, constant per config)
_LAYOUT_PATH  = REPO / "sbi/data/combined/tensors/layout_features.npy"
_LAYOUT_NAMES = REPO / "sbi/data/combined/tensors/layout_feature_names.txt"


def _per_config_layout() -> dict:
    """Returns {config: layout_vector (8,)} from the combined tensor."""
    layout = np.load(_LAYOUT_PATH)    # (2000, 8)
    return {cfg: layout[i * 500].astype(np.float64)
            for i, cfg in enumerate(CONFIG_ORDER)}


def load_multiseed() -> tuple:
    """Load and concatenate multi-seed tensors for all 4 configs.

    Returns
    -------
    X_feat     : (N, T*ns*na*3 + 8 + 4 = 1992)  float64  — X_mean only
    X_mean_raw : (N, T, ns, na, 3)               float32  — for informative-window filter
    y          : (N, 8)                           float64
    df_meta    : DataFrame
    param_names: list[str]
    """
    layout_map = _per_config_layout()
    all_mean, all_y, all_meta = [], [], []

    param_names = None
    for cfg in CONFIG_ORDER:
        td  = REPO / f"sbi/data/multi_seed/config_{cfg}/tensors_k1"
        X_m = np.load(td / "X_mean.npz")["X"]   # (N_c, T, ns, na, 3)
        y_c = np.load(td / "y.npz")["y"]         # (N_c, 8)
        df_c = pd.read_csv(REPO / f"sbi/data/multi_seed/config_{cfg}/samples_filtered.csv")
        df_c["config"] = cfg

        if param_names is None:
            with open(td / "param_names.txt") as f:
                param_names = [l.strip() for l in f if l.strip()]

        all_mean.append(X_m)
        all_y.append(y_c)
        all_meta.append(df_c)

    X_mean_all = np.concatenate(all_mean, axis=0).astype(np.float32)   # (N,T,ns,na,3)
    y_all      = np.concatenate(all_y,    axis=0).astype(np.float64)   # (N,8)
    df_meta    = pd.concat(all_meta, ignore_index=True)
    N          = len(X_mean_all)

    # Feature: X_mean only, flattened
    X_flat = X_mean_all.reshape(N, -1).astype(np.float64)   # (N, T*ns*na*3 = 1980)

    # Layout features (constant per config)
    layout_rows = np.array([layout_map[cfg] for cfg in df_meta["config"]])  # (N, 8)

    # Config one-hot (4)
    cfg_id = np.array([CONFIG_ORDER.index(cfg) for cfg in df_meta["config"]])
    onehot = np.eye(len(CONFIG_ORDER), dtype=np.float64)[cfg_id]

    X_feat = np.hstack([X_flat, layout_rows, onehot])

    print(f"Loaded multi-seed data: N={N}  feat_dim={X_feat.shape[1]}")
    print(f"  X_mean flattened: {X_flat.shape[1]}  layout: 8  onehot: 4")
    return X_feat, X_mean_all, y_all, df_meta, param_names


def _resolve_test_indices(df_meta: pd.DataFrame, test_df: pd.DataFrame):
    """Match 20 test cases to rows in df_meta by (config, sample_id).

    Returns boolean mask over df_meta (True = test case).
    Also returns aligned arrays for g_axial, quintile, windows per test row.
    """
    df_meta = df_meta.reset_index(drop=True)
    key = df_meta["config"] + "_" + df_meta["sample_id"].astype(str)

    test_keys = test_df["config"] + "_" + test_df["sample_id"].astype(str)
    found = test_keys.isin(key.values)

    if not found.all():
        missing = test_df[~found][["config", "sample_id"]].values.tolist()
        print(f"WARNING: {(~found).sum()} test cases not in multi-seed data "
              f"(no visible roots): {missing}")

    # Build index map: test_key → position in df_meta
    key_to_idx = {k: i for i, k in enumerate(key)}
    test_indices = []
    valid_test_rows = []
    for _, r in test_df.iterrows():
        k = f"{r['config']}_{int(r['sample_id'])}"
        if k in key_to_idx:
            test_indices.append(key_to_idx[k])
            valid_test_rows.append(r)

    te_mask = np.zeros(len(df_meta), dtype=bool)
    te_mask[test_indices] = True

    valid_test_df = pd.DataFrame(valid_test_rows).reset_index(drop=True)
    return te_mask, test_indices, valid_test_df


# ── Plotting helpers (same style as evaluate_tabicl_only.py) ─────────────────

def plot_actual_vs_pred(y_true, y_pred, param_names, quintiles, title, out_path):
    n = len(param_names); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
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
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True", fontsize=9); ax.set_ylabel("Predicted", fontsize=9)
        mae = np.abs(y_true[:, i] - y_pred[:, i]).mean()
        ax.set_title(f"{name}\nMAE={mae:.3g}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, title="g_axial", title_fontsize=7, markerscale=1.2)
    for ax in axf[n:]: ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.01); fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}", flush=True)


def plot_gaxial_vs_error(y_true, y_pred, param_names, g_axial, quintiles,
                          windows, title, out_path):
    n = len(param_names); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
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
                           edgecolors=color, s=110, linewidths=1.8,
                           zorder=4, label=QUINTILE_LABELS[q])
        ax.set_xscale("log")
        ax.set_xlabel("True g_axial (log)", fontsize=9)
        ax.set_ylabel("|Error|", fontsize=9)
        ax.set_title(f"{name}\nmean={abs_err[:, i].mean():.3g}", fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6.5, title="Quintile", title_fontsize=7, markerscale=1.0)
    for ax in axf[n:]: ax.set_visible(False)
    fig.colorbar(sc, ax=axf[min(n, ncol) - 1], label="obs windows", shrink=0.8)
    fig.suptitle(title, fontsize=12, y=1.01); fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}", flush=True)


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_tabicl_scenario(X_tr, y_tr, X_te, y_te, param_names,
                        target_params, known_params, X_mean_tr,
                        g_axial, quintiles, windows, scen_label):
    from tabicl import TabICLRegressor

    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    # Append known params as covariates (true values for Scen2)
    Xtr = np.hstack([X_tr, y_tr[:, k_idx]]) if known_params else X_tr
    Xte = np.hstack([X_te, y_te[:, k_idx]]) if known_params else X_te

    # Informative filter: count (T, ns, na) cells where mean root presence > 0
    presence  = X_mean_tr[:, :, :, :, 1]   # (N_tr, T, ns, na)
    win_count = (presence > 0).reshape(len(X_mean_tr), -1).sum(1)
    mask      = win_count >= MIN_WINDOWS
    Xtr = Xtr[mask]; ytr_all = y_tr[mask]
    print(f"  Informative filter: {mask.sum()}/{len(X_tr)} samples "
          f"(≥{MIN_WINDOWS} windows)", flush=True)

    # Standardize (fit on train, apply to test)
    mu, sig = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sig
    Xte_s = (Xte - mu) / sig

    scaler    = ParamScaler(target_params)
    ytr_norm  = scaler.normalize(ytr_all[:, t_idx])
    yte_true  = y_te[:, t_idx]

    preds_norm = []
    for i, pname in enumerate(target_params):
        print(f"  [{i+1}/{len(target_params)}] {pname}…", flush=True)
        m = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42,
                            device="cpu", n_jobs=N_JOBS)
        m.fit(Xtr_s, ytr_norm[:, i])
        p = m.predict(Xte_s)
        preds_norm.append(p)
        print(f"  [{i+1}/{len(target_params)}] done", flush=True)

    y_pred_norm = np.column_stack(preds_norm)
    y_pred_norm = np.clip(y_pred_norm, 0.0, 1.0)
    y_pred      = scaler.denormalize(y_pred_norm)

    print(f"  Mean MAE: {np.abs(yte_true - y_pred).mean():.4f}", flush=True)
    for i, p in enumerate(target_params):
        pct_err = (np.abs(yte_true[:, i] - y_pred[:, i]) /
                   (np.abs(yte_true[:, i]) + 1e-8) * 100).mean()
        print(f"  {p:30s}  MAE={np.abs(yte_true[:,i]-y_pred[:,i]).mean():.4f}  "
              f"|%err|={pct_err:.1f}%", flush=True)

    tag = scen_label.replace(" ", "_").replace(":", "")
    plot_actual_vs_pred(
        yte_true, y_pred, target_params, quintiles,
        f"TabICL-MS-k1 [{scen_label}] — True vs Predicted (N={len(y_te)})",
        OUT / f"actual_vs_pred_tabicl_multiseed_k1_{tag}.png",
    )
    plot_gaxial_vs_error(
        yte_true, y_pred, target_params, g_axial, quintiles, windows,
        f"TabICL-MS-k1 [{scen_label}] — g_axial vs Recovery Error",
        OUT / f"gaxial_vs_error_tabicl_multiseed_k1_{tag}.png",
    )
    return {"y_true": yte_true, "y_pred": y_pred, "param_names": target_params}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading multi-seed data…", flush=True)
    X_feat, X_mean_all, y_all, df_meta, param_names = load_multiseed()
    test_df = pd.read_csv(REPO / "calibration/test_cases_selected.csv")

    te_mask, test_indices, valid_test_df = _resolve_test_indices(df_meta, test_df)
    tr_mask = ~te_mask

    X_tr      = X_feat[tr_mask];           y_tr      = y_all[tr_mask]
    X_mean_tr = X_mean_all[tr_mask]
    X_te      = X_feat[test_indices];      y_te      = y_all[test_indices]

    g_axial   = valid_test_df["g_axial"].values
    quintiles = valid_test_df["g_axial_quintile"].tolist()
    windows   = valid_test_df["windows_with_roots"].values
    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]

    print(f"\nTrain: {tr_mask.sum()} samples  Test: {len(test_indices)} cases")
    print(f"Feature dim: {X_feat.shape[1]}", flush=True)

    rows = []
    for scen_label, target_params, known_params in [
        ("Scen1: all unknown", param_names, []),
        ("Scen2: elong known", arch_params, ELONGATION_PARAMS),
    ]:
        print(f"\n══ TabICL multiseed {scen_label} ══", flush=True)
        res = run_tabicl_scenario(
            X_tr, y_tr, X_te, y_te, param_names,
            target_params, known_params, X_mean_tr,
            g_axial, quintiles, windows, scen_label,
        )
        for j, (_, row) in enumerate(valid_test_df.iterrows()):
            for i, p in enumerate(res["param_names"]):
                rows.append({
                    "model":            "TabICL-MS-k1",
                    "scenario":         scen_label,
                    "sample_id":        row["sample_id"],
                    "config":           row["config"],
                    "g_axial_true":     row["g_axial"],
                    "g_axial_quintile": row["g_axial_quintile"],
                    "windows":          row["windows_with_roots"],
                    "parameter":        p,
                    "true_value":       res["y_true"][j, i],
                    "pred_value":       res["y_pred"][j, i],
                    "abs_error":        abs(res["y_true"][j, i] - res["y_pred"][j, i]),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "results_tabicl_multiseed_k1.csv", index=False)
    print(f"\nSaved results: {OUT / 'results_tabicl_multiseed_k1.csv'}", flush=True)

    # Summary table
    print("\n── Mean |% error| per parameter ──")
    for scen in df_out["scenario"].unique():
        sub = df_out[df_out["scenario"] == scen]
        print(f"\n{scen}")
        for p in sub["parameter"].unique():
            psub = sub[sub["parameter"] == p]
            pct = (psub["abs_error"] / (psub["true_value"].abs() + 1e-8) * 100).mean()
            print(f"  {p:35s}  {pct:.1f}%")

    print(f"\nDone. Outputs in: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
