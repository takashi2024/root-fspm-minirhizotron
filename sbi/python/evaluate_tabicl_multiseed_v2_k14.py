"""evaluate_tabicl_multiseed_v2_k14.py — TabICL evaluation on 20 test cases using
K=14 multi-seed v2 observations (skin_depth=0.2 cm).

Identical pipeline to evaluate_tabicl_multiseed_v2.py but uses the K=14 seed mean
from tensors_k14/X_mean.npz. Provides the k=14 extended baseline for the seed-count
ablation (Fig. 5 in the publication).

Data:
  Training: sbi/data/multi_seed_v2/config_*/tensors_k14/  (v2, K=14 mean)
  Test:     20 cases from calibration/test_cases_selected_v2.csv

Output:
  sbi/data/combined/test_case_eval_multiseed_v2_k14/
    results_tabicl_ms_v2_k14.csv
    actual_vs_pred_tabicl_ms_v2_k14_*.png

Usage (from repo root):
    python sbi/python/evaluate_tabicl_multiseed_v2_k14.py
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
N_JOBS       = 6

QUINTILE_COLORS = {
    "Q1": "#1a9850", "Q2": "#91cf60", "Q3": "#fee08b",
    "Q4": "#fc8d59", "Q5": "#d73027",
}
QUINTILE_LABELS = {
    "Q1": "Q1 (low g)", "Q2": "Q2", "Q3": "Q3",
    "Q4": "Q4", "Q5": "Q5 (high g)",
}

REPO = Path(__file__).parent.parent.parent
OUT  = REPO / "sbi/data/combined/test_case_eval_multiseed_v2_k14"
OUT.mkdir(parents=True, exist_ok=True)

_LAYOUT_PATH  = REPO / "sbi/data/combined/tensors/layout_features.npy"
_LAYOUT_NAMES = REPO / "sbi/data/combined/tensors/layout_feature_names.txt"


def _per_config_layout() -> dict:
    layout = np.load(_LAYOUT_PATH)   # (2000, 8)
    return {cfg: layout[i * 500].astype(np.float64)
            for i, cfg in enumerate(CONFIG_ORDER)}


def load_k14_v2() -> tuple:
    """Load v2 K=14 multi-seed tensors for all 4 configs."""
    layout_map = _per_config_layout()
    all_mean, all_y, all_meta = [], [], []
    param_names = None

    for cfg in CONFIG_ORDER:
        td  = REPO / f"sbi/data/multi_seed_v2/config_{cfg}/tensors_k14"
        X_m = np.load(td / "X_mean.npz")["X"]   # (N_c, T, ns, na, 3)
        y_c = np.load(td / "y.npz")["y"]
        df_c = pd.read_csv(REPO / f"sbi/data/multi_seed_v2/config_{cfg}/samples_filtered.csv")
        df_c["config"] = cfg

        if param_names is None:
            with open(td / "param_names.txt") as f:
                param_names = [l.strip() for l in f if l.strip()]

        all_mean.append(X_m)
        all_y.append(y_c)
        all_meta.append(df_c)

    X_mean_all = np.concatenate(all_mean, axis=0).astype(np.float32)
    y_all      = np.concatenate(all_y,    axis=0).astype(np.float64)
    df_meta    = pd.concat(all_meta, ignore_index=True)
    N          = len(X_mean_all)

    X_flat = X_mean_all.reshape(N, -1).astype(np.float64)

    layout_rows = np.array([layout_map[cfg] for cfg in df_meta["config"]])
    cfg_id  = np.array([CONFIG_ORDER.index(cfg) for cfg in df_meta["config"]])
    onehot  = np.eye(len(CONFIG_ORDER), dtype=np.float64)[cfg_id]
    X_feat  = np.hstack([X_flat, layout_rows, onehot])

    print(f"Loaded v2 K=14 data: N={N}  feat_dim={X_feat.shape[1]}")
    return X_feat, X_mean_all, y_all, df_meta, param_names


def _resolve_test_indices(df_meta, test_df):
    df_meta = df_meta.reset_index(drop=True)
    key     = df_meta["config"] + "_" + df_meta["sample_id"].astype(str)
    test_keys = test_df["config"] + "_" + test_df["sample_id"].astype(str)
    found = test_keys.isin(key.values)
    if not found.all():
        missing = test_df[~found][["config", "sample_id"]].values.tolist()
        print(f"WARNING: {(~found).sum()} test cases not in v2 K=14 data: {missing}")

    key_to_idx = {k: i for i, k in enumerate(key)}
    test_indices, valid_test_rows = [], []
    for _, r in test_df.iterrows():
        k = f"{r['config']}_{int(r['sample_id'])}"
        if k in key_to_idx:
            test_indices.append(key_to_idx[k])
            valid_test_rows.append(r)

    te_mask = np.zeros(len(df_meta), dtype=bool)
    te_mask[test_indices] = True
    return te_mask, test_indices, pd.DataFrame(valid_test_rows).reset_index(drop=True)


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


def run_tabicl_scenario(X_tr, y_tr, X_te, y_te, param_names,
                        target_params, known_params, X_mean_tr,
                        g_axial, quintiles, windows, scen_label):
    from tabicl import TabICLRegressor

    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    Xtr = np.hstack([X_tr, y_tr[:, k_idx]]) if known_params else X_tr
    Xte = np.hstack([X_te, y_te[:, k_idx]]) if known_params else X_te

    presence  = X_mean_tr[:, :, :, :, 1]
    win_count = (presence > 0).reshape(len(X_mean_tr), -1).sum(1)
    mask      = win_count >= MIN_WINDOWS
    Xtr = Xtr[mask]; ytr_all = y_tr[mask]
    print(f"  Informative filter: {mask.sum()}/{len(X_tr)} samples", flush=True)

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

    y_pred_norm = np.clip(np.column_stack(preds_norm), 0.0, 1.0)
    y_pred      = scaler.denormalize(y_pred_norm)

    print(f"  Mean MAE: {np.abs(yte_true - y_pred).mean():.4f}", flush=True)
    tag_safe = scen_label.replace(" ", "_").replace(":", "")
    plot_actual_vs_pred(
        yte_true, y_pred, target_params, quintiles,
        f"TabICL v2 K=14 [{scen_label}] — True vs Predicted (N={len(y_te)})",
        OUT / f"actual_vs_pred_tabicl_ms_v2_k14_{tag_safe}.png",
    )
    return {"y_true": yte_true, "y_pred": y_pred, "param_names": target_params}


def main():
    print("Loading v2 K=14 data…", flush=True)
    X_feat, X_mean_all, y_all, df_meta, param_names = load_k14_v2()
    test_df = pd.read_csv(REPO / "calibration/test_cases_selected_v2.csv")

    te_mask, test_indices, valid_test_df = _resolve_test_indices(df_meta, test_df)
    tr_mask = ~te_mask

    X_tr      = X_feat[tr_mask];      y_tr      = y_all[tr_mask]
    X_mean_tr = X_mean_all[tr_mask]
    X_te      = X_feat[test_indices]; y_te      = y_all[test_indices]

    g_axial   = valid_test_df["g_axial"].values
    quintiles = valid_test_df["g_axial_quintile"].tolist()
    windows   = valid_test_df["windows_with_roots"].values
    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]

    print(f"\nTrain: {tr_mask.sum()} samples  Test: {len(test_indices)} cases")

    rows = []
    for scen_label, target_params, known_params in [
        ("Scen1: all unknown", param_names, []),
        ("Scen2: elong known", arch_params, ELONGATION_PARAMS),
    ]:
        print(f"\n══ TabICL v2 K=14 {scen_label} ══", flush=True)
        res = run_tabicl_scenario(
            X_tr, y_tr, X_te, y_te, param_names,
            target_params, known_params, X_mean_tr,
            g_axial, quintiles, windows, scen_label,
        )
        for j, (_, row) in enumerate(valid_test_df.iterrows()):
            for i, p in enumerate(res["param_names"]):
                rows.append({
                    "model":            "TabICL-MS-k14",
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
    df_out.to_csv(OUT / "results_tabicl_ms_v2_k14.csv", index=False)
    print(f"\nSaved: {OUT}/results_tabicl_ms_v2_k14.csv", flush=True)

    print("\n── Mean |% error| per parameter ──")
    for scen in df_out["scenario"].unique():
        sub = df_out[df_out["scenario"] == scen]
        print(f"\n{scen}")
        for p in sub["parameter"].unique():
            psub = sub[sub["parameter"] == p]
            pct  = (psub["abs_error"] / (psub["true_value"].abs() + 1e-8) * 100).mean()
            print(f"  {p:35s}  {pct:.1f}%")

    print(f"\nDone. Outputs in: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
