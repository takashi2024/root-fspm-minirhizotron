"""evaluate_tabicl_only.py — TabICL evaluation on 20 test cases, no PCA, no MultiOutputRegressor.

Mirrors the WeedMap usage: TabICLRegressor directly, one output at a time, n_estimators=4.

Usage (from repo root):
    python sbi/python/evaluate_tabicl_only.py
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
N_ESTIMATORS = 4       # matches WeedMap eval setting

QUINTILE_COLORS = {
    "Q1": "#1a9850", "Q2": "#91cf60", "Q3": "#fee08b",
    "Q4": "#fc8d59", "Q5": "#d73027",
}
QUINTILE_LABELS = {
    "Q1": "Q1 (low g)", "Q2": "Q2", "Q3": "Q3",
    "Q4": "Q4", "Q5": "Q5 (high g)",
}

REPO  = Path(__file__).parent.parent.parent
OUT   = REPO / "sbi/data/combined/test_case_eval"
OUT.mkdir(parents=True, exist_ok=True)


def load_all():
    td = REPO / "sbi/data/combined/tensors"
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


def plot_actual_vs_pred(y_true, y_pred, param_names, quintiles, title, out_path):
    n = len(param_names); ncol = min(4, n); nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 4.2*nrow))
    axf = np.array(axes).flatten()
    for i, name in enumerate(param_names):
        ax = axf[i]
        for q, color in QUINTILE_COLORS.items():
            sel = [j for j,qq in enumerate(quintiles) if qq==q]
            if sel:
                ax.scatter(y_true[sel,i], y_pred[sel,i], color=color, s=90,
                           edgecolors="white", linewidths=0.5, label=QUINTILE_LABELS[q], zorder=3)
        lo = min(y_true[:,i].min(), y_pred[:,i].min())
        hi = max(y_true[:,i].max(), y_pred[:,i].max())
        pad = (hi-lo)*0.08+1e-6
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=1,alpha=0.5)
        ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(lo-pad,hi+pad)
        ax.set_xlabel("True",fontsize=9); ax.set_ylabel("Predicted",fontsize=9)
        mae = np.abs(y_true[:,i]-y_pred[:,i]).mean()
        ax.set_title(f"{name}\nMAE={mae:.3g}",fontsize=9)
        ax.grid(True,alpha=0.2)
        if i==0: ax.legend(fontsize=7,title="g_axial",title_fontsize=7,markerscale=1.2)
    for ax in axf[n:]: ax.set_visible(False)
    fig.suptitle(title,fontsize=12,y=1.01); fig.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}", flush=True)


def plot_gaxial_vs_error(y_true, y_pred, param_names, g_axial, quintiles, windows, title, out_path):
    n = len(param_names); ncol = min(4, n); nrow = int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 4.2*nrow))
    axf = np.array(axes).flatten()
    abs_err = np.abs(y_true - y_pred)
    for i, name in enumerate(param_names):
        ax = axf[i]
        sc = ax.scatter(g_axial, abs_err[:,i], c=windows, cmap="viridis",
                        s=90, edgecolors="white", linewidths=0.5, vmin=0, vmax=max(windows), zorder=3)
        for q, color in QUINTILE_COLORS.items():
            sel = [j for j,qq in enumerate(quintiles) if qq==q]
            if sel:
                ax.scatter(g_axial[sel], abs_err[sel,i], facecolors="none",
                           edgecolors=color, s=110, linewidths=1.8, zorder=4, label=QUINTILE_LABELS[q])
        ax.set_xscale("log")
        ax.set_xlabel("True g_axial (log)",fontsize=9); ax.set_ylabel("|Error|",fontsize=9)
        ax.set_title(f"{name}\nmean={abs_err[:,i].mean():.3g}",fontsize=9)
        ax.grid(True,which="both",alpha=0.2)
        if i==0: ax.legend(fontsize=6.5,title="Quintile",title_fontsize=7,markerscale=1.0)
    for ax in axf[n:]: ax.set_visible(False)
    fig.colorbar(sc, ax=axf[min(n,ncol)-1], label="obs windows", shrink=0.8)
    fig.suptitle(title,fontsize=12,y=1.01); fig.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}", flush=True)


def run_tabicl_scenario(X_tr, y_tr, X_te, y_te, param_names,
                        target_params, known_params, X_tr_raw,
                        g_axial, quintiles, windows, scen_label):
    from tabicl import TabICLRegressor

    t_idx = [param_names.index(p) for p in target_params]
    k_idx = [param_names.index(p) for p in known_params]

    # Append known params as covariates
    Xtr = np.hstack([X_tr, y_tr[:,k_idx]]) if known_params else X_tr
    Xte = np.hstack([X_te, y_te[:,k_idx]]) if known_params else X_te

    # Filter training to informative samples
    presence  = X_tr_raw[:,:,:,:,1]
    win_count = (presence>0).reshape(len(X_tr_raw),-1).sum(1)
    mask      = win_count >= MIN_WINDOWS
    Xtr = Xtr[mask]; ytr_all = y_tr[mask]
    print(f"  Informative filter: {mask.sum()}/{len(X_tr)} samples (≥{MIN_WINDOWS} windows)", flush=True)

    # Standardize
    mu, sig = Xtr.mean(0), Xtr.std(0)+1e-8
    Xtr_s = (Xtr-mu)/sig; Xte_s = (Xte-mu)/sig

    scaler   = ParamScaler(target_params)
    ytr_norm = scaler.normalize(ytr_all[:,t_idx])
    yte_true = y_te[:,t_idx]

    # Fit one TabICLRegressor per output (no PCA, no MultiOutputRegressor — mirrors WeedMap)
    preds_norm = []
    for i, pname in enumerate(target_params):
        print(f"  [{i+1}/{len(target_params)}] {pname}…", flush=True)
        m = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42, device="cpu")
        m.fit(Xtr_s, ytr_norm[:,i])
        p = m.predict(Xte_s)
        preds_norm.append(p)
        print(f"  [{i+1}/{len(target_params)}] done", flush=True)

    y_pred_norm = np.column_stack(preds_norm)
    y_pred_norm = np.clip(y_pred_norm, 0.0, 1.0)
    y_pred = scaler.denormalize(y_pred_norm)

    print(f"  Mean MAE: {np.abs(yte_true-y_pred).mean():.4f}", flush=True)
    for i, p in enumerate(target_params):
        print(f"  {p:30s}  MAE={np.abs(yte_true[:,i]-y_pred[:,i]).mean():.4f}", flush=True)

    tag = scen_label.replace(" ","_").replace(":","")
    plot_actual_vs_pred(yte_true, y_pred, target_params, quintiles,
                        f"TabICL [{scen_label}] — True vs Predicted (N=20)",
                        OUT / f"actual_vs_pred_tabicl_{tag}.png")
    plot_gaxial_vs_error(yte_true, y_pred, target_params, g_axial, quintiles, windows,
                         f"TabICL [{scen_label}] — g_axial vs Recovery Error",
                         OUT / f"gaxial_vs_error_tabicl_{tag}.png")
    return {"y_true": yte_true, "y_pred": y_pred, "param_names": target_params}


def main():
    print("Loading data…", flush=True)
    X_base, y_all, param_names, X_raw_all = load_all()
    test_df = pd.read_csv(REPO / "calibration/test_cases_selected.csv")

    test_idx = np.array([CONFIG_ORDER.index(r["config"])*500 + int(r["sample_id"])
                         for _,r in test_df.iterrows()])
    mask = np.ones(len(X_base), dtype=bool); mask[test_idx] = False
    X_tr = X_base[mask];     y_tr = y_all[mask];     X_tr_raw = X_raw_all[mask]
    X_te = X_base[test_idx]; y_te = y_all[test_idx]

    g_axial   = test_df["g_axial"].values
    quintiles = test_df["g_axial_quintile"].tolist()
    windows   = test_df["windows_with_roots"].values
    arch_params = [p for p in param_names if p not in ELONGATION_PARAMS]

    rows = []
    for scen_label, target_params, known_params in [
        ("Scen1: all unknown", param_names,  []),
        ("Scen2: elong known", arch_params,  ELONGATION_PARAMS),
    ]:
        print(f"\n══ TabICL {scen_label} ══", flush=True)
        res = run_tabicl_scenario(X_tr, y_tr, X_te, y_te, param_names,
                                  target_params, known_params, X_tr_raw,
                                  g_axial, quintiles, windows, scen_label)
        for j, (_,row) in enumerate(test_df.iterrows()):
            for i, p in enumerate(res["param_names"]):
                rows.append({"model":"TabICL","scenario":scen_label,
                             "sample_id":row["sample_id"],"config":row["config"],
                             "g_axial_true":row["g_axial"],"g_axial_quintile":row["g_axial_quintile"],
                             "windows":row["windows_with_roots"],"parameter":p,
                             "true_value":res["y_true"][j,i],"pred_value":res["y_pred"][j,i],
                             "abs_error":abs(res["y_true"][j,i]-res["y_pred"][j,i])})

    # Merge with existing XGBoost results if present
    xgb_csv = OUT / "results_xgboost.csv"
    tabicl_df = pd.DataFrame(rows)
    tabicl_df.to_csv(OUT / "results_tabicl.csv", index=False)

    combined_csv = OUT / "results.csv"
    if combined_csv.exists():
        existing = pd.read_csv(combined_csv)
        existing = existing[existing["model"] != "TabICL"]
        combined = pd.concat([existing, tabicl_df], ignore_index=True)
    else:
        combined = tabicl_df
    combined.to_csv(combined_csv, index=False)

    print(f"\nDone. Outputs in: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
