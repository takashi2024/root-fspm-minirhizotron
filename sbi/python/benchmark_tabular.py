"""benchmark_tabular.py — Compare XGBoost and TabICL against MLP baseline on X_summary.

Usage (from repo root):
    python sbi/python/benchmark_tabular.py --tensors_dir sbi/data/combined/tensors --out_dir sbi/data/combined/benchmark

Outputs (in --out_dir):
    results.csv            — R², RMSE per parameter per model
    scatter_<model>.png    — colored by layout config (A/B/C/D)
    oof_<model>.npz        — cached out-of-fold predictions for scatter regeneration
    summary.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler

CONFIG_LABELS  = ["A", "B2", "C", "D2"]
CONFIG_COLORS  = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]  # blue, orange, green, pink


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="sbi/configs/sbi_config.json")
    p.add_argument("--tensors_dir",  default=None,
                   help="Override tensors dir (ignores config outdir)")
    p.add_argument("--out_dir",      default=None,
                   help="Override benchmark output dir")
    p.add_argument("--nonzero_only", action="store_true")
    p.add_argument("--kfold",        type=int, default=5)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def load_data(tensors_dir: Path, nonzero_only: bool, seed: int):
    X_raw = np.load(tensors_dir / "X_summary.npz")["X"]   # (N, 11, 15, 4, 3)
    y_raw = np.load(tensors_dir / "y.npz")["y"]           # (N, 8)
    with open(tensors_dir / "param_names.txt") as f:
        param_names = [l.strip() for l in f if l.strip()]

    N = len(X_raw)
    mask = None
    if nonzero_only:
        mask = X_raw.reshape(N, -1).sum(1) > 0
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        print(f"Nonzero filter: {mask.sum()}/{N} simulations kept")
        N = len(X_raw)

    X_flat = X_raw.reshape(N, -1).astype(np.float64)  # (N, 1980)

    # Load config_ids for scatter coloring (before layout append so mask aligns)
    config_ids = None
    config_id_path = tensors_dir / "config_ids.npy"
    if config_id_path.exists():
        config_ids = np.load(str(config_id_path))
        if mask is not None:
            config_ids = config_ids[mask]

    layout_path = tensors_dir / "layout_features.npy"
    if layout_path.exists():
        layout = np.load(str(layout_path)).astype(np.float64)
        if mask is not None:
            layout = layout[mask]
        n_configs = int(config_ids.max()) + 1
        onehot    = np.eye(n_configs, dtype=np.float64)[config_ids]
        X_flat    = np.hstack([X_flat, layout, onehot])
        print(f"Appended {layout.shape[1]} layout features + {n_configs}-way one-hot: "
              f"X_flat shape = {X_flat.shape}")
    elif config_ids is not None:
        n_configs = int(config_ids.max()) + 1
        onehot    = np.eye(n_configs, dtype=np.float64)[config_ids]
        X_flat    = np.hstack([X_flat, onehot])
        print(f"Appended {n_configs}-way one-hot config_id: X_flat shape = {X_flat.shape}")

    scaler = ParamScaler(param_names)
    y_norm = scaler.normalize(y_raw.astype(np.float64))

    return X_flat, y_norm, y_raw, param_names, scaler, config_ids


def make_models():
    from xgboost import XGBRegressor
    from tabicl import TabICLRegressor

    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        ),
        n_jobs=1,
    )

    import torch
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TabICL device: {device}")
    # TabICL treats each feature as a transformer token → 1992 features causes
    # a ~47 GB attention buffer. PCA(64) before TabICL keeps it tractable.
    # MultiOutputRegressor wraps it because TabICL's multi-output path has a shape bug.
    tabicl = MultiOutputRegressor(
        Pipeline([
            ("pca", PCA(n_components=64)),
            ("model", TabICLRegressor(random_state=42, device=device)),
        ]),
        n_jobs=1,
    )

    return {"XGBoost": xgb, "TabICL": tabicl}


def run_kfold(model, X, y, k, seed):
    """K-fold CV. Returns (y_true, y_pred, val_indices) aligned across all folds."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    preds, trues, indices = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        m_clone = clone_model(model)
        m_clone.fit(X[tr_idx], y[tr_idx])
        preds.append(m_clone.predict(X[va_idx]))
        trues.append(y[va_idx])
        indices.append(va_idx)
        print(f"  fold {fold}/{k} done")
    return np.vstack(trues), np.vstack(preds), np.concatenate(indices)


def clone_model(model):
    from sklearn.base import clone
    return clone(model)


def make_scatter(y_true_raw, y_pred_raw, param_names, model_name, out_path,
                 config_ids=None):
    n_col = 4
    n_row = int(np.ceil(len(param_names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))
    r2 = r2_score(y_true_raw, y_pred_raw)

    for i, name in enumerate(param_names):
        ax = axes.flatten()[i]
        if config_ids is not None:
            for cid, (label, color) in enumerate(zip(CONFIG_LABELS, CONFIG_COLORS)):
                sel = config_ids == cid
                ax.scatter(y_true_raw[sel, i], y_pred_raw[sel, i],
                           alpha=0.4, s=15, color=color, label=label)
            if i == 0:
                ax.legend(title="Config", fontsize=7, title_fontsize=7,
                          markerscale=1.5, handletextpad=0.3)
        else:
            ax.scatter(y_true_raw[:, i], y_pred_raw[:, i], alpha=0.5, s=20)
        lo = min(y_true_raw[:, i].min(), y_pred_raw[:, i].min())
        hi = max(y_true_raw[:, i].max(), y_pred_raw[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR²={r2[i]:.3f}")

    for ax in axes.flatten()[len(param_names):]:
        ax.set_visible(False)
    fig.suptitle(f"{model_name} — {len(y_true_raw)}-sample 5-fold CV", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.parent.parent

    if args.tensors_dir:
        tensors_dir = Path(args.tensors_dir)
        if not tensors_dir.is_absolute():
            tensors_dir = repo_root / tensors_dir
        out_dir = Path(args.out_dir) if args.out_dir else tensors_dir.parent / "benchmark"
    else:
        with open(repo_root / args.config) as f:
            cfg = json.load(f)
        tensors_dir = repo_root / cfg["outdir"] / "tensors"
        out_dir = repo_root / cfg["outdir"] / "benchmark"
        if args.out_dir:
            out_dir = Path(args.out_dir)
            if not out_dir.is_absolute():
                out_dir = repo_root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_norm, y_raw, param_names, scaler, config_ids = load_data(
        tensors_dir, nonzero_only=args.nonzero_only, seed=args.seed)
    print(f"Data: {X.shape[0]} simulations × {X.shape[1]} features, {y_norm.shape[1]} targets")

    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)

    models  = make_models()
    all_rows = []

    # Load results from a previous partial run so we can skip completed models
    existing_csv = out_dir / "results.csv"
    completed = set()
    if existing_csv.exists():
        prev = pd.read_csv(existing_csv)
        all_rows = prev.to_dict("records")
        completed = set(prev["model"].unique())
        print(f"Loaded existing results for: {completed}")

    models = {k: v for k, v in models.items() if k not in completed}
    if not models:
        print("All models already completed.")
        return

    for model_name, model in models.items():
        print(f"\n── {model_name} ({args.kfold}-fold CV) ──")
        t0 = time.time()
        y_true_norm, y_pred_norm, val_idx = run_kfold(
            model, X_scaled, y_norm, args.kfold, args.seed)
        elapsed = time.time() - t0

        y_pred_norm = np.clip(y_pred_norm, 0.0, 1.0)
        y_true_raw_cv = scaler.denormalize(y_true_norm)
        y_pred_raw_cv = scaler.denormalize(y_pred_norm)

        # Config IDs aligned to OOF order
        oof_config_ids = config_ids[val_idx] if config_ids is not None else None

        # Cache OOF predictions for later scatter regeneration
        np.savez(str(out_dir / f"oof_{model_name.lower()}.npz"),
                 y_true=y_true_raw_cv, y_pred=y_pred_raw_cv,
                 config_ids=oof_config_ids if oof_config_ids is not None else np.array([]))

        r2_n = r2_score(y_true_norm, y_pred_norm)
        r2_r = r2_score(y_true_raw_cv, y_pred_raw_cv)
        rmse = np.sqrt(np.mean((y_pred_raw_cv - y_true_raw_cv) ** 2, axis=0))

        print(f"  elapsed: {elapsed:.1f}s  mean R²(norm)={r2_n.mean():.4f}  mean R²(raw)={r2_r.mean():.4f}")
        for i, name in enumerate(param_names):
            print(f"  {name:30s}  R²={r2_r[i]:.3f}  RMSE={rmse[i]:.4f}")
            all_rows.append({"model": model_name, "parameter": name,
                             "R2_norm": r2_n[i], "R2_raw": r2_r[i], "RMSE_raw": rmse[i]})

        make_scatter(y_true_raw_cv, y_pred_raw_cv, param_names, model_name,
                     out_dir / f"scatter_{model_name.lower()}.png",
                     config_ids=oof_config_ids)

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    pivot = df.pivot(index="parameter", columns="model", values="R2_raw").round(3)

    summary_lines = [
        f"Benchmark: {args.kfold}-fold CV, N={len(X)}, features={X.shape[1]}",
        f"Nonzero only: {args.nonzero_only}",
        "", pivot.to_string(), "",
        "Mean R² per model:",
    ]
    for col in pivot.columns:
        summary_lines.append(f"  {col:10s}: {pivot[col].mean():.4f}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {out_dir}/results.csv, summary.txt, scatter_*.png")


if __name__ == "__main__":
    main()
