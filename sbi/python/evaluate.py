"""evaluate.py — Evaluation metrics and scatter plots for the trained RootCNN.

Usage (from repo root):
    python sbi/python/evaluate.py --config sbi/configs/sbi_config.json

Outputs (in checkpoints dir):
    eval_metrics.csv            — RMSE, MAE, R² per parameter (normalised + raw)
    scatter_<param_name>.png    — predicted vs true scatter plots (one per parameter)
    eval_summary.txt            — plain-text summary
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_splits
from model import RootCNN
from utils import ParamScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="sbi/configs/sbi_config.json")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p.add_argument("--split",        default="test", choices=["train", "val", "test"])
    p.add_argument("--batch",        type=int, default=256)
    p.add_argument("--nonzero_only", action="store_true", help="Exclude all-zero windows")
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(torch.sigmoid(model(x)).cpu().numpy())
        targets.append(y.numpy())
    return np.vstack(preds), np.vstack(targets)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.parent.parent
    with open(repo_root / args.config) as f:
        cfg = json.load(f)

    tensors_dir = repo_root / cfg["outdir"] / "tensors"
    ckpt_dir    = repo_root / cfg["outdir"] / "checkpoints"
    ckpt_path   = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "model_best.pt"

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = make_splits(tensors_dir, nonzero_only=args.nonzero_only)
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = ds_map[args.split]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    param_names = ds.param_names

    scaler = ParamScaler.load(str(ckpt_dir / "scaler_params.json"))

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RootCNN(n_params=ds.n_params, in_channels=ds.in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred_norm, y_true_norm = predict_all(model, loader, device)

    y_pred_raw = scaler.denormalize(y_pred_norm.astype(np.float64))
    y_true_raw = scaler.denormalize(y_true_norm.astype(np.float64))

    # ── Metrics ───────────────────────────────────────────────────────────────
    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_true_norm) ** 2, axis=0))
    mae_norm  = np.mean(np.abs(y_pred_norm - y_true_norm), axis=0)
    r2_norm   = r2_score(y_true_norm, y_pred_norm)

    rmse_raw  = np.sqrt(np.mean((y_pred_raw - y_true_raw) ** 2, axis=0))
    mae_raw   = np.mean(np.abs(y_pred_raw - y_true_raw), axis=0)
    r2_raw    = r2_score(y_true_raw, y_pred_raw)

    metrics = pd.DataFrame({
        "parameter":  param_names,
        "RMSE_norm":  rmse_norm,
        "MAE_norm":   mae_norm,
        "R2_norm":    r2_norm,
        "RMSE_raw":   rmse_raw,
        "MAE_raw":    mae_raw,
        "R2_raw":     r2_raw,
    })
    metrics_path = ckpt_dir / "eval_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))

    # ── Scatter plots ─────────────────────────────────────────────────────────
    n_col = 4
    n_row = int(np.ceil(len(param_names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(y_true_raw[:, i], y_pred_raw[:, i], alpha=0.3, s=10, rasterized=True)
        lo = min(y_true_raw[:, i].min(), y_pred_raw[:, i].min())
        hi = max(y_true_raw[:, i].max(), y_pred_raw[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR²={r2_raw[i]:.3f}  RMSE={rmse_raw[i]:.4f}")

    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs True — {args.split} split  (N={len(ds)})", y=1.01)
    fig.tight_layout()
    scatter_path = ckpt_dir / f"scatter_{args.split}.png"
    fig.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # ── Text summary ─────────────────────────────────────────────────────────
    summary_path = ckpt_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation split: {args.split}  (N={len(ds)} windows)\n")
        f.write(f"Checkpoint: {ckpt_path}\n\n")
        f.write(metrics.to_string(index=False))
        f.write(f"\n\nMean R² (normalised): {r2_norm.mean():.4f}\n")
        f.write(f"Mean R² (raw):        {r2_raw.mean():.4f}\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
