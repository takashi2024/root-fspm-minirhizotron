"""train_summary.py — MLP regression on X_summary features (per-simulation).

Uses the aggregated (RLD, presence, angle) summary statistics instead of raw
FOV images. Input shape: (N_sim, 11*15*4*3=1980). One sample per simulation.

Usage (from repo root):
    python sbi/python/train_summary.py --config sbi/configs/sbi_config.json

Outputs (in <outdir>/checkpoints_summary/):
    model_best.pt
    scaler_params.json
    scaler_features.json
    train_log.csv
    eval_metrics.csv
    scatter_test.png
    eval_summary.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import ParamScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sbi/configs/sbi_config.json")
    p.add_argument("--outdir", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--nonzero_only", action="store_true",
                   help="Exclude simulations with all-zero summary (no visible roots)")
    return p.parse_args()


class SummaryMLP(nn.Module):
    def __init__(self, in_dim: int, n_params: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, n_params),
        )

    def forward(self, x):
        return self.net(x)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo_root = Path(__file__).parent.parent.parent
    with open(repo_root / args.config) as f:
        cfg = json.load(f)

    tensors_dir = repo_root / cfg["outdir"] / "tensors"
    ckpt_dir = Path(args.outdir) if args.outdir else \
               repo_root / cfg["outdir"] / "checkpoints_summary"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_raw = np.load(tensors_dir / "X_summary.npz")["X"]   # (N, 11, 15, 4, 3)
    y_raw = np.load(tensors_dir / "y.npz")["y"]           # (N, 8)
    with open(tensors_dir / "param_names.txt") as f:
        param_names = [l.strip() for l in f if l.strip()]

    N = len(X_raw)

    # Optional: filter to simulations that produced any visible roots
    if args.nonzero_only:
        mask = X_raw.reshape(N, -1).sum(1) > 0
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        print(f"Nonzero filter: {mask.sum()}/{N} simulations kept")
        N = len(X_raw)

    # ── Flatten and z-score the features ─────────────────────────────────────
    X_flat = X_raw.reshape(N, -1).astype(np.float64)      # (N, 1980)
    feat_mean = X_flat.mean(axis=0)
    feat_std  = X_flat.std(axis=0) + 1e-8
    X_norm = ((X_flat - feat_mean) / feat_std).astype(np.float32)

    # Save feature scaler
    np.savez(str(ckpt_dir / "feature_scaler.npz"), mean=feat_mean, std=feat_std)

    # ── Normalise targets ─────────────────────────────────────────────────────
    scaler = ParamScaler(param_names)
    y_norm = scaler.normalize(y_raw.astype(np.float64)).astype(np.float32)
    scaler.save(str(ckpt_dir / "scaler_params.json"))

    # ── Train/val/test split (simulation-level, 80/10/10) ────────────────────
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_train = int(N * 0.8)
    n_val   = int(N * 0.1)
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    def _ds(idx_arr):
        return TensorDataset(
            torch.from_numpy(X_norm[idx_arr]),
            torch.from_numpy(y_norm[idx_arr]),
        )

    train_dl = DataLoader(_ds(tr), batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(_ds(va), batch_size=args.batch, shuffle=False, num_workers=0)
    test_dl  = DataLoader(_ds(te), batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Splits — train: {len(tr)}, val: {len(va)}, test: {len(te)}")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    in_dim = X_norm.shape[1]
    model = SummaryMLP(in_dim, len(param_names)).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_total:,}  input_dim: {in_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[10])

    criterion = nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred   = torch.sigmoid(logits)
                loss   = criterion(pred, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(x)
        return total / len(loader.dataset)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    patience, patience_limit = 0, 40
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        tr_l = run_epoch(train_dl, train=True)
        va_l = run_epoch(val_dl,   train=False)
        scheduler.step()

        log_rows.append({"epoch": epoch, "train_loss": tr_l, "val_loss": va_l})
        print(f"Epoch {epoch:4d}/{args.epochs}  train={tr_l:.5f}  val={va_l:.5f}  ({time.time()-t0:.1f}s)")

        if va_l < best_val:
            best_val = va_l
            patience = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": va_l}, ckpt_dir / "model_best.pt")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
                break

    pd.DataFrame(log_rows).to_csv(ckpt_dir / "train_log.csv", index=False)
    print(f"\nBest val loss: {best_val:.5f}  →  {ckpt_dir}/model_best.pt")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    ckpt = torch.load(ckpt_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            targets.append(y.numpy())
    y_pred_norm = np.vstack(preds)
    y_true_norm = np.vstack(targets)

    y_pred_raw = scaler.denormalize(y_pred_norm.astype(np.float64))
    y_true_raw = scaler.denormalize(y_true_norm.astype(np.float64))

    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_true_norm) ** 2, axis=0))
    r2_norm   = r2_score(y_true_norm, y_pred_norm)
    rmse_raw  = np.sqrt(np.mean((y_pred_raw - y_true_raw) ** 2, axis=0))
    r2_raw    = r2_score(y_true_raw, y_pred_raw)

    metrics = pd.DataFrame({
        "parameter": param_names,
        "RMSE_norm": rmse_norm,
        "R2_norm":   r2_norm,
        "RMSE_raw":  rmse_raw,
        "R2_raw":    r2_raw,
    })
    metrics.to_csv(ckpt_dir / "eval_metrics.csv", index=False)
    print("\n" + metrics.to_string(index=False))
    print(f"\nMean R² (normalised): {r2_norm.mean():.4f}")
    print(f"Mean R² (raw):        {r2_raw.mean():.4f}")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    n_col = 4
    n_row = int(np.ceil(len(param_names) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))
    for i, name in enumerate(param_names):
        ax = axes.flatten()[i]
        ax.scatter(y_true_raw[:, i], y_pred_raw[:, i], alpha=0.5, s=20)
        lo = min(y_true_raw[:, i].min(), y_pred_raw[:, i].min())
        hi = max(y_true_raw[:, i].max(), y_pred_raw[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR²={r2_raw[i]:.3f}")
    for ax in axes.flatten()[len(param_names):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(ckpt_dir / "scatter_test.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {ckpt_dir}/scatter_test.png")


if __name__ == "__main__":
    main()
