"""train.py — Training loop for the RootCNN parameter regression model.

Usage (from repo root):
    python sbi/python/train.py --config sbi/configs/sbi_config.json

Outputs (in <outdir>/checkpoints/):
    model_best.pt       — best validation-loss checkpoint
    model_final.pt      — weights after last epoch
    scaler_params.json  — ParamScaler bounds for denormalisation
    train_log.csv       — per-epoch loss history
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from the python/ directory or the repo root
sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_splits, SBIDataset
from model import RootCNN, mse_loss_normalized
from utils import NoiseAugmentation, ParamScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="sbi/configs/sbi_config.json")
    p.add_argument("--tensors", default=None,
                   help="Override tensors dir (default: config outdir/tensors)")
    p.add_argument("--outdir",  default=None,
                   help="Override checkpoint output dir")
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--batch",   type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--no-aug",       action="store_true", help="Disable noise augmentation")
    p.add_argument("--nonzero_only", action="store_true", help="Exclude all-zero windows from training")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = mse_loss_normalized(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += mse_loss_normalized(logits, y).item() * len(x)
    return total_loss / len(loader.dataset)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent.parent
    with open(repo_root / args.config) as f:
        cfg = json.load(f)

    cnn_cfg = cfg.get("cnn", {})
    epochs    = args.epochs or cnn_cfg.get("epochs",    100)
    batch_sz  = args.batch  or cnn_cfg.get("batch_size", 64)
    lr        = args.lr     or cnn_cfg.get("lr",        1e-3)
    dropout   = cnn_cfg.get("dropout", 0.2)
    train_f   = cnn_cfg.get("train_frac", 0.8)
    val_f     = cnn_cfg.get("val_frac",   0.1)

    tensors_dir = Path(args.tensors) if args.tensors else \
                  repo_root / cfg["outdir"] / "tensors"
    ckpt_dir    = Path(args.outdir) if args.outdir else \
                  repo_root / cfg["outdir"] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    aug = None if args.no_aug else NoiseAugmentation(
        pixel_dropout_p=0.05, false_positive_p=0.02,
        occlusion_n=2, enabled=True
    )
    train_ds, val_ds, test_ds = make_splits(
        tensors_dir, train_frac=train_f, val_frac=val_f,
        seed=args.seed, augmentation=aug,
        nonzero_only=args.nonzero_only,
    )
    print(f"Dataset splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RootCNN(
        n_params    = train_ds.n_params,
        in_channels = train_ds.in_channels,
        dropout     = dropout,
    ).to(device)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params_total:,}")

    warmup_epochs = cnn_cfg.get("warmup_epochs", 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    # ── Save scaler ───────────────────────────────────────────────────────────
    train_ds.scaler.save(str(ckpt_dir / "scaler_params.json"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    log_rows = []

    for epoch in range(1, epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, device)
        val_loss   = eval_epoch(model, val_dl, device)
        scheduler.step()
        elapsed    = time.time() - t0

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch:4d}/{epochs}  train={train_loss:.5f}  "
              f"val={val_loss:.5f}  ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss}, ckpt_dir / "model_best.pt")

    torch.save({"epoch": epochs, "model_state": model.state_dict()},
               ckpt_dir / "model_final.pt")
    pd.DataFrame(log_rows).to_csv(ckpt_dir / "train_log.csv", index=False)
    print(f"\nBest val loss: {best_val:.5f}  →  {ckpt_dir}/model_best.pt")


if __name__ == "__main__":
    main()
