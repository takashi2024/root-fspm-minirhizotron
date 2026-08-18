"""
create_tensors_k14.py

Subsample the first 14 seeds from tensors_k20/X_raw.npz to create tensors_k14/.
No new simulations needed — tensors_k20 already has K=20 seeds per sample.

Usage (from repo root):
    python sbi/python/create_tensors_k14.py
"""

from pathlib import Path
import numpy as np
import shutil

REPO    = Path(__file__).parent.parent.parent
CONFIGS = ["A", "B2", "C", "D2"]
K14     = 14

for cfg in CONFIGS:
    src = REPO / f"sbi/data/multi_seed/config_{cfg}/tensors_k20"
    dst = REPO / f"sbi/data/multi_seed/config_{cfg}/tensors_k14"
    dst.mkdir(exist_ok=True)

    # Load raw tensor (N, K=20, T, ns, na, nc)
    X_raw = np.load(src / "X_raw.npz")["X"]
    N, K, T, ns, na, nc = X_raw.shape
    print(f"Config {cfg}: X_raw {X_raw.shape} → slicing to k={K14}")

    X14 = X_raw[:, :K14]              # (N, 14, T, ns, na, nc)
    X_mean = X14.mean(axis=1)         # (N, T, ns, na, nc)
    X_std  = X14.std(axis=1, ddof=1)  # (N, T, ns, na, nc)

    np.savez_compressed(dst / "X_raw.npz",  X=X14.astype(np.float32))
    np.savez_compressed(dst / "X_mean.npz", X=X_mean.astype(np.float32))
    np.savez_compressed(dst / "X_std.npz",  X=X_std.astype(np.float32))

    # Copy unchanged files
    for fname in ["y.npz", "param_names.txt"]:
        shutil.copy2(src / fname, dst / fname)

    # Seed list: first 14 entries
    seed_list = (src / "seed_list.txt").read_text().strip().splitlines()
    (dst / "seed_list.txt").write_text("\n".join(seed_list[:K14]) + "\n")

    print(f"  → {dst}  (X_mean {X_mean.shape})")

print("\nDone. tensors_k14/ created for all configs.")
