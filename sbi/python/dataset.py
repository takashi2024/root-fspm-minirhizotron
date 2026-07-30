"""dataset.py — PyTorch Dataset for the SBI pipeline."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import NoiseAugmentation, ParamScaler


class SBIDataset(Dataset):
    """
    Per-window B&W FOV image dataset.

    Each item is one (station, azimuth) observation window for one simulation.
    Input:  T temporal frames stacked as channels → tensor (T, H, W) float32
    Target: 8 normalised parameter values → tensor (8,) float32

    Data source:
        X_fov.npz          — shape (N_win, T, H, W) uint8
        X_fov_window_ids.npz — shape (N_win, 3) int32: (sample_id, station_idx, azimuth_idx)
        y.npz              — shape (N_sim, 8) float32 raw param values
        param_names.txt    — one name per line

    The y tensor rows are indexed by (sample_id - 1) so that all windows from the
    same simulation share the same label.
    """

    def __init__(
        self,
        tensors_dir: str | Path,
        indices: list[int] | None = None,
        augmentation: NoiseAugmentation | None = None,
        scaler: ParamScaler | None = None,
    ):
        tensors_dir = Path(tensors_dir)

        fov_data   = np.load(tensors_dir / "X_fov.npz")
        win_data   = np.load(tensors_dir / "X_fov_window_ids.npz")
        y_data     = np.load(tensors_dir / "y.npz")

        self.X_raw     = fov_data["X"]          # (N_win, T, H, W) uint8
        self.win_ids   = win_data["window_ids"]  # (N_win, 3) int32
        self.y_raw     = y_data["y"]             # (N_sim, 8) float32

        with open(tensors_dir / "param_names.txt") as f:
            self.param_names = [l.strip() for l in f if l.strip()]

        self.scaler      = scaler or ParamScaler(self.param_names)
        self.augmentation = augmentation

        if indices is not None:
            self.X_raw   = self.X_raw[indices]
            self.win_ids = self.win_ids[indices]

        # Pre-normalise labels (broadcast by sample_id)
        sample_ids   = self.win_ids[:, 0]              # 1-indexed
        y_by_sim     = self.y_raw[sample_ids - 1, :]   # (N_win, 8)
        self.y_norm  = self.scaler.normalize(y_by_sim.astype(np.float64))  # (N_win, 8)

    def __len__(self) -> int:
        return len(self.X_raw)

    def __getitem__(self, idx: int):
        # Convert uint8 [0, 255] → float32 [0, 1]
        x = self.X_raw[idx].astype(np.float32) / 255.0  # (T, H, W)
        if self.augmentation is not None:
            x = self.augmentation(x)
        return (
            torch.from_numpy(x),                          # (T, H, W)
            torch.from_numpy(self.y_norm[idx]),           # (8,)
        )

    @property
    def n_params(self) -> int:
        return self.y_norm.shape[1]

    @property
    def in_channels(self) -> int:
        return self.X_raw.shape[1]  # T

    @property
    def img_size(self) -> tuple[int, int]:
        return self.X_raw.shape[2], self.X_raw.shape[3]


def make_splits(
    tensors_dir: str | Path,
    train_frac: float = 0.8,
    val_frac:   float = 0.1,
    seed:       int   = 42,
    augmentation: NoiseAugmentation | None = None,
    nonzero_only: bool = False,
) -> tuple["SBIDataset", "SBIDataset", "SBIDataset"]:
    """
    Create train / val / test splits that are stratified by simulation (not window).

    All windows from the same simulation are kept in the same split to avoid
    data leakage through shared labels.

    nonzero_only: if True, exclude all-zero windows from every split. Blank windows
    carry no discriminative information (all-zero images are identical regardless of
    parameters), so filtering them out improves the signal-to-noise ratio significantly
    when root density is sparse (<30% non-zero windows).
    """
    tensors_dir = Path(tensors_dir)
    X_raw   = np.load(tensors_dir / "X_fov.npz")["X"]          # (N_win, T, H, W)
    win_ids = np.load(tensors_dir / "X_fov_window_ids.npz")["window_ids"]
    sim_ids = np.unique(win_ids[:, 0])

    rng = np.random.default_rng(seed)
    rng.shuffle(sim_ids)

    n_train = int(len(sim_ids) * train_frac)
    n_val   = int(len(sim_ids) * val_frac)
    train_sims = set(sim_ids[:n_train])
    val_sims   = set(sim_ids[n_train:n_train + n_val])

    nonzero_mask = X_raw.reshape(len(X_raw), -1).sum(1) > 0  # (N_win,) bool

    def _idx(sim_set):
        return [i for i, sid in enumerate(win_ids[:, 0])
                if sid in sim_set and (not nonzero_only or nonzero_mask[i])]

    train_idx = _idx(train_sims)
    val_idx   = _idx(val_sims)
    test_idx  = [i for i, sid in enumerate(win_ids[:, 0])
                 if sid not in train_sims and sid not in val_sims
                 and (not nonzero_only or nonzero_mask[i])]

    scaler = ParamScaler()
    train_ds = SBIDataset(tensors_dir, train_idx, augmentation=augmentation, scaler=scaler)
    val_ds   = SBIDataset(tensors_dir, val_idx,   augmentation=None,          scaler=scaler)
    test_ds  = SBIDataset(tensors_dir, test_idx,  augmentation=None,          scaler=scaler)

    return train_ds, val_ds, test_ds
