"""utils.py — shared utilities for the TabICL SBI pipeline."""

import json
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Parameter scaler
# ──────────────────────────────────────────────────────────────────────────────

PARAM_BOUNDS = {
    "elongation_rate_min":     (0.10,  0.70, "log"),    # was (0.05, 0.5)
    "elongation_rate_max":     (0.40,  2.00, "log"),    # was (0.2, 1.5)
    "elongation_logistic_k":   (0.05,  0.4,  "log"),
    "elongation_logistic_t50": (10.0, 80.0,  "linear"), # was (5.0, 50.0)
    "branch_angle_order1":     (40.0, 85.0,  "linear"),
    "g_axial":                 (0.01,  0.50, "log"),
    "branching_prob_order1":   (0.05,  0.90, "linear"),  # was (0.05, 1.0)
    "branching_prob_order2":   (0.0,   0.90, "linear"),  # was (0.0, 1.0)
}


class ParamScaler:
    """
    Maps raw FSPM parameter values ↔ [0, 1] normalised values.

    For log-scale parameters, normalisation is done in log-space so that the
    CNN learns a uniform distribution over orders of magnitude rather than
    concentrating on large values.
    """

    def __init__(self, param_names: list[str] | None = None):
        self.param_names = param_names or list(PARAM_BOUNDS.keys())
        self.bounds = {k: PARAM_BOUNDS[k] for k in self.param_names}

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """Raw values (N, P) → normalised [0, 1] (N, P)."""
        out = np.empty_like(y, dtype=np.float32)
        for i, name in enumerate(self.param_names):
            lo, hi, scale = self.bounds[name]
            v = y[:, i]
            if scale == "log":
                out[:, i] = (np.log(np.clip(v, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
            else:
                out[:, i] = (np.clip(v, lo, hi) - lo) / (hi - lo)
        return out

    def denormalize(self, y_norm: np.ndarray) -> np.ndarray:
        """Normalised [0, 1] (N, P) → raw values (N, P)."""
        out = np.empty_like(y_norm, dtype=np.float32)
        for i, name in enumerate(self.param_names):
            lo, hi, scale = self.bounds[name]
            v = np.clip(y_norm[:, i], 0.0, 1.0)
            if scale == "log":
                out[:, i] = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
            else:
                out[:, i] = lo + v * (hi - lo)
        return out

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"param_names": self.param_names, "bounds": self.bounds}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ParamScaler":
        with open(path) as f:
            d = json.load(f)
        scaler = cls(d["param_names"])
        scaler.bounds = {k: tuple(v) for k, v in d["bounds"].items()}
        return scaler


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic noise augmentation
# ──────────────────────────────────────────────────────────────────────────────

class NoiseAugmentation:
    """
    Applies synthetic noise to binary (0/1 float) image tensors.

    Input tensor shape: (..., T, H, W) with values in [0, 1].
    All operations are applied per-batch-element in-place on a copy.
    """

    def __init__(
        self,
        pixel_dropout_p: float = 0.05,
        false_positive_p: float = 0.02,
        occlusion_n: int = 0,
        occlusion_max_size: float = 0.25,
        brightness_range: tuple[float, float] = (0.9, 1.1),
        enabled: bool = True,
    ):
        self.pixel_dropout_p   = pixel_dropout_p
        self.false_positive_p  = false_positive_p
        self.occlusion_n       = occlusion_n
        self.occlusion_max_size = occlusion_max_size
        self.brightness_range  = brightness_range
        self.enabled           = enabled

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (..., T, H, W) float32, values in [0, 1]."""
        if not self.enabled:
            return x
        x = x.copy()
        # Random pixel dropout (remove true-positive root pixels)
        if self.pixel_dropout_p > 0:
            mask = np.random.rand(*x.shape) < self.pixel_dropout_p
            x[mask & (x > 0.5)] = 0.0
        # False positive noise (add spurious root pixels)
        if self.false_positive_p > 0:
            mask = np.random.rand(*x.shape) < self.false_positive_p
            x[mask & (x < 0.5)] = 1.0
        # Random occlusion patches (zero out rectangular regions)
        if self.occlusion_n > 0:
            H, W = x.shape[-2], x.shape[-1]
            for _ in range(self.occlusion_n):
                ph = int(np.random.uniform(0.05, self.occlusion_max_size) * H)
                pw = int(np.random.uniform(0.05, self.occlusion_max_size) * W)
                r0 = np.random.randint(0, H - ph + 1)
                c0 = np.random.randint(0, W - pw + 1)
                x[..., r0:r0 + ph, c0:c0 + pw] = 0.0
        # Brightness / contrast jitter (grayscale equivalent)
        if self.brightness_range != (1.0, 1.0):
            factor = np.random.uniform(*self.brightness_range)
            x = np.clip(x * factor, 0.0, 1.0)
        return x.astype(np.float32)
