"""model.py — CNN architectures for FSPM parameter regression.

Option B: per-window model.
  Input:  (batch, T=11, H=64, W=64) — T temporal B&W FOV frames as channels
  Output: (batch, n_params=8) normalised parameter vector

During inference on a full observation (one simulation), forward() is called for
each of the 60 (station × azimuth) windows and the predictions are mean-pooled
before the final parameter estimate is returned.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Building block
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline deterministic model
# ──────────────────────────────────────────────────────────────────────────────

class RootCNN(nn.Module):
    """
    Baseline CNN for FSPM parameter regression.

    Architecture:
        Block 1: in_channels → 64              (64×64, no pool)
        Block 2: 64 → 128,  MaxPool2d(2)       (32×32)
        Block 3: 128 → 256, MaxPool2d(2)       (16×16)
        Block 4: 256 → 256, AdaptiveAvgPool2d  (1×1)
        Head:    256 → 128 → n_params

    Output: raw logits; apply Sigmoid outside to get [0,1] normalised values.
    During training pass raw logits to BCEWithLogitsLoss or MSELoss(Sigmoid(·)).
    """

    def __init__(
        self,
        n_params:    int = 8,
        in_channels: int = 11,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, 64,  pool=False),
            _ConvBlock(64,  128, pool=True),
            _ConvBlock(128, 256, pool=True),
            _ConvBlock(256, 256, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_params),
        )
        self.n_params = n_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, H, W) → logits (batch, n_params)."""
        return self.head(self.encoder(x))

    def predict_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Sigmoid(logits) — normalised [0, 1] parameter predictions."""
        return torch.sigmoid(self(x))

    def pool_windows(self, x_windows: torch.Tensor) -> torch.Tensor:
        """
        x_windows: (n_windows, T, H, W)  — all FOV windows for one simulation.
        Returns mean-pooled normalised prediction: (n_params,).
        """
        logits = self(x_windows)               # (n_windows, n_params)
        return torch.sigmoid(logits).mean(0)   # (n_params,)


# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic extension
# ──────────────────────────────────────────────────────────────────────────────

class RootCNNProbabilistic(nn.Module):
    """
    Probabilistic extension of RootCNN.

    Outputs (mean_norm, log_var) per parameter, both in normalised space.
    The mean is in [0, 1] (Sigmoid applied); log_var is clamped to [-10, 5]
    for numerical stability.

    Loss: Gaussian NLL = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
    """

    def __init__(
        self,
        n_params:    int = 8,
        in_channels: int = 11,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, 64,  pool=False),
            _ConvBlock(64,  128, pool=True),
            _ConvBlock(128, 256, pool=True),
            _ConvBlock(256, 256, pool=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head_mean   = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, n_params),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_params),
        )
        self.n_params = n_params

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_norm, log_var) each of shape (batch, n_params)."""
        features = self.encoder(x)
        mean_norm = torch.sigmoid(self.head_mean(features))
        log_var   = self.head_logvar(features).clamp(-10.0, 5.0)
        return mean_norm, log_var

    def pool_windows(self, x_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean-pool predictions across all windows.
        Returns (mean_norm, log_var) each of shape (n_params,).
        """
        means, log_vars = self(x_windows)   # (n_win, n_params) each
        pooled_mean     = means.mean(0)
        # Pool variances: Var[mean] ≈ mean(var) / n_win (law of total variance approx)
        pooled_logvar   = (log_vars.exp().mean(0) / x_windows.shape[0]).log()
        return pooled_mean, pooled_logvar


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def mse_loss_normalized(pred_logits: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Standard MSE in normalised parameter space (after Sigmoid)."""
    return nn.functional.mse_loss(torch.sigmoid(pred_logits), target_norm)


def gaussian_nll_loss(
    mean_norm: torch.Tensor,
    log_var:   torch.Tensor,
    target_norm: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss for the probabilistic model."""
    var    = log_var.exp()
    nll    = 0.5 * (log_var + (target_norm - mean_norm) ** 2 / var)
    return nll.mean()
