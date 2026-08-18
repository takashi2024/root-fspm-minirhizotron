"""infer.py — CNN inference on real minirhizotron observations → warm-start JSON for CMA-ES.

Converts a real observation CSV (same format as calibration/data/mr_obs_stations_*.csv)
into the per-window image tensor expected by RootCNN, runs inference, mean-pools over
all windows, and writes a warm-start species parameter JSON compatible with
calibration/Parameter_VD_template.json.

Usage (from repo root):
    python sbi/python/infer.py \\
        --obs_csv calibration/data/mr_obs_stations_1tube.csv \\
        --config  sbi/configs/sbi_config.json \\
        --output  sbi/data/warmstart_VD.json

The output JSON can be passed as the starting point in main_Optimization_mr_flexible.jl
by setting the warm-start parameters directly in the VD template file or patching the
optimizer script's PARAM_DEFAULTS dict.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import RootCNN
from utils import ParamScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs_csv",    required=True)
    p.add_argument("--config",     default="sbi/configs/sbi_config.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output",     default="sbi/data/warmstart_VD.json")
    p.add_argument("--template",   default=None,
                   help="Path to VD template JSON to embed defaults for untuned parameters")
    return p.parse_args()


def obs_csv_to_window_tensor(
    obs_csv: str,
    snapshot_days: list[int],
    station_positions_cm: list[float],
    camera_azimuths_deg: list[float],
    sowing_date: str = "2025-04-01",
    img_height: int = 64,
    img_width: int  = 64,
) -> np.ndarray:
    """
    Convert a real observation CSV to a per-window image tensor.

    The real CSV contains root_length_cm per (date, station_idx, azimuth_deg).
    We convert root length to a presence-based binary proxy:
        root_length_cm > 0 → 1.0 (root present in window)
        root_length_cm == 0 → 0.0

    Returns:
        X: (n_windows, T, H, W) float32 with values in {0, 1}
            n_windows = ns × na
            T = len(snapshot_days)
    """
    import datetime
    sow = datetime.date.fromisoformat(sowing_date)

    df = pd.read_csv(obs_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sim_day"] = df["date"].apply(lambda d: (d - sow).days + 1)

    ns = len(station_positions_cm)
    na = len(camera_azimuths_deg)
    T  = len(snapshot_days)

    X = np.zeros((ns, na, T, img_height, img_width), dtype=np.float32)

    for t_idx, snap_day in enumerate(snapshot_days):
        sub = df[df["sim_day"] == snap_day]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            si = int(row["station_idx"]) - 1   # 0-indexed
            az = float(row["azimuth_deg"])
            # Find closest azimuth
            ai = int(np.argmin([abs(az - a) for a in camera_azimuths_deg]))
            if 0 <= si < ns and row["root_length_cm"] > 0:
                # Fill the entire image with 1.0 as a presence indicator
                X[si, ai, t_idx, :, :] = 1.0

    # Reshape to (n_windows, T, H, W) — station outer, azimuth inner
    X_win = X.reshape(ns * na, T, img_height, img_width)
    return X_win


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.parent.parent

    with open(repo_root / args.config) as f:
        cfg = json.load(f)

    ckpt_dir  = repo_root / cfg["outdir"] / "checkpoints"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "model_best.pt"
    scaler    = ParamScaler.load(str(ckpt_dir / "scaler_params.json"))
    param_names = scaler.param_names

    snapshot_days        = cfg["snapshot_days"]
    station_positions_cm = None  # loaded from base env
    camera_azimuths_deg  = None

    base_env_path = repo_root / cfg["base_env_json"]
    with open(base_env_path) as f:
        base_env = json.load(f)
    station_positions_cm = base_env["minirhizotron"]["station_positions_cm"]
    camera_azimuths_deg  = base_env["minirhizotron"]["camera_azimuths_deg"]

    img_height = cfg["img_height"]
    img_width  = cfg["img_width"]

    # ── Build window tensor from obs CSV ──────────────────────────────────────
    X_win = obs_csv_to_window_tensor(
        args.obs_csv, snapshot_days,
        station_positions_cm, camera_azimuths_deg,
        img_height=img_height, img_width=img_width,
    )
    print(f"Observation tensor: {X_win.shape}  ({X_win.sum():.0f} non-zero pixels)")

    # ── Run CNN inference ─────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = RootCNN(n_params=len(param_names), in_channels=len(snapshot_days)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(X_win).to(device)
        pred_norm = model.pool_windows(x_t).cpu().numpy()  # (n_params,)

    # Denormalise to raw parameter values
    pred_raw = scaler.denormalize(pred_norm[np.newaxis, :])[0]

    print("\nCNN parameter estimates:")
    for name, val in zip(param_names, pred_raw):
        print(f"  {name:30s} = {val:.6f}")

    # ── Build warm-start JSON ─────────────────────────────────────────────────
    template_path = (Path(args.template) if args.template else
                     repo_root / cfg["vd_template_json"])
    with open(template_path) as f:
        warmstart = json.load(f)

    # Overwrite estimated parameters
    for name, val in zip(param_names, pred_raw):
        warmstart[name] = float(val)
    warmstart["root_elongation_mode"] = "logistic"
    warmstart["_sbi_source"] = str(ckpt_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(warmstart, f, indent=2)
    print(f"\nWarm-start JSON written to: {out_path}")
    print("Use this file as the species template in main_Optimization_mr_flexible.jl")


if __name__ == "__main__":
    main()
