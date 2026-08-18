"""combine_datasets.py — Merge 4 per-config datasets into a single N=2000 dataset.

Training set: A, B2, C, D2 — all have tube 12.5 cm from nearest plant column,
giving ~64% visible-root samples. B and D (tube 25 cm from column, ~25-30% visible)
are excluded from training to avoid diluting the dataset with mostly-zero observations.

Usage (from repo root):
    python sbi/python/combine_datasets.py

Outputs (in sbi/data/combined/tensors/):
    X_summary.npz       — (2000, 11, 15, 4, 3) float32
    y.npz               — (2000, 8) float32
    param_names.txt
    config_ids.npy      — (2000,) int32  0=A, 1=B2, 2=C, 3=D2
    layout_features.npy — (2000, 8) float32  physical layout covariates (see LAYOUT below)
    layout_feature_names.txt
"""

from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent

CONFIGS = [
    ("A",  "sbi/data/config_A"),
    ("B2", "sbi/data/config_B2"),
    ("C",  "sbi/data/config_C"),
    ("D2", "sbi/data/config_D2"),
]

# Physical layout parameters for each config.
# These are the covariates that govern what the tube observes — the model needs
# them to correctly decode X_summary into root parameters.
#
# Columns (all in cm or plants/m² as indicated):
#   row_spacing_cm       — distance between plant rows (x-direction)
#   within_row_cm        — within-row plant spacing (y-direction, tube-parallel)
#   density_per_m2       — plant density
#   box_depth_cm         — rhizobox depth (D, y-extent)
#   tube_x_cm            — tube x position (from left, cm)
#   tube_to_nearest_row  — |tube_x - nearest column x| (cm)
#   n_rows               — number of plant rows (x-direction)
#   n_cols_per_row       — plants per row (y-direction)
LAYOUT = {
    #            row_x  within_y  dens  depth  tube_x  dist_row  n_rows  n_per_row
    "A":  np.array([25.0,  12.5,   32.0, 100.0, 50.0,   12.5,     4.0,    8.0], dtype=np.float32),
    "B2": np.array([50.0,   6.25,  32.0, 100.0, 37.5,   12.5,     2.0,   16.0], dtype=np.float32),
    "C":  np.array([25.0,   8.0,   50.0,  80.0, 50.0,   12.5,     4.0,   10.0], dtype=np.float32),
    "D2": np.array([50.0,   4.0,   50.0, 100.0, 37.5,   12.5,     2.0,   25.0], dtype=np.float32),
}

LAYOUT_FEATURE_NAMES = [
    "row_spacing_cm",
    "within_row_spacing_cm",
    "density_per_m2",
    "box_depth_cm",
    "tube_x_cm",
    "tube_to_nearest_row_cm",
    "n_rows",
    "n_plants_per_row",
]

OUT_DIR = REPO_ROOT / "sbi/data/combined/tensors"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X_list, y_list, ids_list, layout_list = [], [], [], []
    param_names = None

    for config_id, (label, rel_path) in enumerate(CONFIGS):
        tensors_dir = REPO_ROOT / rel_path / "tensors"
        X = np.load(tensors_dir / "X_summary.npz")["X"]   # (500, 11, 15, 4, 3)
        y = np.load(tensors_dir / "y.npz")["y"]           # (500, 8)
        N = len(X)

        if param_names is None:
            with open(tensors_dir / "param_names.txt") as f:
                param_names = [l.strip() for l in f if l.strip()]

        layout_row = np.tile(LAYOUT[label], (N, 1))        # (500, 8)

        print(f"Config {label}: X_summary={X.shape}  y={y.shape}  layout={layout_row.shape}")
        X_list.append(X)
        y_list.append(y)
        ids_list.append(np.full(N, config_id, dtype=np.int32))
        layout_list.append(layout_row)

    X_all      = np.concatenate(X_list,      axis=0).astype(np.float32)  # (2000, 11, 15, 4, 3)
    y_all      = np.concatenate(y_list,      axis=0).astype(np.float32)  # (2000, 8)
    ids_all    = np.concatenate(ids_list,    axis=0)                      # (2000,)
    layout_all = np.concatenate(layout_list, axis=0).astype(np.float32)  # (2000, 8)

    np.savez_compressed(str(OUT_DIR / "X_summary.npz"), X=X_all)
    np.savez_compressed(str(OUT_DIR / "y.npz"),         y=y_all)
    np.save(str(OUT_DIR / "config_ids.npy"),            ids_all)
    np.save(str(OUT_DIR / "layout_features.npy"),       layout_all)
    with open(OUT_DIR / "param_names.txt", "w") as f:
        f.write("\n".join(param_names) + "\n")
    with open(OUT_DIR / "layout_feature_names.txt", "w") as f:
        f.write("\n".join(LAYOUT_FEATURE_NAMES) + "\n")

    print(f"\nCombined: X_summary={X_all.shape}  y={y_all.shape}")
    print(f"Layout features ({layout_all.shape[1]}): {LAYOUT_FEATURE_NAMES}")
    print(f"Config counts: { {l: int((ids_all == i).sum()) for i, (l, _) in enumerate(CONFIGS)} }")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
