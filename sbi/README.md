# SBI Pipeline — Simulation-Based Inference for Root FSPM Parameters

Amortised inverse model: train a CNN on synthetic minirhizotron images to predict
hidden FSPM root architectural parameters, then use the CNN output as a warm-start
for the existing CMA-ES calibration.

**The FSPM and CMA-ES calibration are unchanged.** This pipeline adds a fast screening
step on top.

---

## Overview

```
Parameter prior (LHS)
        ↓
Forward FSPM simulation (Julia)
        ↓
B&W 5×5 cm FOV image rendering  ← new: FovRenderer.jl
        ↓
CNN training (PyTorch)
        ↓
Predict parameters from real observations
        ↓
Warm-start CMA-ES calibration
```

---

## Directory layout

```
sbi/
├── configs/sbi_config.json        all pipeline settings
├── julia/
│   ├── FovRenderer.jl             B&W FOV image renderer
│   ├── PriorSampler.jl            Latin-Hypercube parameter sampler
│   ├── SimRunner.jl               manual day-loop simulation + image extraction
│   └── generate_dataset.jl        CLI entry point (Julia)
├── python/
│   ├── dataset.py                 PyTorch Dataset
│   ├── model.py                   RootCNN + RootCNNProbabilistic
│   ├── train.py                   training loop
│   ├── evaluate.py                RMSE / R² / scatter plots
│   ├── infer.py                   obs CSV → warm-start JSON
│   └── utils.py                   ParamScaler, NoiseAugmentation
└── data/                          generated at runtime (gitignore this)
    ├── samples.csv
    ├── images/
    └── tensors/
```

---

## Step 1 — Generate synthetic training data (Julia)

```bash
# From repo root
julia sbi/julia/generate_dataset.jl sbi/configs/sbi_config.json
```

This will:
1. Sample 500 parameter sets via Latin Hypercube Sampling
2. Run 90-day FSPM simulations for each
3. Render 64×64 px B&W FOV images at 11 snapshot days (14–84, every 7 days)
4. Save PNG images + NPZ tensors under `sbi/data/`

**Estimated time:** ~4–8 hours on a single core for N=500.
For a quick smoke-test, edit `sbi/configs/sbi_config.json` → `"n_samples": 5`.

**Outputs:**
| File | Shape | Description |
|------|-------|-------------|
| `data/samples.csv` | (500, 12) | Sampled parameter values + viability flag |
| `data/images/*.png` | 64×64 uint8 | B&W FOV images |
| `data/tensors/X_fov.npz` | (30000, 11, 64, 64) uint8 | Per-window input tensor |
| `data/tensors/X_summary.npz` | (500, 11, 15, 4, 3) float32 | RLD / presence / angle maps |
| `data/tensors/y.npz` | (500, 8) float32 | True parameter values |
| `data/tensors/param_names.txt` | — | One parameter name per line |

---

## Step 2 — Install Python requirements

```bash
pip install -r sbi/python/requirements.txt
```

---

## Step 3 — Train the CNN (Python)

```bash
python sbi/python/train.py --config sbi/configs/sbi_config.json
```

Options:
```
--epochs N      Override number of training epochs (default: 100)
--batch  N      Override batch size (default: 64)
--lr     F      Override learning rate (default: 0.001)
--no-aug        Disable noise augmentation
```

**Outputs** (in `sbi/data/checkpoints/`):
- `model_best.pt` — best validation checkpoint
- `model_final.pt` — final epoch weights
- `scaler_params.json` — parameter normalisation bounds
- `train_log.csv` — per-epoch loss history

---

## Step 4 — Evaluate parameter recovery

```bash
python sbi/python/evaluate.py --config sbi/configs/sbi_config.json --split test
```

**Outputs** (in `sbi/data/checkpoints/`):
- `eval_metrics.csv` — RMSE, MAE, R² per parameter
- `scatter_test.png` — predicted vs true scatter plots

Acceptable baseline with N=500: R² > 0.4 for elongation rates and branching probabilities.
Increase N (2000+) for production-quality recovery.

---

## Step 5 — Predict parameters from real observations

```bash
python sbi/python/infer.py \
    --obs_csv calibration/data/mr_obs_stations_1tube.csv \
    --config  sbi/configs/sbi_config.json \
    --output  sbi/data/warmstart_VD.json
```

The output `warmstart_VD.json` is a species parameter file in the same format as
`calibration/Parameter_VD_template.json`.

---

## Step 6 — Use CNN output as CMA-ES warm-start

Copy `warmstart_VD.json` to the calibration species template path and run the
existing calibration script:

```bash
cp sbi/data/warmstart_VD.json calibration/Parameter_VD_warmstart.json

# Edit calibration/main_Optimization_mr_flexible.jl:
# Change: vd_template_path = "calibration/Parameter_VD_template.json"
# To:     vd_template_path = "calibration/Parameter_VD_warmstart.json"

julia calibration/main_Optimization_mr_flexible.jl calibration/configs/calib_rmse_baseline.json
```

The CMA-ES will start from the CNN-predicted parameters rather than the default template,
typically converging faster when the CNN R² is > 0.5 for the key parameters.

---

## Image coordinate system

Each 5×5 cm FOV image corresponds to one (station, azimuth) observation window:

```
Image width  (columns): along-tube axis
  column 1 = station_pos − 2.5 cm
  column 64 = station_pos + 2.5 cm

Image height (rows): arc offset from azimuth centre
  row 1  = −2.5 cm (image_arc_cm/2 below centre)
  row 64 = +2.5 cm (above centre)
```

The projection uses exactly the same geometry as `tube_station_azimuth_lengths_cm`
in `src/MetricsAndOutput.jl`: a root segment that contributes to the root-length
metric for window (si, ai) will appear as a white pixel in the corresponding FOV image.

---

## Parameter bounds and prior distributions

| Parameter | Prior | Range |
|-----------|-------|-------|
| `elongation_rate_min` | log-uniform | 0.001 – 1.0 cm/day |
| `elongation_rate_max` | log-uniform | 0.001 – 1.5 cm/day (≥ e_min) |
| `elongation_logistic_k` | log-uniform | 0.05 – 0.4 |
| `elongation_logistic_t50` | uniform | 5 – 50 days |
| `branch_angle_order1` | uniform | 40 – 85° |
| `g_axial` | log-uniform | 0.01 – 0.50 |
| `branching_prob_order1` | uniform | 0.05 – 1.0 |
| `branching_prob_order2` | uniform | 0.0 – 1.0 |

---

## Notes and known limitations

- **~5–15% of parameter combinations produce zero visible roots** in the minirhizotron.
  These samples are kept in the dataset (labelled `has_visible_roots=false` in `samples.csv`)
  and train the CNN to recognise "no roots → slow/weak growth" parameter combinations.
- The CNN predicts per-window, then mean-pools across all 60 (station × azimuth) windows.
  Unobserved windows (roots not yet reached) contribute null images that drive predictions
  toward slow-growth parameters — this is the correct Bayesian behaviour.
- The `infer.py` obs-to-tensor conversion uses a simple root-presence proxy (length > 0 → 1)
  rather than actual image rendering, since real minirhizotron images are not available
  in this pipeline. Extend `obs_csv_to_window_tensor` if real scanned images are added.
