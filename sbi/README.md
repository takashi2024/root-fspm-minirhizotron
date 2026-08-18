# TabICL Pipeline — Zero-Shot Parameter Inference for Root FSPM

This folder contains the TabICL-based parameter inference pipeline used as the ML
comparison method in the publication. Given synthetic minirhizotron observations
from the FSPM, **TabICL** (zero-shot tabular in-context learning) predicts the
8 root architectural parameters without any model training.

---

## Overview

```
Parameter prior (Latin Hypercube Sampling)
        ↓
Forward FSPM simulation  (Julia, src/)
        ↓
Summary statistics extraction
  RLD, presence probability, crossing-angle per tile
  averaged over K independent seed realizations
        ↓
TabICL zero-shot regression
  train = N=2000 synthetic samples (4 configs × 500)
  test  = 20 held-out test cases
        ↓
Parameter recovery evaluation vs. CMA-ES
```

No model training step — TabICL uses in-context learning (zero-shot at test time).

---

## Directory layout

```
sbi/
├── configs_v2/                    v2 environment configs (skin_depth=0.2 cm)
│   ├── Parameter_SBI_env_*_v2.json
│   └── sbi_config_*_v2.json
├── julia/
│   ├── PriorSampler.jl            Latin-Hypercube parameter sampler
│   ├── SimRunner.jl               FSPM day-loop + summary stats extraction
│   ├── FovRenderer.jl             FOV image renderer (used by publication Fig. 2)
│   ├── generate_dataset.jl        Generate base dataset (samples.csv + X_summary)
│   ├── generate_multiseed_dataset.jl  Generate K-seed summary tensors
│   └── extend_multiseed_dataset.jl    Extend existing K=7 dataset to K=20
├── python/
│   ├── utils.py                   ParamScaler (shared normalisation utility)
│   ├── combine_datasets.py        Merge 4 per-config datasets → combined tensors
│   ├── create_tensors_k14.py      Subsample K=14 from K=20 tensors
│   ├── evaluate_tabicl_multiseed_v2.py     Evaluate TabICL K=7  (publication)
│   ├── evaluate_tabicl_multiseed_v2_k1.py  Evaluate TabICL K=1  (ablation)
│   ├── evaluate_tabicl_multiseed_v2_k14.py Evaluate TabICL K=14 (ablation)
│   ├── analyze_seed_convergence.py         Relative error vs. K (figS1)
│   ├── analyze_seed_convergence_variance.py Variance-based convergence analysis
│   └── requirements.txt
├── run_multiseed_all_configs_v2.sh         Generate K=7 datasets for all 4 configs
├── run_extend_multiseed_all_configs_v2.sh  Extend to K=20
└── data/
    ├── combined/tensors/           Combined training tensors (N=2000, tracked)
    └── combined/test_case_eval_multiseed_v2*/  TabICL results (tracked)
```

---

## Reproduce from scratch

### Requirements

```bash
pip install -r sbi/python/requirements.txt
# Julia 1.10 with packages: JSON, Printf, NPZ, CSV, DataFrames, Statistics
```

### Step 0 — Generate base dataset (needed for samples.csv)

This step generates the prior parameter samples and runs single-seed FSPM simulations
to identify which samples have visible roots. The `samples.csv` output is read by the
multi-seed dataset generator. X_fov images produced here are not used for TabICL.

```bash
# Run from repo root — one per config (~4-8h each at N=500)
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_A_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_B2_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_C_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_D2_v2.json
```

### Step 1 — Generate K=7 multi-seed dataset (v2, skin_depth=0.2 cm)

```bash
bash sbi/run_multiseed_all_configs_v2.sh
# Output: sbi/data/multi_seed_v2/config_{A,B2,C,D2}/
# ~12h total (4 configs × ~3h, 6 threads each)
```

### Step 2 — Extend to K=20 (needed for K=14 ablation)

```bash
bash sbi/run_extend_multiseed_all_configs_v2.sh
# Adds seeds 8-20; output: tensors_k20/ in each config dir
```

### Step 3 — Combine datasets and create K=14 tensors

```bash
python3 sbi/python/combine_datasets.py        # → data/combined/tensors/
python3 sbi/python/create_tensors_k14.py      # → config_*/tensors_k14/
```

### Step 4 — Evaluate TabICL on 20 test cases

```bash
python3 sbi/python/evaluate_tabicl_multiseed_v2.py     # K=7 (main result)
python3 sbi/python/evaluate_tabicl_multiseed_v2_k1.py  # K=1 ablation
python3 sbi/python/evaluate_tabicl_multiseed_v2_k14.py # K=14 ablation
# Output CSVs: data/combined/test_case_eval_multiseed_v2*/results_*.csv
```

Pre-computed results from the publication are committed at:
`sbi/data/combined/test_case_eval_multiseed_v2*/`

---

## Data description

### Training data (N=2000 combined)

| File | Shape | Description |
|------|-------|-------------|
| `combined/tensors/layout_features.npy` | (2000, 8) | Per-sample planting geometry |
| `combined/tensors/y.npz` | (2000, 8) | True parameter values |
| `combined/tensors/X_summary.npz` | (2000, 11, 15, 4, 3) | Single-seed summary stats |

For multi-seed evaluation (K=7), data is loaded directly from
`multi_seed_v2/config_*/tensors/X_mean.npz` (per-config, not combined).

### Features used by TabICL

Each sample: flattened multi-seed mean (T×ns×na×3 = 1,980 features) + 8 layout
features (row spacing, plant density, tube position, etc.) + 4-dim config one-hot.
Total: 1,992 features.

### Test cases (N=20)

Defined in `calibration/test_cases_selected_v2.csv`.
Same 20 cases used for both CMA-ES and TabICL evaluation.

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

Two evaluation scenarios:
- **S1**: All 8 parameters unknown (TabICL predicts all 8)
- **S2**: Elongation parameters known (TabICL predicts 4 architectural parameters)
