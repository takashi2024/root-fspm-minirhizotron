# root-fspm-minirhizotron

Reproducible code for the paper:

> **Parameterization of a Functional-Structural Plant Model from Minirhizotron Observations: Comparing Process-Based Calibration and Simulation-Based Inference**

This repository covers the full pipeline from synthetic dataset generation through CMA-ES calibration, TabICL inference, structural evaluation, and publication figure generation.

---

## Repository structure

```
src/                     Julia FSPM source modules (core simulation engine)
sbi/
  configs/               Environment and pipeline configuration JSONs (4 configs: A, B, C, D)
  julia/                 Dataset generation scripts (FSPM simulation + FOV image rendering)
  python/                TabICL training and evaluation scripts
  run_multiseed_all_configs.sh   Generate multi-seed datasets for all configs
calibration/
  main_Optimization_mr_flexible.jl   CMA-ES calibration entry point
  objectives/            Loss functions and diagnostics
  configs/test_cases/    Per-test-case configuration JSONs (~140 files)
  *.sh                   Shell runners for S1/S2 calibration at k=1, 7, 14
  *.py                   Test case selection, observation extraction, aggregation
  evaluate_ml_structure_unified.jl   Structural evaluation (SS_rld, SS_angle)
publication/
  scripts/               Figure and table generation scripts
  figures/               Rendered publication figures (PNGs, LaTeX, Excel)
```

---

## Planting configurations

| Config | Plants | Row spacing | Tube position |
|--------|--------|-------------|---------------|
| A      | 32     | 12.5 cm     | 12.5 cm from nearest column |
| B      | 32     | 6.25 cm     | 12.5 cm from nearest column |
| C      | 40     | 8.0 cm      | 12.5 cm from nearest column |
| D      | 50     | 4.0 cm      | 12.5 cm from nearest column |

Configs B and D are repositioned versions that reduce zero-observation windows from ~70–75% to ~36–37% compared to the original tube placement at 25 cm.

---

## Reproducible pipeline

### Requirements

- **Julia 1.10.x** with packages pre-installed in the global environment (`~/.julia/environments/v1.10`): `CMAEvolutionStrategy`, `CairoMakie`, `CSV`, `DataFrames`, `JSON`, `Printf`, `Statistics`, `Random`
- **Python 3.10+**: `torch`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `openpyxl`, `tab-icl`

---

### Step 1 — Parameter generation (Latin Hypercube Sampling)

```bash
# Generates 500 parameter sets per config via LHS over the 8-parameter prior
# Called internally by generate_multiseed_dataset.jl — no separate invocation needed
# See: sbi/julia/PriorSampler.jl
```

500 × 4 configs = **2,000 parameter sets** total.

---

### Step 2 — FSPM simulation (k=7 observation seeds)

```bash
bash sbi/run_multiseed_all_configs.sh
```

Runs `sbi/julia/generate_multiseed_dataset.jl` for configs A, B, C, D. Each parameter set is simulated for 90 days and FOV images are rendered at 11 snapshot days (days 14–84 in 7-day steps) across 13 stations × 4 azimuths = 52 windows.

Output: `sbi/data/multi_seed/config_{A,B,C,D}/tensors/`

---

### Step 3 — Null observation filter

Applied during training and evaluation via the `--nonzero_only` flag. Removes parameter sets where all observation windows are zero (roots have not reached any station).

- 2,000 generated → **~1,270 pass** the filter
- After excluding 20 test cases: **1,198 training samples**

---

### Step 4 — Test case selection (N=20)

```bash
python3 calibration/select_test_cases.py
```

Selects 5 cases per config (20 total) using:
- `windows_with_roots ≥ 10` (sufficient observational signal)
- Stratification by `g_axial` quintile (Q1–Q5)
- Greedy maximin in 3D parameter space {branch_angle_order1, branching_prob_order1, branching_prob_order2}

Outputs: `calibration/test_cases_selected.csv`

---

### Step 5 — TabICL training

```bash
cd sbi/python
python3 train.py --nonzero_only
```

Trains the tabular in-context learning model on multi-seed averaged observations. Training set: 1,198 samples.

---

### Step 6 — CMA-ES calibration

Two scenarios × three k values = 6 conditions:

```bash
# S2 (elongation known, 4 architectural params), k=1
bash calibration/run_test_case_calibrations.sh

# S2, k=7
bash calibration/run_test_cases_s2ms.sh

# S2, k=14
bash calibration/run_test_cases_s2ms_k14.sh

# S1 (all 8 params unknown), k=1
# (S1 configs are in calibration/configs/test_cases/tc_*_s1.json)
# run_test_case_calibrations.sh covers both S1 and S2

# S1, k=7
bash calibration/run_test_cases_s1ms.sh

# S1, k=14
bash calibration/run_test_cases_s1ms_k14.sh
```

Each case takes ~2.8–8 hours on a single CPU core. Use the `_parallel.sh` variant to run multiple cases concurrently.

---

### Step 7 — Structural evaluation

```bash
# One mode at a time; run all 6 modes for complete results
julia calibration/evaluate_ml_structure_unified.jl s2_k1
julia calibration/evaluate_ml_structure_unified.jl s2_k7
julia calibration/evaluate_ml_structure_unified.jl s2_k14
julia calibration/evaluate_ml_structure_unified.jl s1_k1
julia calibration/evaluate_ml_structure_unified.jl s1_k7
julia calibration/evaluate_ml_structure_unified.jl s1_k14
```

Metrics computed: **SS_rld** (root length density skill score) and **SS_angle** (crossing angle skill score). Both follow the skill score convention SS = 1 − RMSE(pred)/RMSE(null), where SS > 0 means better than a mean-field predictor.

---

### Step 8 — Result aggregation

```bash
python3 calibration/summarize_test_case_results.py   # parameter recovery
python3 calibration/aggregate_ml_structure.py         # structural metrics
```

---

### Step 9 — Figure generation

```bash
# Julia pre-renders for Figure 2
julia publication/scripts/render_fig2a_makie.jl
julia publication/scripts/render_fov_images.jl

# Python figures
python3 publication/scripts/plot_fig1_configs_geometry.py
python3 publication/scripts/plot_fig2_root_minirhizotron.py
python3 publication/scripts/plot_fig3_parameter_recovery.py   # S1: all params unknown
python3 publication/scripts/plot_fig4_parameter_recovery.py   # S2: elongation known
python3 publication/scripts/plot_fig5_structural_k_effect.py
python3 publication/scripts/plot_figS1_seed_convergence.py
python3 publication/scripts/plot_figS2_ssangle_per_config.py
python3 publication/scripts/plot_figS3_equifinality.py
python3 publication/scripts/make_table1_structural.py
```

All figures are saved to `publication/figures/`.

---

## Key parameters (8 total)

| Parameter | Description | Scenario |
|-----------|-------------|----------|
| `elongation_rate_min` | Minimum daily elongation (cm/day) | S1 only |
| `elongation_rate_max` | Maximum daily elongation (cm/day) | S1 only |
| `elongation_logistic_k` | Logistic growth rate constant | S1 only |
| `elongation_logistic_t50` | Day of half-maximum elongation | S1 only |
| `branch_angle_order1` | Mean lateral root emergence angle (°) | S1 & S2 |
| `g_axial` | Gravitropism strength of axial roots | S1 & S2 |
| `branching_prob_order1` | Probability of first-order lateral initiation | S1 & S2 |
| `branching_prob_order2` | Probability of second-order lateral initiation | S1 & S2 |

S2 scenario fixes elongation parameters at true values; only the 4 architectural parameters are estimated.

---

## Citation

If you use this code, please cite the paper (reference to be added upon publication).

---

## License

MIT
