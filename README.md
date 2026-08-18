# root-fspm-minirhizotron

Reproducible code for the paper:

> **Parameterization of a Functional-Structural Plant Model from Minirhizotron Observations: Comparing Process-Based Calibration and Simulation-Based Inference**

This repository covers the full pipeline from synthetic dataset generation through CMA-ES calibration, TabICL inference, structural evaluation, and publication figure generation.

All results use **v2 observations** (skin depth = 2 mm), which represent root segments visible through the 2 mm rhizobox wall — the biologically correct setting for minirhizotron tubes.

---

## Repository structure

```
src/                     Julia FSPM source modules (core simulation engine)
sbi/
  configs_v2/            Environment and pipeline configs (skin_depth=0.2 cm; 4 configs: A, B2, C, D2)
  julia/                 Dataset generation scripts (FSPM simulation + summary stats extraction)
  python/                TabICL evaluation scripts (zero-shot, no training)
  run_multiseed_all_configs_v2.sh    Generate K=7 multi-seed datasets for all configs
  run_extend_multiseed_all_configs_v2.sh  Extend to K=20 seeds
calibration/
  main_Optimization_mr_flexible.jl   CMA-ES calibration entry point
  objectives/            Loss functions (mr_objectives.jl) and diagnostics (mr_diagnostics.jl)
  configs/test_cases_v2/ Per-test-case configuration JSONs (120 files, 20 cases × 6 modes)
  data_v2/test_cases/    Observed summary statistics for the 20 test cases (60 CSVs)
  *.sh                   Shell runners for S1/S2 calibration at k=1, 7, 14
  *.py                   Observation extraction, config generation, aggregation
  evaluate_ml_structure_unified.jl   Structural evaluation (SS_rld, SS_angle)
publication/
  scripts/               Figure and table generation scripts
  figures/               Rendered publication figures (PNGs, LaTeX, Excel)
```

---

## Planting configurations

| Config | Plants | Row spacing | Notes |
|--------|--------|-------------|-------|
| A      | 32     | 12.5 cm     | Baseline |
| B2     | 32     | 6.25 cm     | Tube repositioned to reduce zero-obs windows |
| C      | 40     | 8.0 cm      | Higher plant density |
| D2     | 50     | 4.0 cm      | Highest density, repositioned tube |

Configs B2 and D2 reposition the observation tube from 25 cm to 12.5 cm from the nearest plant column, reducing zero-observation windows from ~70–75% to ~36–37%.

---

## Reproducible pipeline

### Requirements

- **Julia 1.10.x** with packages: `CMAEvolutionStrategy`, `CairoMakie`, `CSV`, `DataFrames`, `JSON`, `Printf`, `Statistics`, `Random`, `NPZ`
- **Python 3.10+**: `tabicl`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`

```bash
pip install -r sbi/python/requirements.txt
```

Pre-computed results are committed — you can reproduce figures directly from Step 6 without re-running calibration or TabICL.

---

### Step 1 — Generate base dataset (prerequisite for multi-seed generation)

Runs the prior sampler and single-seed FSPM simulation per config to produce `samples.csv`, which the multi-seed generator reads to know which parameter sets yield visible roots.

```bash
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_A_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_B2_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_C_v2.json
julia sbi/julia/generate_dataset.jl sbi/configs_v2/sbi_config_D2_v2.json
```

500 parameter sets per config (2,000 total) are drawn via Latin Hypercube Sampling from the 8-parameter prior.

---

### Step 2 — Generate K=7 multi-seed dataset (v2, skin_depth=0.2 cm)

```bash
bash sbi/run_multiseed_all_configs_v2.sh
```

Simulates each parameter set for 7 independent random seeds and extracts summary statistics (root length density, presence probability, crossing-angle histogram) at 11 time points × 13 stations × 4 azimuths. No images are generated.

Output: `sbi/data/multi_seed_v2/config_{A,B2,C,D2}/tensors/`

---

### Step 3 — Extend to K=20 seeds (needed for K=14 ablation)

```bash
bash sbi/run_extend_multiseed_all_configs_v2.sh
```

Adds seeds 8–20 to the existing K=7 dataset.
Output: `sbi/data/multi_seed_v2/config_*/tensors_k14/` (created by `python3 sbi/python/create_tensors_k14.py`).

---

### Step 4 — Combine datasets and create K=14 tensors

```bash
python3 sbi/python/combine_datasets.py        # → sbi/data/combined/tensors/
python3 sbi/python/create_tensors_k14.py      # → config_*/tensors_k14/
```

Filters out zero-observation samples (all windows empty). Approximately 1,270 of 2,000 samples pass the filter.

---

### Step 5 — TabICL evaluation (zero-shot, no training)

TabICL uses in-context learning: the entire training set is passed as context at test time. No model training is required.

```bash
python3 sbi/python/evaluate_tabicl_multiseed_v2.py     # K=7  (main result)
python3 sbi/python/evaluate_tabicl_multiseed_v2_k1.py  # K=1  (ablation)
python3 sbi/python/evaluate_tabicl_multiseed_v2_k14.py # K=14 (ablation)
```

Output CSVs: `sbi/data/combined/test_case_eval_multiseed_v2*/results_*.csv`
Pre-computed results are committed; re-running will overwrite them.

---

### Step 6 — CMA-ES calibration (20 test cases × 6 modes)

Two scenarios (S1: all 8 params; S2: 4 architectural params with elongation known) × three k values = 6 modes.

```bash
bash calibration/run_test_cases_v2_s2.sh      # S2, k=1
bash calibration/run_test_cases_v2_s2ms.sh    # S2, k=7
bash calibration/run_test_cases_v2_s2ms_k14.sh  # S2, k=14
bash calibration/run_test_cases_v2_s1.sh      # S1, k=1
bash calibration/run_test_cases_v2_s1ms.sh    # S1, k=7
bash calibration/run_test_cases_v2_s1ms_k14.sh  # S1, k=14
```

Each test case takes 2–8 hours on a single CPU core. CMA-ES configuration JSONs are in `calibration/configs/test_cases_v2/`.

> **Note for users on a different machine**: the committed config JSONs contain absolute paths specific to the original machine. Before running CMA-ES, regenerate them with your local paths:
> ```bash
> python3 calibration/generate_test_case_configs_k1_k14_v2.py
> python3 calibration/generate_test_case_configs_s2ms_v2.py
> python3 calibration/generate_test_case_configs_s1ms_v2.py
> ```
> This overwrites the 120 config JSONs with paths correct for your machine.

---

### Step 7 — Structural evaluation

```bash
julia calibration/evaluate_ml_structure_unified.jl s2_k1
julia calibration/evaluate_ml_structure_unified.jl s2_k7
julia calibration/evaluate_ml_structure_unified.jl s2_k14
julia calibration/evaluate_ml_structure_unified.jl s1_k1
julia calibration/evaluate_ml_structure_unified.jl s1_k7
julia calibration/evaluate_ml_structure_unified.jl s1_k14
```

Metrics: **SS_rld** (root length density skill score) and **SS_angle** (crossing angle skill score). Skill score SS = 1 − RMSE(pred)/RMSE(null); SS > 0 means better than a mean-field null predictor.

Pre-computed results are committed as `calibration/comparison_structure_metrics_v2_*.csv`.

---

### Step 8 — Result aggregation

```bash
python3 calibration/summarize_test_case_results_v2.py   # parameter recovery
python3 calibration/aggregate_ml_structure_s2ms_v2.py   # structural metrics (S2)
python3 calibration/aggregate_ml_structure_s1ms_v2.py   # structural metrics (S1)
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
