# Convergence Analysis Report

**Date:** 2026-06-17  
**Purpose:** Determine the minimum number of simulation seed replicates (K) and physical tube positions (M) needed for stable calibration and evaluation metrics.

---

## 1. Seed Convergence (K)

**Question:** How many seed replicates must be averaged before R² stabilises?  
**Method:** Run N=30 seeds; compute running-average R²(k) for k=1..30; find k where 3-point rolling σ(R²) < 0.01.

### Setup

Analysis scripts: `calibration/analyse_seed_convergence.jl`  
Two modes were used:

- **Mode B (pre-calibration):** Three bracketing parameter files (low/template/high branching) were run against template observations to bound K across the likely parameter space.
- **Mode A (post-calibration):** Best-fit parameters from the `replicate_seed_average` experiment were used on the actual 3-tube mock observations.

All analyses used the 200 cm box with 1-tube configuration at x=100 cm.

### Results

| Scenario | Parameters | R²(k=1) | R²(k=12) | R²(k=30) | Conv. k (σ<0.01) |
|---|---|---|---|---|---|
| Low branch | branching_prob=0.10 | −0.172 | −0.139 | −0.145 | k≈6 |
| Template | branching_prob=0.20 | +0.241 | +0.147 | +0.186 | k≈8 |
| High branch | branching_prob=0.80 | −0.938 | −0.532 | −0.508 | k≈7 |
| Calibrated (Mode A) | best_params, 3-tube | +0.415 | +0.676 | +0.657 | k≈6 |

> The convergence k values above are where 3-point rolling σ(R²) first drops below 0.01. The R² curves for the pre-calibration scenarios (low/template/high) show continued oscillation beyond these k values; in practice, the recommended K is higher for CMA-ES stability.

### R²(k) curve — calibrated scenario (Mode A)

| k | R²(all) | R²(nz) |
|---|---|---|
| 1 | +0.415 | −0.312 |
| 4 ← was used | +0.622 | +0.144 |
| 6 | +0.638 | +0.179 |
| 9 | +0.689 | +0.294 |
| 12 | +0.676 | +0.264 |
| 15 | +0.671 | +0.250 |
| 20 | +0.665 | +0.234 |
| 30 | +0.657 | +0.216 |

The calibrated scenario stabilises around k=9 and fluctuates ≤0.03 thereafter.

### Interpretation

- The pre-calibration bracketing scenarios show that even with template parameters, R²(k) does not fully stabilise within 30 seeds. This reflects the stochastic butterfly effect: even small differences between simulated and observed parameter sets cause divergent root architectures, making individual-seed R² very noisy.
- The calibrated scenario converges faster (best-fit parameters reduce mismatch, so seed-to-seed variance in R² shrinks). First stable region appears at k≈9.
- k=4 (the previous `calibration_seeds` count) gives R²≈0.62, while k=9–12 gives R²≈0.67–0.69. The difference is ~0.05 R² and corresponds to CMA-ES optimising against a significantly noisier loss.

### Recommendation: **K = 12**

- Rounds up beyond the k≈9 convergence point of the calibrated scenario with a safety margin.
- Conservative relative to the pre-calibration bracketing (template k≈8, but these don't fully stabilise).
- All 16 `calib_200cm_*.json` configs have been updated to `calibration_seeds = [11, 22, 99, 88, 33, 55, 77, 7, 13, 42, 66, 91]`.

---

## 2. Tube Convergence (M)

**Question:** How many evenly-spaced physical tube positions are needed for stable spatial averaging?  
**Method:** Simulate 15 tubes evenly spaced from x=25 cm to x=175 cm (200 cm box); compare running-average of m=1..15 tubes against template observations.

### Setup

Analysis script: `calibration/analyse_tube_convergence.jl`  
Experiment: `replicate_seed_average_20260617_172848`, seed=42  
Observations: template VD parameters (ground truth proxy)  
Predictions: best_params from the `replicate_seed_average` calibration  
Tube positions: 15 tubes at x=25, 35.7, 46.4, ..., 175.0 cm (spacing ≈10.7 cm)

### Results

| m (tubes) | x range (cm) | R²(all) | R²(nz) | RMSE |
|---|---|---|---|---|
| 1 | 25.0 | −0.357 | −0.839 | 2.046 |
| 2 | 25–35.7 | −0.006 | −0.480 | 1.437 |
| 3 | 25–46.4 | +0.397 | −0.380 | 1.036 |
| 4 | 25–57.1 | +0.571 | −0.040 | 0.952 |
| 5 | 25–67.9 | +0.583 | +0.086 | 0.896 |
| 6 | 25–78.6 | +0.622 | +0.159 | 0.868 |
| **7** | **25–89.3** | **+0.655** | **+0.190** | **0.814** |
| 8 | 25–100.0 | +0.692 | +0.247 | 0.802 |
| 9 | 25–110.7 | +0.712 | +0.265 | 0.766 |
| 10 | 25–121.4 | +0.721 | +0.300 | 0.774 |
| 11 | 25–132.1 | +0.750 | +0.373 | 0.751 |
| 12 | 25–142.9 | +0.737 | +0.337 | 0.749 |
| 13 | 25–153.6 | +0.763 | +0.380 | 0.720 |
| 14 | 25–164.3 | +0.763 | +0.455 | 0.705 |
| 15 | 25–175.0 | +0.743 | +0.412 | 0.707 |

### Interpretation

- **m=1 (single tube):** R²=−0.357. A single tube is dominated by positional variance — unreliable.
- **m=3 (current 3-tube setup):** Even the best-case 3-tube coverage (25–46 cm, all near one edge) reaches only R²=0.40. The actual field 3-tube setup (x=75, 100, 125 cm, all near centre) provides limited spatial coverage and will have similar or worse spatial averaging.
- **Elbow at m≈7–8:** The gain per additional tube drops substantially after m=7 (R²=0.655 → 0.692 at m=8, vs. gains of 0.17–0.40 per tube for m=1–4).
- **Not fully converged at m=15:** R² still oscillates (0.74–0.76) at the upper end. True convergence likely requires m≥20 tubes for σ(R²)<0.01.
- **R²(nz) lags far behind:** Non-zero cell R² only reaches +0.41 at m=15. Non-zero cell prediction (intensity given presence) is harder to calibrate spatially and needs more tubes.

### Note on current 3-tube positions

The actual field setup uses x=75, 100, 125 cm (three tubes clustered near the box centre). These all sample a similar mid-box region, so spatial diversity is lower than m=3 evenly-spaced tubes. The effective spatial averaging is likely closer to m=2 in this analysis.

### Recommendation: **M ≥ 7 physical tubes**

- Evenly distributed across the plant zone (e.g. x=25, 50, 75, 100, 125, 150, 175 cm in a 200 cm box).
- Minimum to reach R²≥0.65 and enter a region of diminishing returns.
- Combined with K=12 seeds, this provides a well-constrained calibration target.

---

## 3. Summary and Action Items

| Parameter | Current | Recommended | Evidence |
|---|---|---|---|
| Seed replicates (K) | 4 | **12** | R² stabilises at k≈9; margin to k=12 |
| Physical tubes (M) | 3 (x=75,100,125 cm) | **≥7** (evenly spaced) | Elbow at m=7, R²=0.65; m=3 gives R²≈0.12–0.40 |

**Completed:**
- All 16 `calib_200cm_*.json` configs updated to K=12 seeds.

**Pending:**
- Re-run all 16 calibrations with K=12 for reliable CMA-ES convergence.
- Plan physical tube placement at ≥7 evenly-spaced positions before next field season.
- Consider extending tube convergence analysis to N=20–25 to confirm true convergence point.

---

## 4. File Locations

| Output | Path |
|---|---|
| Seed conv. metrics (low branch) | `calibration/seed_convergence_pretrial/low_branch/convergence_metrics.csv` |
| Seed conv. metrics (template) | `calibration/seed_convergence_pretrial/template/convergence_metrics.csv` |
| Seed conv. metrics (high branch) | `calibration/seed_convergence_pretrial/high_branch/convergence_metrics.csv` |
| Seed conv. metrics (calibrated) | `calibration/experiments/replicate_seed_average_20260617_172848/seed_convergence/convergence_metrics.csv` |
| Tube conv. metrics | `calibration/experiments/replicate_seed_average_20260617_172848/tube_convergence/convergence_metrics.csv` |
| Seed conv. plots | `**/seed_convergence/convergence_r2.png`, `convergence_r2_stability.png` |
| Tube conv. plots | `**/tube_convergence/convergence_r2.png`, `convergence_r2_stability.png` |
| Analysis scripts | `calibration/analyse_seed_convergence.jl`, `calibration/analyse_tube_convergence.jl` |
| Bracketing param files | `calibration/Parameter_VD_{low_branch,template,high_branch}.json` |
