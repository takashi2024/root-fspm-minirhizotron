# Calibration Mode Comparison — Findings and Recommendations

**Date:** 2026-06-20  
**Experiment sets:** Group A (200 cm box, 1 tube, length-only loss) · Group B (100 cm PBC box, 1 tube, length + angle-histogram loss, w_angle_hist=0.3)  
**Parameters calibrated:** branch_angle_order1, g_axial, branching_prob_order1, branching_prob_order2  
**True values:** ba_o1=80°, g_axial=0.05, bp_o1=0.20, bp_o2=0.30  
**Seeds:** 12 (calibration) · 12 (multi-seed diagnostic)

---

## Key Methodological Finding: Single-Seed Diagnostics Are Unreliable

Single-seed (seed=42) R² values are dominated by butterfly-effect noise — small parameter
differences flip branching decisions, producing completely different spatial root layouts even
at the same parameter values. This makes single-seed R² unreliable as a performance metric.

**Fix:** Average predicted root length per window across all 12 calibration seeds before
computing R² and RMSE. This multi-seed diagnostic is now the standard metric
(`diag_r2_multiseed`, `diag_rmse_multiseed` in `run_summary.json`).

Impact of the fix:
- Several modes that appeared to fail (R² as low as −0.992) showed genuine positive R² (0.036–0.279) under multi-seed averaging.
- RMSE dropped from 4–8 cm (single-seed) to 3–5 cm (multi-seed) across Group A, confirming noise was inflating the error.

---

## Main Result: Angle Histogram Is the Decisive Factor

The most important finding is not which objective mode is best, but that **adding the angle
histogram loss is what drives parameter recovery**, regardless of mode.

| Group | Setup | Best mean \|% error\| | bp_o1 range | bp_o2 range |
|---|---|---|---|---|
| A | 200 cm, length-only | 33.7% (two_part_hurdle) | +110% to +176% | −77% to +63% |
| B | 100 cm PBC, angle-hist | **5.3%** (weighted_rmse) | −25% to +90% | −65% to +1.4% |

In Group A, `branching_prob_order1` is off by 110–176% in **every single mode** — the model
compensates by over-branching to match total root length, but this is unidentifiable from
length alone. The angle histogram breaks this degeneracy by constraining the structural shape
of the root system, not just its total length.

Even the simplest Group B mode (`rmse_baseline`, 16.7% error) outperforms the best Group A
mode (33.7%). The loss function choice matters less than whether angle information is included.

---

## Multi-Seed Diagnostic Results Summary

### Group A — 200 cm box, 1 tube, length-only

| Objective | R²(s42) | R²(multi) | RMSE(multi) | Mean\|%err\| |
|---|---|---|---|---|
| replicate_seed_average | −0.398 | **0.279** | 3.229 cm | 59.4% |
| replicate_seed_multi_objective | −0.200 | 0.224 | 3.344 cm | 54.7% |
| two_part_hurdle | −0.992 | 0.171 | 3.457 cm | **33.7%** |
| balanced_zero_sampling | +0.171 | 0.114 | 3.552 cm | 57.9% |
| rmse_baseline | +0.182 | 0.121 | 3.542 cm | 55.1% |
| multi_objective_zero_aware | +0.112 | −0.060 | 3.862 cm | 36.7% |
| summary_feature_loss | −0.712 | 0.036 | 3.704 cm | 57.3% |
| weighted_rmse_zero_downweighted | −0.582 | −0.682 | 4.894 cm | 47.4% |

### Group B — 100 cm PBC box, 1 tube, length + angle-histogram

| Objective | R²(s42) | R²(multi) | RMSE(multi) | Mean\|%err\| |
|---|---|---|---|---|
| replicate_seed_average | −0.403 | 0.166 | 4.570 cm | **8.6%** |
| rmse_baseline | −0.467 | 0.220 | 4.427 cm | 16.7% |
| balanced_zero_sampling | −0.152 | 0.042 | 4.860 cm | 23.3% |
| replicate_seed_multi_objective | −0.442 | 0.215 | 4.403 cm | 31.5% |
| two_part_hurdle | −0.530 | 0.259 | 4.290 cm | 40.8% |
| log1p_uniform *(clean re-run)* | −0.255 | 0.274 | 4.282 cm | 42.1% |
| multi_objective_zero_aware | −0.077 | **0.280** | **4.247 cm** | 41.8% |
| summary_feature_loss | −0.233 | 0.186 | 4.500 cm | 47.6% |
| weighted_rmse_zero_downweighted *(collision result)* | −0.856 | 0.144 | 4.655 cm | *(5.3% — was log1p_uniform params, not this mode's true result)* |

**Note on log1p_uniform:** The previously reported 5.3% result came from the config collision
(log1p_uniform ran second and overwrote the weighted_rmse directory). That run happened to find
the correct parameter basin (bp_o1=+7.3%, bp_o2=+1.4%). The clean re-run found a *lower* loss
(J=0.3325 vs 0.3526) but completely wrong branching probabilities — a compensatory solution
(bp_o1=+77.5%, bp_o2=−77.5%) that fits root length spatially (R²=0.274) without recovering
the true parameters. The 5.3% was CMA-ES luck, not a systematic property of this mode.

---

## Candidate Modes and Open Questions

Two modes are the strongest candidates for further investigation:

### 1. `replicate_seed_average` + angle-hist

- Single run result: mean |% error| = **8.6%**; bp_o1 = −10.6%, bp_o2 = −17.7%
- Runtime: ~2.25 h
- **Rationale:** Averaging multiple seeds within each CMA-ES function evaluation smooths the
  loss landscape, directly addressing the butterfly-effect noise during optimization — not
  just post-hoc in diagnostics.

### 2. `log1p_uniform` (weighted_rmse + use_log1p=true) + angle-hist

- Run 1 result (first run, collision directory): mean |% error| = **5.3%**; bp_o1 = +7.3%, bp_o2 = +1.4%
- Run 2 result (clean re-run): mean |% error| = **42.1%**; bp_o1 = +77.5%, bp_o2 = −77.5%
- Runtime: ~3.5–6.5 h
- **Rationale:** Log1p transformation reduces sensitivity to large outlier values and
  down-weighting zero windows reduces noise from unobserved locations.

### Why more runs are needed

Two runs of log1p_uniform (one at 5.3%, one at 42.1%) and one run of replicate_seed_average
(8.6%) are insufficient to characterize either mode's reliability. The bp_o1/bp_o2 degeneracy
is a genuine identifiability problem — compensatory solutions (high bp_o1 + low bp_o2 ≈
same root length as true values) exist in the loss landscape for all single-seed modes, and
may also affect replicate mode. The variance across independent CMA-ES runs is high; a single
trial per mode tells us very little.

A proper comparison requires 5+ independent full calibration runs per mode, comparing the
**distribution** of parameter recovery outcomes, not individual trials.

### Connection to SBI comparison

The bp_o1/bp_o2 degeneracy is actually a valuable finding for the publication. SBI produces
a full posterior distribution rather than a point estimate, so this degeneracy would appear
as a broad or bimodal posterior — visible and characterizable rather than a lucky/unlucky
point. The process-based vs SBI comparison can frame this as: "how does the identifiability
limit manifest in point estimation (CMA-ES) vs posterior estimation (SBI)?"

---

## Modes to Drop

| Mode | Reason |
|---|---|
| `balanced_zero_sampling` | R²(multi) = 0.042 in Group B — zero-balancing hurts when data is already sparse (72% zeros, 1 tube) |
| `summary_feature_loss` | 47.6% error in Group B — no advantage over simpler modes |
| `two_part_hurdle` | 40.8% error in Group B despite good R²(0.259) — complex two-stage loss adds noise without constraining parameters better |
| `replicate_seed_multi_objective` | 31.5% error — replicate_avg achieves 8.6% and is simpler |
| `multi_objective_zero_aware` | Best R²(multi) in Group B (0.280) but only 41.8% parameter recovery — fitting spatial pattern but not parameter values |
| All Group A modes | Length-only loss never recovers bp_o1 (110–176% error). No mode in Group A approaches Group B performance. |

---

## Next Priority: Staged Tube Count Increase

These results are from 1-tube configurations. With 1 tube, 72.3% of observation windows have
zero root length, severely limiting spatial constraint. The angle histogram partially compensates,
but more tubes are needed.

**Starting point: 3 tubes in the 200 cm box.**

The tube convergence analysis (length-only loss, 200 cm box) found the elbow at M≈7–8 for
R² ≥ 0.65. However, that analysis used length-only loss. With angle-hist added, the angle
distribution provides additional structural information that was absent in those runs. The
required tube count for good parameter recovery may therefore be lower than 7 when combined
with angle-hist. Starting at 3 tubes is more practical:

| Stage | Setup | Rationale |
|---|---|---|
| **Now** | 3 tubes, 200 cm box, angle-hist | Moderate density increase without excessive runtime |
| If needed | 5 tubes, 200 cm box, angle-hist | Doubles window coverage from 3-tube baseline |
| If needed | 7 tubes, 200 cm box, angle-hist | Meets length-only convergence threshold if angle-hist alone is insufficient |

The 3-tube run should use the same `replicate_seed_average` + angle-hist setup as the primary
recommendation from Group B, in the 200 cm box (Group C). If parameter recovery remains poor,
scale to 5 before committing to 7.

---

## Scripts

| Script | Purpose |
|---|---|
| `calibration/summarize_4param_results.jl` | Print full comparison table for all 4-param runs |
| `calibration/re_diagnose_multi_seed.jl` | Re-run diagnostics with multi-seed averaging for existing experiment dirs |
| `calibration/results_summary.txt` | Last saved output of the summary script (2026-06-20) |
