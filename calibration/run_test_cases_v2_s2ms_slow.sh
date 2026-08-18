#!/bin/bash
# run_test_cases_v2_s2ms_slow.sh
#
# Phase 2: CMA-ES S2MS v2 for slow cases only (A_497 and C_100).
# Run after Phase 1 completes. These cases take 6-12h each.
#
# Usage (from repo root):
#   bash calibration/run_test_cases_v2_s2ms_slow.sh

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl

CONFIGS=(
    calibration/configs/test_cases_v2/tc_v2_A_497_s2ms.json
    calibration/configs/test_cases_v2/tc_v2_C_100_s2ms.json
)

LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v2_s2ms_slow_runner.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES v2 S2MS Phase 2 (slow cases): A_497 + C_100" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "SKIP (config missing): $cfg" | tee -a "$LOG"
        continue
    fi
    tag=$(basename "$cfg" .json)
    exp_dir="calibration/experiments/$tag"
    if [ -f "$exp_dir/best_params.json" ]; then
        echo "SKIP (done): $tag" | tee -a "$LOG"
        continue
    fi
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" 2>&1 | tee -a "$LOG_DIR/${tag}.log"
    echo "<<< Done: $tag  ($(date))" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "Phase 2 S2MS v2 complete: $(date)" | tee -a "$LOG"
