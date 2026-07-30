#!/bin/bash
# run_test_cases_s1ms_k14.sh
#
# Runs CMA-ES calibration for all 20 test cases under Scenario 1 Multi-Seed k=14 (S1MS-k14):
#   - Observations: K=14 multi-seed mean (obs_multiseed_k14_*.csv)
#   - All 8 parameters calibrated (elongation + architecture)
#   - sigma0=0.10 cold-start from template
#   - N_WORKERS parallel workers (FIFO queue)
#
# Usage (from repo root):
#   nohup bash calibration/run_test_cases_s1ms_k14.sh [N_WORKERS] \
#     > calibration/logs/s1ms_k14_runner.log 2>&1 &

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(calibration/configs/test_cases/tc_*_s1ms_k14.json)
N_WORKERS=${1:-6}

LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/s1ms_k14_runner.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES S1MS-k14  workers=$N_WORKERS  cases=${#CONFIGS[@]}" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

active=0
for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)
    exp_dir="calibration/experiments/$tag"
    if [ -f "$exp_dir/best_params.json" ]; then
        echo "SKIP (done): $tag" | tee -a "$LOG"
        continue
    fi
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" >> "$LOG_DIR/${tag}.log" 2>&1 &
    active=$((active + 1))
    if [ "$active" -ge "$N_WORKERS" ]; then
        wait -n 2>/dev/null || wait
        active=$((active - 1))
    fi
done

wait

echo "" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
echo "All S1MS-k14 calibrations complete: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
