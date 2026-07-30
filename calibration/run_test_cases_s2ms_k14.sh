#!/bin/bash
# run_test_cases_s2ms_k14.sh
#
# Runs CMA-ES calibration for 18 test cases under Scenario 2 Multi-Seed (S2MS):
#   - Observations: K=7 multi-seed mean (obs_multiseed_*.csv)
#   - 4 architectural parameters (elongation fixed at true values)
#   - sigma0=0.10 cold-start
#   - 6 parallel workers (FIFO queue)
#
# Excludes A_497 and C_100 (prohibitive simulation runtime).
#
# Usage (from repo root):
#   bash calibration/run_test_cases_s2ms_k14.sh [N_WORKERS]
# N_WORKERS defaults to 6.

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(calibration/configs/test_cases/tc_*_s2ms_k14.json)
N_WORKERS=${1:-6}

LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_s2ms.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES S2MS  workers=$N_WORKERS  cases=${#CONFIGS[@]}" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

# FIFO parallel runner — keeps N_WORKERS jobs running at all times
active=0
for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" >> "$LOG_DIR/${tag}.log" 2>&1 &
    active=$((active + 1))
    if [ "$active" -ge "$N_WORKERS" ]; then
        wait -n 2>/dev/null || wait   # wait for any one job to finish
        active=$((active - 1))
    fi
done

# Wait for remaining jobs
wait

echo "" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
echo "All S2MS calibrations complete: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
