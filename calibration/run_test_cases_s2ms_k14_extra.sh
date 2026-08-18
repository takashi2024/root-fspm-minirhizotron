#!/bin/bash
# run_test_cases_s2ms_k14_extra.sh
#
# Runs S2MS k=14 CMA-ES for A_497 and C_100 (not in original glob).
#
# Usage (from repo root):
#   nohup bash calibration/run_test_cases_s2ms_k14_extra.sh \
#     > calibration/logs/s2ms_k14_extra.log 2>&1 &

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(
  calibration/configs/test_cases/tc_A_497_s2ms_k14.json
  calibration/configs/test_cases/tc_C_100_s2ms_k14.json
)
LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/s2ms_k14_extra.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES S2MS k=14 extra (A_497, C_100)  started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" >> "$LOG_DIR/${tag}.log" 2>&1 &
done

wait

echo "" | tee -a "$LOG"
echo "S2MS k=14 extra done: $(date)" | tee -a "$LOG"
