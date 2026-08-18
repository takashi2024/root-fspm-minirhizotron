#!/bin/bash
# run_test_cases_s2ms_extra.sh
#
# Runs S2MS k=7 CMA-ES for A_497 and C_100 (previously excluded).
# 2 cases sequentially (both are compute-heavy; run with 2 workers).
#
# Usage (from repo root):
#   nohup bash calibration/run_test_cases_s2ms_extra.sh \
#     > calibration/logs/s2ms_extra.log 2>&1 &

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(
  calibration/configs/test_cases/tc_A_497_s2ms.json
  calibration/configs/test_cases/tc_C_100_s2ms.json
)
LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/s2ms_extra.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES S2MS extra (A_497, C_100)  started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" >> "$LOG_DIR/${tag}.log" 2>&1 &
done

wait

echo "" | tee -a "$LOG"
echo "S2MS extra done: $(date)" | tee -a "$LOG"
