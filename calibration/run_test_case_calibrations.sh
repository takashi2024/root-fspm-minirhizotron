#!/bin/bash
# run_test_case_calibrations.sh
#
# Runs CMA-ES calibration sequentially for all 20 test cases (Scenario 2).
# Each calibration writes results to calibration/experiments/tc_<config>_<sample_id>_s2/
#
# Estimated runtime: ~3h per calibration × 20 = ~60h total.
# Run in background: nohup bash calibration/run_test_case_calibrations.sh > calibration/logs/run_test_cases.log 2>&1 &

set -e

JULIA="/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
MAIN="calibration/main_Optimization_mr_flexible.jl"
CONFIG_DIR="calibration/configs/test_cases"
LOG_DIR="calibration/logs/test_cases"

mkdir -p "$LOG_DIR"

CONFIGS=($(ls "$CONFIG_DIR"/tc_*.json | sort))
TOTAL=${#CONFIGS[@]}

echo "Starting ${TOTAL} calibrations at $(date)"
echo "Configs: $CONFIG_DIR"
echo "Logs:    $LOG_DIR"
echo ""

DONE=0
FAILED=0

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)
    log="$LOG_DIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] Starting $tag ($((DONE+1))/${TOTAL})"

    if "$JULIA" "$MAIN" "$cfg" > "$log" 2>&1; then
        echo "[$(date '+%H:%M:%S')] Done    $tag"
        DONE=$((DONE + 1))
    else
        echo "[$(date '+%H:%M:%S')] FAILED  $tag (see $log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Finished at $(date)"
echo "Completed: $DONE / $TOTAL"
echo "Failed:    $FAILED"
