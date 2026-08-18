#!/bin/bash
# watch_and_run_ml_structure.sh
#
# Waits for sbi/data/combined/test_case_eval/results.csv to appear
# (written when TabICL training in evaluate_test_cases.py completes),
# then launches the parallel structural evaluation with N_PARALLEL workers.
#
# Usage (from repo root):
#   bash calibration/watch_and_run_ml_structure.sh [N_PARALLEL]
#   Default N_PARALLEL = 6

N_PARALLEL="${1:-6}"
WATCH_FILE="sbi/data/combined/test_case_eval/results.csv"
RUNNER="calibration/run_ml_structure_parallel.sh"
LOG="calibration/logs/ml_structure_runner.log"

mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] Watching for TabICL results: $WATCH_FILE"
echo "  Will launch $N_PARALLEL parallel workers when ready"

while [ ! -f "$WATCH_FILE" ]; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] results.csv found — TabICL complete"
echo "[$(date '+%H:%M:%S')] Launching structural evaluation (N=$N_PARALLEL)..."

nohup bash "$RUNNER" "$N_PARALLEL" >> "$LOG" 2>&1 &
RUNNER_PID=$!
echo "[$(date '+%H:%M:%S')] Runner started (PID $RUNNER_PID)"
echo "  Log: $LOG"
