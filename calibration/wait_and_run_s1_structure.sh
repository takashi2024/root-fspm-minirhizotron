#!/bin/bash
# wait_and_run_s1_structure.sh
#
# Polls until all 20 Scenario 1 CMA-ES calibrations have best_params.json,
# then launches the structural evaluation in parallel (N_PARALLEL workers).
#
# Usage (from repo root):
#   nohup bash calibration/wait_and_run_s1_structure.sh [N_PARALLEL] \
#     > calibration/logs/s1_structure_watcher.log 2>&1 &

N_PARALLEL="${1:-6}"
N_CASES=20
POLL_INTERVAL=300   # check every 5 minutes
EXP_DIR="calibration/experiments"
RUNNER="calibration/run_ml_structure_parallel_s1.sh"
RUNNER_LOG="calibration/logs/s1_structure_runner.log"

mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] Watcher started — waiting for $N_CASES S1 calibrations"
echo "  Polling every ${POLL_INTERVAL}s"
echo "  Will launch: bash $RUNNER $N_PARALLEL"

while true; do
    N_DONE=$(ls "$EXP_DIR"/tc_*_s1/best_params.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] S1 calibrations done: $N_DONE / $N_CASES"

    if [ "$N_DONE" -ge "$N_CASES" ]; then
        echo "[$(date '+%H:%M:%S')] All $N_CASES calibrations complete — launching structural eval"
        nohup bash "$RUNNER" "$N_PARALLEL" >> "$RUNNER_LOG" 2>&1 &
        RUNNER_PID=$!
        echo "[$(date '+%H:%M:%S')] Structural runner started (PID $RUNNER_PID)"
        echo "  Log: $RUNNER_LOG"
        break
    fi

    sleep "$POLL_INTERVAL"
done
