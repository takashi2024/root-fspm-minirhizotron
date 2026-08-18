#!/bin/bash
# wait_and_run_s2ms_structure.sh
#
# Polls until all 18 S2MS CMA-ES calibrations have best_params.json,
# then launches the structural evaluation in parallel (N_PARALLEL workers).
#
# Usage (from repo root):
#   nohup bash calibration/wait_and_run_s2ms_structure.sh [N_PARALLEL] \
#     > calibration/logs/s2ms_structure_watcher.log 2>&1 &

N_PARALLEL="${1:-4}"
N_CASES=18
POLL_INTERVAL=300   # check every 5 minutes
EXP_DIR="calibration/experiments"
RUNNER="calibration/run_ml_structure_parallel_s2ms.sh"
RUNNER_LOG="calibration/logs/s2ms_structure_runner.log"

mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] Watcher started — waiting for $N_CASES S2MS calibrations"
echo "  Polling every ${POLL_INTERVAL}s"
echo "  Will launch: bash $RUNNER $N_PARALLEL"

while true; do
    N_DONE=$(ls "$EXP_DIR"/tc_*_s2ms/best_params.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] S2MS calibrations done: $N_DONE / $N_CASES"

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
