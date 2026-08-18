#!/bin/bash
# watch_and_switch_to_parallel.sh
#
# Waits for tc_A_201_s2 to finish, then:
#   1. Kills the current sequential runner (and its Julia child)
#   2. Launches the parallel runner for all remaining cases
#
# Usage (run from repo root):
#   bash calibration/watch_and_switch_to_parallel.sh SEQUENTIAL_PID [N_PARALLEL]
#
# SEQUENTIAL_PID: PID of the currently running sequential runner (bash script)
# N_PARALLEL:     number of parallel workers (default: 4)

SEQUENTIAL_PID="${1:?Usage: $0 SEQUENTIAL_PID [N_PARALLEL]}"
N_PARALLEL="${2:-4}"

EXP_DIR="calibration/experiments"
WATCH_TAG="tc_A_201_s2"
WATCH_FILE="$EXP_DIR/$WATCH_TAG/best_params.json"
PARALLEL_RUNNER="calibration/run_test_case_calibrations_parallel.sh"
PARALLEL_LOG="calibration/logs/run_test_cases_parallel.log"

echo "[$(date '+%H:%M:%S')] Watching for $WATCH_TAG to complete..."
echo "  Sequential runner PID: $SEQUENTIAL_PID"
echo "  Will launch $N_PARALLEL parallel workers after switch"

# Poll until best_params.json appears in tc_A_201_s2
while [ ! -f "$WATCH_FILE" ]; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] $WATCH_TAG completed — switching to parallel runner"

# Kill sequential runner and its Julia child
if kill -0 "$SEQUENTIAL_PID" 2>/dev/null; then
    # Find Julia child spawned by the sequential runner
    JULIA_PID=$(pgrep -P "$SEQUENTIAL_PID" julia 2>/dev/null || true)
    kill "$SEQUENTIAL_PID" 2>/dev/null && echo "  Killed sequential runner (PID $SEQUENTIAL_PID)" || true
    if [ -n "$JULIA_PID" ]; then
        # Give Julia a moment to notice parent is gone; then force kill
        sleep 2
        kill "$JULIA_PID" 2>/dev/null && echo "  Killed Julia child (PID $JULIA_PID)" || true
    fi
else
    echo "  Sequential runner already finished (PID $SEQUENTIAL_PID not found)"
fi

sleep 3  # brief pause before launching

echo "[$(date '+%H:%M:%S')] Launching parallel runner (N=$N_PARALLEL)..."
nohup bash "$PARALLEL_RUNNER" "$N_PARALLEL" >> "$PARALLEL_LOG" 2>&1 &
PARALLEL_PID=$!
echo "[$(date '+%H:%M:%S')] Parallel runner started (PID $PARALLEL_PID)"
echo "  Log: $PARALLEL_LOG"
