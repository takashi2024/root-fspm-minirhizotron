#!/bin/bash
# run_test_case_calibrations_parallel.sh
#
# Runs CMA-ES calibrations in parallel using a job-pool of N_PARALLEL workers.
# Skips any case that already has a completed best_params.json.
#
# Usage:
#   bash calibration/run_test_case_calibrations_parallel.sh [N_PARALLEL]
#   Default N_PARALLEL = 4
#
# Each worker picks the next pending config from the queue and runs it.
# When a worker finishes a case, it immediately picks up the next one.

set -e

JULIA="/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
MAIN="calibration/main_Optimization_mr_flexible.jl"
CONFIG_DIR="calibration/configs/test_cases"
LOG_DIR="calibration/logs/test_cases"
EXP_DIR="calibration/experiments"

N_PARALLEL="${1:-4}"

mkdir -p "$LOG_DIR"

# Build list of pending configs (skip already completed)
PENDING=()
for cfg in $(ls "$CONFIG_DIR"/tc_*.json | sort); do
    tag=$(basename "$cfg" .json)
    if [ -f "$EXP_DIR/$tag/best_params.json" ]; then
        echo "SKIP (done): $tag"
    else
        PENDING+=("$cfg")
    fi
done

TOTAL=${#PENDING[@]}
echo ""
echo "Parallel calibration runner — $(date)"
echo "N_PARALLEL: $N_PARALLEL"
echo "Pending:    $TOTAL cases"
echo "Log dir:    $LOG_DIR"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "All cases already completed."
    exit 0
fi

# ── Job pool using a fifo semaphore ──────────────────────────────────────────
FIFO=$(mktemp -t calib_fifo.XXXX) && rm "$FIFO" && mkfifo "$FIFO"
exec 3<>"$FIFO"

# Fill semaphore with N_PARALLEL tokens
for _ in $(seq 1 "$N_PARALLEL"); do
    echo >&3
done

DONE=0
FAILED=0

for cfg in "${PENDING[@]}"; do
    tag=$(basename "$cfg" .json)
    log="$LOG_DIR/${tag}.log"

    # Acquire a token (blocks if all workers are busy)
    read -u 3

    (
        echo "[$(date '+%H:%M:%S')] START  $tag"
        if "$JULIA" "$MAIN" "$cfg" > "$log" 2>&1; then
            echo "[$(date '+%H:%M:%S')] DONE   $tag"
        else
            echo "[$(date '+%H:%M:%S')] FAILED $tag  (see $log)"
        fi
        # Release token
        echo >&3
    ) &
done

# Wait for all background jobs to finish
wait

exec 3>&-
echo ""
echo "All done — $(date)"
echo "Completed: $((TOTAL - FAILED)) / $TOTAL"
