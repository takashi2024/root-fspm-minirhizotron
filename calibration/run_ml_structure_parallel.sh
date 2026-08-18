#!/bin/bash
# run_ml_structure_parallel.sh
#
# Runs evaluate_ml_structure.jl for each of the 20 test cases in parallel
# using a job-pool (N_PARALLEL workers). Each worker processes one test case
# and writes calibration/ml_struct_tmp/<tag>.csv.
# After all workers finish, aggregates results into comparison_structure_metrics.csv.
#
# Usage (from repo root):
#   bash calibration/run_ml_structure_parallel.sh [N_PARALLEL]
#   Default N_PARALLEL = 4

set -e

JULIA="/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
SCRIPT="calibration/evaluate_ml_structure.jl"
CONFIG_DIR="calibration/configs/test_cases"
TMP_DIR="calibration/ml_struct_tmp"
LOG_DIR="calibration/logs/ml_structure"
AGG_SCRIPT="calibration/aggregate_ml_structure.py"

N_PARALLEL="${1:-6}"

mkdir -p "$TMP_DIR" "$LOG_DIR"

# Build list of tags to process (skip already completed)
PENDING=()
for cfg in $(ls "$CONFIG_DIR"/tc_*.json | sort); do
    tag=$(basename "$cfg" .json)
    if [ -f "$TMP_DIR/${tag}.csv" ]; then
        echo "SKIP (done): $tag"
    else
        PENDING+=("$tag")
    fi
done

TOTAL=${#PENDING[@]}
echo ""
echo "ML structure evaluation — parallel runner — $(date)"
echo "N_PARALLEL: $N_PARALLEL"
echo "Pending:    $TOTAL cases"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "All cases already done. Running aggregation only."
    python3 "$AGG_SCRIPT"
    exit 0
fi

# ── Job pool using a fifo semaphore ──────────────────────────────────────────
FIFO=$(mktemp -t mlstruct_fifo.XXXX) && rm "$FIFO" && mkfifo "$FIFO"
exec 3<>"$FIFO"

for _ in $(seq 1 "$N_PARALLEL"); do echo >&3; done

FAILED=0

for tag in "${PENDING[@]}"; do
    log="$LOG_DIR/${tag}.log"

    read -u 3

    (
        echo "[$(date '+%H:%M:%S')] START  $tag"
        if "$JULIA" "$SCRIPT" "$tag" > "$log" 2>&1; then
            echo "[$(date '+%H:%M:%S')] DONE   $tag"
        else
            echo "[$(date '+%H:%M:%S')] FAILED $tag  (see $log)"
        fi
        echo >&3
    ) &
done

wait
exec 3>&-

echo ""
echo "All workers done — $(date)"
echo "Aggregating results..."
python3 "$AGG_SCRIPT"
