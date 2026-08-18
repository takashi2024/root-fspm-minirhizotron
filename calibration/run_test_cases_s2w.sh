#!/bin/bash
# run_test_cases_s2w.sh
#
# Runs CMA-ES Scenario 2 Warm-start (S2W) calibrations for all 20 test cases
# in parallel using a FIFO job pool.
#
# Usage (from repo root):
#   bash calibration/run_test_cases_s2w.sh [N_PARALLEL]
#   Default N_PARALLEL = 6
#
# Each worker calls main_Optimization_mr_flexible.jl with a tc_*_s2w config.
# Results land in calibration/experiments/tc_*_s2w/best_params.json

set -e

JULIA="/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia"
MAIN="calibration/main_Optimization_mr_flexible.jl"
CONFIG_DIR="calibration/configs/test_cases"
LOG_DIR="calibration/logs/s2w"
EXP_DIR="calibration/experiments"

N_PARALLEL="${1:-6}"

mkdir -p "$LOG_DIR"

# ── Build list of pending cases (skip already completed) ─────────────────────
PENDING=()
for cfg in $(ls "$CONFIG_DIR"/tc_*_s2w.json | sort); do
    tag=$(basename "$cfg" .json)
    if [ -f "$EXP_DIR/${tag}/best_params.json" ]; then
        echo "SKIP (done): $tag"
    else
        PENDING+=("$cfg")
    fi
done

TOTAL=${#PENDING[@]}
echo ""
echo "CMA-ES S2W calibrations — parallel runner — $(date)"
echo "N_PARALLEL:  $N_PARALLEL"
echo "Pending:     $TOTAL / 20 cases"
echo "sigma0:      0.05 (TabICL warm-start)"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "All 20 S2W cases already complete."
    exit 0
fi

# ── FIFO semaphore ───────────────────────────────────────────────────────────
FIFO=$(mktemp -t s2w_fifo.XXXX) && rm "$FIFO" && mkfifo "$FIFO"
exec 3<>"$FIFO"
for _ in $(seq 1 "$N_PARALLEL"); do echo >&3; done

# ── Launch workers ───────────────────────────────────────────────────────────
for cfg_path in "${PENDING[@]}"; do
    tag=$(basename "$cfg_path" .json)
    log="$LOG_DIR/${tag}.log"
    read -u 3
    (
        echo "[$(date '+%H:%M:%S')] START  $tag"
        if "$JULIA" "$MAIN" "$cfg_path" > "$log" 2>&1; then
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
echo "All S2W workers done — $(date)"
echo ""

# ── Quick status check ───────────────────────────────────────────────────────
DONE=0; FAILED=0
for cfg in $(ls "$CONFIG_DIR"/tc_*_s2w.json | sort); do
    tag=$(basename "$cfg" .json)
    if [ -f "$EXP_DIR/${tag}/best_params.json" ]; then
        DONE=$((DONE+1))
    else
        FAILED=$((FAILED+1))
        echo "  MISSING best_params.json: $tag"
    fi
done
echo "Completed: $DONE / 20"
[ "$FAILED" -gt 0 ] && echo "Failed:    $FAILED" && exit 1 || exit 0
