#!/bin/bash
# run_ml_structure_v2_parallel.sh
#
# Parallel structural evaluation for v2 CMA-ES calibrations.
# Runs evaluate_ml_structure_v2.jl per case in a FIFO worker pool.
# Covers both s2_k7 (20 cases) and s1_k7 (18 cases; A_497/C_100 auto-skipped).
#
# Usage (from repo root):
#   bash calibration/run_ml_structure_v2_parallel.sh [N_WORKERS]
#   Default N_WORKERS = 6

set -euo pipefail

JULIA="${JULIA:-/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia}"
SCRIPT="calibration/evaluate_ml_structure_v2.jl"
CONFIG_DIR_S2="calibration/configs/test_cases_v2"
TMP_S2="calibration/ml_struct_tmp_v2_s2ms"
TMP_S1="calibration/ml_struct_tmp_v2_s1ms"
LOG_DIR="calibration/logs/ml_structure_v2"
AGG_S2="calibration/aggregate_ml_structure_s2ms_v2.py"
AGG_S1="calibration/aggregate_ml_structure_s1ms_v2.py"

N_WORKERS="${1:-6}"

mkdir -p "$TMP_S2" "$TMP_S1" "$LOG_DIR"

run_mode() {
    local MODE="$1"          # s2_k7 or s1_k7
    local SUFFIX="$2"        # _s2ms or _s1ms
    local TMP_DIR="$3"
    local AGG_SCRIPT="$4"

    echo ""
    echo "======================================================================"
    echo "Mode: $MODE  (suffix: $SUFFIX)  —  $(date)"
    echo "======================================================================"

    # Collect configs for this mode
    PENDING=()
    while IFS= read -r cfg; do
        tag=$(basename "$cfg" .json)
        if [ -f "$TMP_DIR/${tag}.csv" ]; then
            echo "SKIP (done): $tag"
        else
            PENDING+=("$tag")
        fi
    done < <(find "$CONFIG_DIR_S2" -name "tc_v2_*${SUFFIX}.json" | sort)

    TOTAL=${#PENDING[@]}
    echo "Pending: $TOTAL  |  N_WORKERS: $N_WORKERS"
    echo ""

    if [ "$TOTAL" -eq 0 ]; then
        echo "All $MODE cases already done."
    else
        # FIFO semaphore job pool (bash 3.2 compatible)
        FIFO=$(mktemp -t mlstruct_v2.XXXX) && rm "$FIFO" && mkfifo "$FIFO"
        exec 3<>"$FIFO"
        for _ in $(seq 1 "$N_WORKERS"); do echo >&3; done

        for tag in "${PENDING[@]}"; do
            log="$LOG_DIR/${tag}.log"
            read -u 3
            (
                echo "[$(date '+%H:%M:%S')] START  $tag"
                if "$JULIA" "$SCRIPT" "$MODE" "$tag" > "$log" 2>&1; then
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
        echo "All $MODE workers done — $(date)"
    fi

    echo "Aggregating $MODE results..."
    python3 "$AGG_SCRIPT"
}

run_mode "s2_k7" "_s2ms" "$TMP_S2" "$AGG_S2"
run_mode "s1_k7" "_s1ms" "$TMP_S1" "$AGG_S1"

echo ""
echo "======================================================================"
echo "Both modes complete — $(date)"
echo "Run: python3 calibration/summarize_test_case_results_v2.py"
echo "======================================================================"
