#!/bin/bash
# run_test_cases_v2_s2ms.sh
#
# Phase 1: CMA-ES S2MS v2 for 18 regular cases (excludes A_497 and C_100).
# skin_depth=0.2 cm, maxiter=100, K=7 multi-seed obs.
#
# Usage (from repo root):
#   bash calibration/run_test_cases_v2_s2ms.sh [N_WORKERS]
# N_WORKERS defaults to 6.

JULIA=/c/Users/au758739/AppData/Local/Programs/Julia-1.11.2/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(calibration/configs/test_cases_v2/tc_v2_*_s2ms.json)
N_WORKERS=${1:-6}

LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v2_s2ms_runner.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES v2 S2MS Phase 1  workers=$N_WORKERS  cases=${#CONFIGS[@]}" | tee -a "$LOG"
echo "skin_depth=0.2 cm  maxiter=100" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

pids=()
for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)

    if [[ "$tag" == *"_497_"* ]] || [[ "$tag" == *"_100_"* ]]; then
        echo "SKIP (Phase 2): $tag" | tee -a "$LOG"
        continue
    fi

    exp_dir="calibration/experiments/$tag"
    if [ -f "$exp_dir/best_params.json" ]; then
        echo "SKIP (done): $tag" | tee -a "$LOG"
        continue
    fi

    # Wait for a slot (poll until a running worker finishes)
    while [ "${#pids[@]}" -ge "$N_WORKERS" ]; do
        new_pids=()
        for p in "${pids[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                new_pids+=("$p")
            fi
        done
        pids=("${new_pids[@]}")
        [ "${#pids[@]}" -ge "$N_WORKERS" ] && sleep 30
    done

    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA $MAIN "$cfg" >> "$LOG_DIR/${tag}.log" 2>&1 &
    pids+=($!)
done

wait

echo "" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
echo "Phase 1 S2MS v2 complete: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
