#!/bin/bash
# run_test_cases_v2_s1.sh
#
# CMA-ES S1 v2 — k=1 single-seed obs — all 20 cases.
# skin_depth=0.2 cm, maxiter=100, K=1 obs, 8 params.
#
# Usage (from repo root):
#   bash calibration/run_test_cases_v2_s1.sh [N_WORKERS]
# N_WORKERS defaults to 6.

JULIA=/c/Users/au758739/AppData/Local/Programs/Julia-1.11.2/bin/julia
MAIN=calibration/main_Optimization_mr_flexible.jl
CONFIGS=(calibration/configs/test_cases_v2/tc_v2_*_s1.json)
N_WORKERS=${1:-6}

LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v2_s1_runner.log"

echo "=======================================================" | tee -a "$LOG"
echo "CMA-ES v2 S1 (k=1)  workers=$N_WORKERS  cases=${#CONFIGS[@]}" | tee -a "$LOG"
echo "skin_depth=0.2 cm  maxiter=100  8 params" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

pids=()
for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .json)

    exp_dir="calibration/experiments/$tag"
    if [ -f "$exp_dir/best_params.json" ]; then
        echo "SKIP (done): $tag" | tee -a "$LOG"
        continue
    fi

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
echo "S1 v2 (k=1) complete: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
