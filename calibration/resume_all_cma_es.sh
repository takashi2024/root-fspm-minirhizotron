#!/bin/bash
# resume_all_cma_es.sh
#
# Auto-resume entry point for Windows Task Scheduler (runs "At startup").
# Relaunches all 6 CMA-ES runner scripts (which already skip completed cases)
# plus the 2 manually-tracked "Phase 2" cases that the standard s2ms runner
# permanently excludes (tc_v2_A_497_s2ms, tc_v2_C_100_s2ms).
#
# This script deliberately blocks (via `wait` at the end) until everything
# finishes, so the Scheduled Task shows as "Running" for the whole multi-day
# duration rather than exiting immediately and leaving orphaned children that
# a task-completion cleanup could kill.

cd /d/rfspm-git || exit 1
mkdir -p calibration/logs
LOG="calibration/logs/auto_resume_$(date +%Y%m%d_%H%M%S).log"
echo "Auto-resume triggered at $(date)" > "$LOG"

pids=()
for scenario in s1 s2 s1ms s2ms s1ms_k14 s2ms_k14; do
    bash "calibration/run_test_cases_v2_${scenario}.sh" 2 >> "$LOG" 2>&1 &
    pids+=($!)
    echo "  launched run_test_cases_v2_${scenario}.sh (pid $!)" >> "$LOG"
done

JULIA=$(which julia)
for tag in tc_v2_A_497_s2ms tc_v2_C_100_s2ms; do
    if [ ! -f "calibration/experiments/${tag}/best_params.json" ]; then
        "$JULIA" calibration/main_Optimization_mr_flexible.jl \
            "calibration/configs/test_cases_v2/${tag}.json" \
            > "calibration/logs/${tag}_manual.log" 2>&1 &
        pids+=($!)
        echo "  launched manual case ${tag} (pid $!)" >> "$LOG"
    fi
done

echo "Launched ${#pids[@]} top-level jobs at $(date)" >> "$LOG"
wait
echo "All resume jobs finished at $(date)" >> "$LOG"
