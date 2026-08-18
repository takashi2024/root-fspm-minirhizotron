#!/bin/bash
# wait_and_run_s2ms_k14.sh
#
# Polls until k=14 structural eval is done (comparison_structure_metrics_s2ms_k14.csv),
# then launches CMA-ES S2MS k=14 calibration (18 cases).
#
# Usage (from repo root):
#   nohup bash calibration/wait_and_run_s2ms_k14.sh [N_WORKERS] \
#     > calibration/logs/s2ms_k14_watcher.log 2>&1 &

N_WORKERS="${1:-6}"
POLL_INTERVAL=120
TRIGGER_FILE="calibration/comparison_structure_metrics_s2ms_k14.csv"
RUNNER="calibration/run_test_cases_s2ms_k14.sh"
LOG="calibration/logs/s2ms_k14_runner.log"

mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] Watcher started — waiting for k=14 structural eval to finish"
echo "  Trigger: $TRIGGER_FILE"
echo "  Will launch: bash $RUNNER $N_WORKERS"

while true; do
    if [ -f "$TRIGGER_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] Structural eval done — launching CMA-ES S2MS k=14"
        nohup bash "$RUNNER" "$N_WORKERS" >> "$LOG" 2>&1 &
        echo "[$(date '+%H:%M:%S')] CMA-ES runner started (PID $!)"
        echo "  Log: $LOG"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Waiting... ($TRIGGER_FILE not yet present)"
    sleep "$POLL_INTERVAL"
done
