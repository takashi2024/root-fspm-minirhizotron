#!/bin/bash
# wait_and_run_s1ms_k14.sh
#
# Polls until all 20 S1MS k=7 cases are done, then launches S1MS k=14.
# Completion is detected when the runner log contains "All S1MS calibrations complete".
#
# Usage (from repo root):
#   nohup bash calibration/wait_and_run_s1ms_k14.sh [N_WORKERS] \
#     > calibration/logs/s1ms_k14_watcher.log 2>&1 &

N_WORKERS="${1:-6}"
POLL_INTERVAL=300
RUNNER="calibration/run_test_cases_s1ms_k14.sh"
S1MS_LOG="calibration/logs/s1ms_runner.log"
LOG="calibration/logs/s1ms_k14_watcher.log"

mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] S1MS-k14 watcher started — waiting for S1MS k=7 to finish"
echo "  Watching: $S1MS_LOG"

while true; do
    if grep -q "All S1MS calibrations complete" "$S1MS_LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] S1MS k=7 complete — launching S1MS k=14"
        nohup bash "$RUNNER" "$N_WORKERS" >> calibration/logs/s1ms_k14_runner.log 2>&1 &
        echo "[$(date '+%H:%M:%S')] S1MS k=14 runner started (PID $!)"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Waiting... (S1MS k=7 not yet complete)"
    sleep "$POLL_INTERVAL"
done
