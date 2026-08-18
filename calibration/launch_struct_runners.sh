#!/bin/bash
# launch_struct_runners.sh
#
# Waits for any currently running evaluate_ml_structure_unified.jl processes to finish,
# then launches the full parallel runners for all ready modes.
# Run from repo root with: nohup bash calibration/launch_struct_runners.sh > calibration/logs/struct_launch.log 2>&1 &

LOG=calibration/logs/struct_launch.log
mkdir -p calibration/logs

echo "[$(date '+%H:%M:%S')] Watcher started — waiting for running Julia evals to finish"

# Wait for any active evaluate_ml_structure_unified.jl processes
while pgrep -f "evaluate_ml_structure_unified.jl" > /dev/null 2>&1; do
    echo "[$(date '+%H:%M:%S')] Waiting for active evals..."
    sleep 30
done

echo "[$(date '+%H:%M:%S')] All evals finished — launching full runners"

# Launch S2 modes (most important first)
nohup bash calibration/run_ml_structure_unified.sh s2_k7  2 \
  >> calibration/logs/struct_s2_k7.log  2>&1 &
echo "[$(date '+%H:%M:%S')] s2_k7  PID: $!"

nohup bash calibration/run_ml_structure_unified.sh s2_k14 2 \
  >> calibration/logs/struct_s2_k14.log 2>&1 &
echo "[$(date '+%H:%M:%S')] s2_k14 PID: $!"

nohup bash calibration/run_ml_structure_unified.sh s2_k1  2 \
  >> calibration/logs/struct_s2_k1.log  2>&1 &
echo "[$(date '+%H:%M:%S')] s2_k1  PID: $!"

# Launch S1 modes (excluding s1_k14 which waits for D2_465)
nohup bash calibration/run_ml_structure_unified.sh s1_k7  2 \
  >> calibration/logs/struct_s1_k7.log  2>&1 &
echo "[$(date '+%H:%M:%S')] s1_k7  PID: $!"

nohup bash calibration/run_ml_structure_unified.sh s1_k1  1 \
  >> calibration/logs/struct_s1_k1.log  2>&1 &
echo "[$(date '+%H:%M:%S')] s1_k1  PID: $!"

echo "[$(date '+%H:%M:%S')] All runners launched. Monitor with:"
echo "  tail -f calibration/logs/struct_*.log"
