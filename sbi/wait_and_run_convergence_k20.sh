#!/bin/bash
# wait_and_run_convergence_k20.sh
#
# Polls until all 4 configs have tensors_k20/X_raw.npz, then runs
# analyze_seed_convergence.py --k20 to regenerate convergence figures.
#
# Usage (from repo root):
#   nohup bash sbi/wait_and_run_convergence_k20.sh \
#     > sbi/logs/convergence_k20_watcher.log 2>&1 &

POLL_INTERVAL=120
CONFIGS=(A B2 C D2)
N_CONFIGS=${#CONFIGS[@]}
DATA_ROOT="sbi/data/multi_seed"
SCRIPT="sbi/python/analyze_seed_convergence.py"
LOG="sbi/logs/convergence_k20_watcher.log"

mkdir -p sbi/logs

echo "[$(date '+%H:%M:%S')] Watcher started — waiting for K=20 tensors (all 4 configs)"
echo "  Polling every ${POLL_INTERVAL}s"

while true; do
    N_DONE=0
    for cfg in "${CONFIGS[@]}"; do
        f="$DATA_ROOT/config_${cfg}/tensors_k20/X_raw.npz"
        if [ -f "$f" ]; then
            N_DONE=$((N_DONE + 1))
        fi
    done

    echo "[$(date '+%H:%M:%S')] tensors_k20 ready: $N_DONE / $N_CONFIGS"

    if [ "$N_DONE" -ge "$N_CONFIGS" ]; then
        echo "[$(date '+%H:%M:%S')] All configs complete — running convergence analysis (K=20)"
        python3 "$SCRIPT" --k20
        EXIT_CODE=$?
        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Done. Figures saved to sbi/figures/seed_convergence_k20*.png"
        else
            echo "[$(date '+%H:%M:%S')] ERROR: analyze_seed_convergence.py exited with code $EXIT_CODE"
        fi
        break
    fi

    sleep "$POLL_INTERVAL"
done
