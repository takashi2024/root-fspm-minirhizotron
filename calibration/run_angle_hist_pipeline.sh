#!/usr/bin/env bash
# run_angle_hist_pipeline.sh
#
# Orchestrates the full angle-hist calibration pipeline:
#   Phase 1 — Wait for 5 running length-only 4-param jobs to finish
#   Phase 2 — Run the 4 remaining length-only 4-param modes
#   Phase 3 — Regenerate mock CSVs with angle/histogram columns
#   Phase 4 — Run 9 angle-hist 4-param calibrations (batches of 5 + 4)
#
# Run from repo root:
#   bash calibration/run_angle_hist_pipeline.sh &> calibration/run_angle_hist_pipeline.log &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
CALIB_SCRIPT="$SCRIPT_DIR/main_Optimization_mr_flexible.jl"
LOG_DIR="$SCRIPT_DIR/pipeline_logs"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ── Phase 1: Wait for the current 5 running jobs ─────────────────────────────
WAIT_PIDS=(45404 45405 45406 45407 45408)
log "Phase 1: Waiting for PIDs ${WAIT_PIDS[*]} to finish..."
for pid in "${WAIT_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        log "  Waiting on PID $pid..."
        while kill -0 "$pid" 2>/dev/null; do sleep 30; done
        log "  PID $pid done."
    else
        log "  PID $pid already finished."
    fi
done
log "Phase 1 complete."

# ── Phase 2: Run remaining 4 length-only 4-param modes ───────────────────────
log "Phase 2: Launching remaining 4 length-only 4-param calibrations..."
REMAINING_CONFIGS=(
    "configs/calib_200cm_1tube_4param_two_part_hurdle.json"
    "configs/calib_200cm_1tube_4param_summary_features.json"
    "configs/calib_200cm_1tube_4param_replicate_avg.json"
    "configs/calib_200cm_1tube_4param_replicate_multi.json"
)
PHASE2_PIDS=()
for cfg in "${REMAINING_CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    logfile="$LOG_DIR/${name}.log"
    log "  Launching $name..."
    "$JULIA" --threads 2 "$CALIB_SCRIPT" "$cfg" \
        >"$logfile" 2>&1 &
    PHASE2_PIDS+=($!)
done
log "Phase 2 PIDs: ${PHASE2_PIDS[*]}"
for pid in "${PHASE2_PIDS[@]}"; do
    wait "$pid" && log "  Job $pid finished OK." || log "  Job $pid exited with error $?."
done
log "Phase 2 complete."

# ── Phase 3: Regenerate mock CSVs with angle/histogram columns ───────────────
log "Phase 3: Regenerating mock observation CSVs (1-tube then 3-tube)..."
cd "$REPO_DIR"

log "  Running generate_mock_mr_obs_1tube_200cm.jl..."
"$JULIA" calibration/generate_mock_mr_obs_1tube_200cm.jl \
    >"$LOG_DIR/mock_1tube_200cm.log" 2>&1 \
    && log "  1-tube mock CSV done." \
    || { log "  ERROR: 1-tube mock generator failed. Check $LOG_DIR/mock_1tube_200cm.log"; exit 1; }

log "  Running generate_mock_mr_obs_3tubes_200cm.jl..."
"$JULIA" calibration/generate_mock_mr_obs_3tubes_200cm.jl \
    >"$LOG_DIR/mock_3tubes_200cm.log" 2>&1 \
    && log "  3-tube mock CSV done." \
    || { log "  ERROR: 3-tube mock generator failed. Check $LOG_DIR/mock_3tubes_200cm.log"; exit 1; }

log "Phase 3 complete."

# ── Phase 4: Launch 9 angle-hist calibrations ────────────────────────────────
log "Phase 4: Launching 9 angle-hist calibrations (batch 1: 5 jobs)..."
ANGLE_CONFIGS=(
    "configs/calib_200cm_1tube_4param_angle_hist.json"
    "configs/calib_200cm_1tube_4param_angle_hist_rmse_baseline.json"
    "configs/calib_200cm_1tube_4param_angle_hist_weighted_rmse.json"
    "configs/calib_200cm_1tube_4param_angle_hist_balanced_sampling.json"
    "configs/calib_200cm_1tube_4param_angle_hist_two_part_hurdle.json"
    "configs/calib_200cm_1tube_4param_angle_hist_summary_features.json"
    "configs/calib_200cm_1tube_4param_angle_hist_replicate_avg.json"
    "configs/calib_200cm_1tube_4param_angle_hist_replicate_multi.json"
    "configs/calib_200cm_1tube_4param_angle_hist_log1p_uniform.json"
)

cd "$SCRIPT_DIR"
BATCH1=("${ANGLE_CONFIGS[@]:0:5}")
BATCH2=("${ANGLE_CONFIGS[@]:5:4}")

BATCH1_PIDS=()
for cfg in "${BATCH1[@]}"; do
    name=$(basename "$cfg" .json)
    logfile="$LOG_DIR/${name}.log"
    log "  Launching $name..."
    "$JULIA" --threads 2 "$CALIB_SCRIPT" "$cfg" \
        >"$logfile" 2>&1 &
    BATCH1_PIDS+=($!)
done
log "Batch 1 PIDs: ${BATCH1_PIDS[*]}"
for pid in "${BATCH1_PIDS[@]}"; do
    wait "$pid" && log "  Job $pid finished OK." || log "  Job $pid exited with error $?."
done
log "Phase 4 batch 1 complete. Launching batch 2 (4 jobs)..."

BATCH2_PIDS=()
for cfg in "${BATCH2[@]}"; do
    name=$(basename "$cfg" .json)
    logfile="$LOG_DIR/${name}.log"
    log "  Launching $name..."
    "$JULIA" --threads 2 "$CALIB_SCRIPT" "$cfg" \
        >"$logfile" 2>&1 &
    BATCH2_PIDS+=($!)
done
log "Batch 2 PIDs: ${BATCH2_PIDS[*]}"
for pid in "${BATCH2_PIDS[@]}"; do
    wait "$pid" && log "  Job $pid finished OK." || log "  Job $pid exited with error $?."
done
log "Phase 4 complete."

log "=========================================="
log "All pipeline stages done."
log "  Length-only results: calibration/experiments/"
log "  Angle-hist results:  calibration/experiments/ (objective names contain same modes)"
log "  Summary:             julia calibration/summarize_4param_results.jl"
log "=========================================="
