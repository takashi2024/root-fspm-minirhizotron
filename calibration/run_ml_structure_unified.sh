#!/bin/bash
# run_ml_structure_unified.sh
#
# Runs evaluate_ml_structure_unified.jl for all 20 cases in a given mode,
# using a FIFO queue of N_WORKERS parallel Julia processes.
#
# Usage (from repo root):
#   nohup bash calibration/run_ml_structure_unified.sh <mode> [N_WORKERS] \
#     > calibration/logs/struct_<mode>.log 2>&1 &
#
# mode is one of: s2_k1 s2_k7 s2_k14 s1_k1 s1_k7 s1_k14
# N_WORKERS defaults to 4 (each worker uses ~1 CPU, runs 12 seeds sequentially)

MODE="${1:?Usage: $0 <mode> [N_WORKERS]}"
N_WORKERS="${2:-4}"

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
SCRIPT=calibration/evaluate_ml_structure_unified.jl
LOG_DIR=calibration/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/struct_${MODE}.log"

# Derive experiment suffix from mode
case "$MODE" in
  s2_k1)   EXP_SUFFIX="_s2" ;;
  s2_k7)   EXP_SUFFIX="_s2ms" ;;
  s2_k14)  EXP_SUFFIX="_s2ms_k14" ;;
  s1_k1)   EXP_SUFFIX="_s1" ;;
  s1_k7)   EXP_SUFFIX="_s1ms" ;;
  s1_k14)  EXP_SUFFIX="_s1ms_k14" ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

# Build list of tags from test_cases_selected.csv
TAGS=()
while IFS=, read -r sample_id _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ config _; do
    [[ "$config" == "config" ]] && continue  # skip header
    [[ -z "$config" || -z "$sample_id" ]] && continue
    TAGS+=("tc_${config}_${sample_id}${EXP_SUFFIX}")
done < calibration/test_cases_selected.csv

echo "=======================================================" | tee -a "$LOG"
echo "Structural eval: mode=$MODE  workers=$N_WORKERS  cases=${#TAGS[@]}" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

# Correct tmp dir per mode (must match ModeConfig in evaluate_ml_structure_unified.jl)
case "$MODE" in
  s2_k1)   TMP_DIR="calibration/ml_struct_tmp_s2_k1" ;;
  s2_k7)   TMP_DIR="calibration/ml_struct_tmp_s2ms" ;;
  s2_k14)  TMP_DIR="calibration/ml_struct_tmp_s2ms_k14" ;;
  s1_k1)   TMP_DIR="calibration/ml_struct_tmp_s1_k1" ;;
  s1_k7)   TMP_DIR="calibration/ml_struct_tmp_s1ms" ;;
  s1_k14)  TMP_DIR="calibration/ml_struct_tmp_s1ms_k14" ;;
esac

active=0
for tag in "${TAGS[@]}"; do
    per_case_csv="${TMP_DIR}/${tag}.csv"
    if [ -f "$per_case_csv" ]; then
        echo "SKIP (done): $tag" | tee -a "$LOG"
        continue
    fi
    echo ">>> $tag  ($(date))" | tee -a "$LOG"
    $JULIA "$SCRIPT" "$MODE" "$tag" >> "$LOG_DIR/struct_${MODE}_${tag}.log" 2>&1 &
    active=$((active + 1))
    if [ "$active" -ge "$N_WORKERS" ]; then
        wait -n 2>/dev/null || wait
        active=$((active - 1))
    fi
done

wait

echo "" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"
echo "All structural evals done for mode=$MODE: $(date)" | tee -a "$LOG"
echo "=======================================================" | tee -a "$LOG"

# Aggregate per-case CSVs into final output
echo "Aggregating results..." | tee -a "$LOG"
python3 - <<'PYEOF'
import os, glob, sys
import pandas as pd

mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODE", "")
calib_dir = "calibration"
tmp_dir_map = {
    "s2_k1":  "ml_struct_tmp_s2_k1",
    "s2_k7":  "ml_struct_tmp_s2ms",
    "s2_k14": "ml_struct_tmp_s2ms_k14",
    "s1_k1":  "ml_struct_tmp_s1_k1",
    "s1_k7":  "ml_struct_tmp_s1ms",
    "s1_k14": "ml_struct_tmp_s1ms_k14",
}
tmp_dir = os.path.join(calib_dir, tmp_dir_map.get(mode, f"ml_struct_tmp_{mode}"))

out_csv_map = {
    "s2_k1":  "comparison_structure_metrics_s2_k1.csv",
    "s2_k7":  "comparison_structure_metrics_s2ms.csv",
    "s2_k14": "comparison_structure_metrics_s2ms_k14.csv",
    "s1_k1":  "comparison_structure_metrics_s1_k1.csv",
    "s1_k7":  "comparison_structure_metrics_s1ms.csv",
    "s1_k14": "comparison_structure_metrics_s1ms_k14.csv",
}
out_csv = os.path.join(calib_dir, out_csv_map.get(mode, f"comparison_structure_metrics_{mode}.csv"))

files = sorted(glob.glob(os.path.join(tmp_dir, "tc_*.csv")))
if not files:
    print(f"No per-case CSVs found in {tmp_dir}/")
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
n_cases = df.groupby(["config", "sample_id"]).ngroups
print(f"Aggregated {n_cases} cases → {len(df)} rows from {len(files)} files")
df.to_csv(out_csv, index=False)
print(f"Wrote → {out_csv}")

print("\nMethod means:")
for method in df["method"].unique():
    sub = df[df["method"] == method]
    print(f"  {method:<20}  SS_rld={sub['ss_rld'].mean():+.3f}  "
          f"SS_angle={sub['ss_angle'].mean():+.3f}  "
          f"Chamfer={sub['chamfer_cm'].mean():.3f} cm  (N={len(sub)})")
PYEOF "$MODE"
