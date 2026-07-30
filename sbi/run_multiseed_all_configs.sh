#!/bin/bash
# run_multiseed_all_configs.sh
#
# Generates K=7 multi-seed datasets for all 4 active SBI configs.
# Runs configs sequentially; each uses --threads 8 for within-config parallelism.
#
# Estimated runtime: ~5h per config × 4 = ~20h total
#
# Usage (from repo root):
#   bash sbi/run_multiseed_all_configs.sh [K]
# K defaults to 7.

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
SCRIPT=sbi/julia/generate_multiseed_dataset.jl
K=${1:-7}
THREADS=6

CONFIGS=(
    sbi/configs/sbi_config_A.json
    sbi/configs/sbi_config_B2.json
    sbi/configs/sbi_config_C.json
    sbi/configs/sbi_config_D2.json
)

echo "======================================================="
echo "Multi-seed dataset generation  K=$K  threads=$THREADS"
echo "Configs: ${#CONFIGS[@]}"
echo "Started: $(date)"
echo "======================================================="

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo ">>> Config: $cfg  ($(date))"
    $JULIA --threads $THREADS $SCRIPT "$cfg" $K
    echo "<<< Done: $cfg  ($(date))"
done

echo ""
echo "======================================================="
echo "All configs complete: $(date)"
echo "======================================================="
