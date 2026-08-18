#!/bin/bash
# run_multiseed_all_configs_v2.sh
#
# Generates K=7 multi-seed datasets for all 4 SBI v2 configs (skin_depth_cm=0.2).
# Runs configs sequentially; each uses --threads 6 for within-config parallelism.
# Output goes to sbi/data/multi_seed_v2/{config_A,config_B2,config_C,config_D2}/
#
# Estimated runtime: ~3h per config × 4 = ~12h total
#
# Usage (from repo root):
#   bash sbi/run_multiseed_all_configs_v2.sh [K]
# K defaults to 7.

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
SCRIPT=sbi/julia/generate_multiseed_dataset.jl
K=${1:-7}
THREADS=6

CONFIGS=(
    sbi/configs_v2/sbi_config_A_v2.json
    sbi/configs_v2/sbi_config_B2_v2.json
    sbi/configs_v2/sbi_config_C_v2.json
    sbi/configs_v2/sbi_config_D2_v2.json
)

echo "======================================================="
echo "Multi-seed v2 dataset generation  K=$K  threads=$THREADS"
echo "skin_depth_cm=0.2  output: sbi/data/multi_seed_v2/"
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
echo "All v2 configs complete: $(date)"
echo "======================================================="
