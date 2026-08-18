#!/bin/bash
# run_extend_multiseed_all_configs.sh
#
# Extends the K=7 multi-seed dataset to K=20 for all 4 configs by simulating
# 13 additional seeds (7008–7020) per sample and merging with existing K=7 data.
#
# Prerequisites:
#   - sbi/data/multi_seed/config_*/per_sample/  must exist (from generate_multiseed_dataset.jl)
#
# Outputs (per config, alongside existing tensors/ and per_sample/):
#   sbi/data/multi_seed/config_*/per_sample_ext/   — 13 new seed checkpoints
#   sbi/data/multi_seed/config_*/tensors_k20/       — K=20 merged tensors
#
# Usage (from repo root):
#   bash sbi/run_extend_multiseed_all_configs.sh [K_ext]
# K_ext defaults to 13  (7 + 13 = 20 total seeds)

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
SCRIPT=sbi/julia/extend_multiseed_dataset.jl
K_EXT=${1:-13}
THREADS=6

CONFIGS=(
    sbi/configs/sbi_config_A.json
    sbi/configs/sbi_config_B2.json
    sbi/configs/sbi_config_C.json
    sbi/configs/sbi_config_D2.json
)

LOG=sbi/logs/multiseed_extension.log
mkdir -p sbi/logs

echo "=======================================================" | tee -a $LOG
echo "Multi-seed extension  K_ext=$K_EXT  (total K=$((7+K_EXT)))  threads=$THREADS" | tee -a $LOG
echo "Configs: ${#CONFIGS[@]}" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG
echo "=======================================================" | tee -a $LOG

for cfg in "${CONFIGS[@]}"; do
    echo "" | tee -a $LOG
    echo ">>> Config: $cfg  ($(date))" | tee -a $LOG
    $JULIA --threads $THREADS $SCRIPT "$cfg" $K_EXT 2>&1 | tee -a $LOG
    echo "<<< Done: $cfg  ($(date))" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=======================================================" | tee -a $LOG
echo "All configs complete: $(date)" | tee -a $LOG
echo "=======================================================" | tee -a $LOG
