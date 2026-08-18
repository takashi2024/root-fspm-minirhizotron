#!/bin/bash
# run_extend_multiseed_all_configs_v2.sh
#
# Extends the v2 K=7 multi-seed dataset to K=14 for all 4 configs by simulating
# 7 additional seeds (7008-7014) per sample.
#
# Prerequisites:
#   sbi/data/multi_seed_v2/config_*/per_sample/  must exist
#
# Outputs:
#   sbi/data/multi_seed_v2/config_*/per_sample_ext/  — 7 new seed checkpoints
#   sbi/data/multi_seed_v2/config_*/tensors_k14/      — K=14 merged tensors

JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia
SCRIPT=sbi/julia/extend_multiseed_dataset.jl
K_EXT=${1:-7}
THREADS=6

CONFIGS=(
    sbi/configs_v2/sbi_config_A_v2.json
    sbi/configs_v2/sbi_config_B2_v2.json
    sbi/configs_v2/sbi_config_C_v2.json
    sbi/configs_v2/sbi_config_D2_v2.json
)

LOG=sbi/logs/multiseed_extension_v2.log
mkdir -p sbi/logs

echo "=======================================================" | tee -a $LOG
echo "Multi-seed v2 extension  K_ext=$K_EXT  (total K=$((7+K_EXT)))  threads=$THREADS" | tee -a $LOG
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
