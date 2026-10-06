#!/usr/bin/env bash
# Retrain HDCA-Net cross-dataset over six seeds, matching DRPreter's seed 1-6 so the error bars are symmetric.
#
# Each seed writes to its own out_dir (results/hdca_gdsc12_diag2_seed<s>/), and run_cross dumps
# crosspred_<pset>_seed<s>.npz on exit.
# Round-robin over three GPUs (0/1/2): one job each, wait for the wave, then start the next.
#
# Usage:
#   cd /home/genetics/yiombi/HCP_Net
#   bash scripts/run_hdca_cross_seeds.sh            # foreground, waits per wave
#   nohup bash scripts/run_hdca_cross_seeds.sh > logs/hdca_cross_all.log 2>&1 &   # all seeds in background
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

CFG=configs/hdca_gdsc12_cross_diag.yaml
EVAL_DIRS="data/matrices_ccle_2015 data/matrices_gcsi_2019"
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)

mkdir -p logs
i=0
for s in "${SEEDS[@]}"; do
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    log="logs/hdca_cross_seed${s}.log"
    echo ">> launch seed=$s  gpu=$g  → $log"
    # cfg out_dir already ends in ...diag2, so --tag carries only seed<n> to avoid duplicating it
    nohup python src/train/train_hdca.py \
        --config "$CFG" --mode cross --align both \
        --eval_dirs $EVAL_DIRS \
        --seed "$s" --tag "seed${s}" --gpu "$g" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    # once one job per GPU is running, wait for this wave to finish
    if (( i % ${#GPUS[@]} == 0 )); then
        echo "   ...waiting for this wave (seeds so far: $i)"
        wait
    fi
done
wait
echo "=== all seeds done. out_dir: results/hdca_gdsc12_diag2_seed{1..6}/cross_dataset/gene_pathway/ ==="
echo "check: each log tail should show CCLE PCC around 0.588 and gCSI around 0.681, within seed spread"
