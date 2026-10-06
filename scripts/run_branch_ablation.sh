#!/usr/bin/env bash
# Branch ablation: retrain gene-only and pathway-only (both already exists as pruned_div03, 6 seeds).
# Cross-dataset regime, 6 seeds, using the reported model config (pruned_div03 = direct-target mask).
# Run on the server: bash scripts/run_branch_ablation.sh
set -u
ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"
CFG=configs/hdca_gdsc12_cross_pruned_div03.yaml       # reported model (direct-target) config
EVAL_DIRS="data/matrices_ccle_2015 data/matrices_gcsi_2019"
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)
ALIGNS=(gene pathway)                                  # 'both' is reused from the existing runs
mkdir -p logs
i=0
for a in "${ALIGNS[@]}"; do
  for s in "${SEEDS[@]}"; do
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    log="logs/ablation_${a}_seed${s}.log"
    echo ">> launch align=$a seed=$s gpu=$g -> $log"
    # cfg out_dir already ends in ...pruned_div03, so --tag carries only the seed number
    nohup python src/train/train_hdca.py \
        --config "$CFG" --mode cross --align "$a" \
        --eval_dirs $EVAL_DIRS \
        --seed "$s" --tag "s${s}" --gpu "$g" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    if (( i % ${#GPUS[@]} == 0 )); then wait; fi
  done
done
wait
echo "=== done. out: results/hdca_gdsc12_pruned_div03_s{1..6}/cross_dataset/{gene,pathway}/ ==="
echo "reference arm (both) already exists at results/hdca_gdsc12_pruned_div03_s{1..6}/.../gene_pathway"
echo "aggregate with: python scripts/aggregate_branch_ablation.py"
