#!/usr/bin/env bash
# Negative control: retrain K sham models on a random-gene mask, then interpret them.
# The architecture is untouched; only the mask file and out_dir in the config are swapped (sed).
# Prerequisite: python scripts/build_random_mask.py --k 5  (writes hcdt_drug_gene_random_{1..K}.npy)
# Run on the server: bash scripts/run_negative_control.sh
set -u
ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"
GPU="${GPU:-3}"
K="${K:-5}"
BASE=configs/hdca_gdsc12_cross_pruned_div03.yaml     # reported direct-target config; only the mask is swapped
EVAL="data/matrices_ccle_2015 data/matrices_gcsi_2019"
mkdir -p logs configs

for k in $(seq 1 "$K"); do
  CFG="configs/_random_mask_${k}.yaml"
  sed -e "s|hcdt_drug_gene_pruned.npy|hcdt_drug_gene_random_${k}.npy|" \
      -e "s|results/hdca_gdsc12_pruned_div03|results/hdca_gdsc12_random_mask_${k}|" \
      "$BASE" > "$CFG"
  echo ">> [sham $k/$K] train (gpu $GPU)  cfg=$CFG"
  python src/train/train_hdca.py --config "$CFG" --mode cross --align both \
      --eval_dirs $EVAL --gpu "$GPU" 2>&1 | tee "logs/random_mask_${k}.log"
  best="results/hdca_gdsc12_random_mask_${k}/cross_dataset/gene_pathway/best.pt"
  echo ">> [sham $k/$K] interpret"
  python src/analysis/interpret_hdca.py --config "$CFG" --model_path "$best" \
      --align both --out_dir "results/interpret_hdca/random_mask_${k}" --gpu "$GPU" \
      2>&1 | tee "logs/interpret_random_mask_${k}.log"
done
echo "=== done. aggregate with: python scripts/eval_negative_control.py --k $K ==="
