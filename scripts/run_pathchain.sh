#!/usr/bin/env bash
# Train and interpret the variant whose pathway branch is hard-masked by the causal-chain mask.
#
#   current (soft gamma=2): 96.5% of the attention mass falls outside HCDT, so the branch is not
#   faithful by construction; this variant (hard) drops unannotated pathways from the softmax denominator.
#
# Prerequisite (local or server): python scripts/build_path_mask_chain.py
#
# Examples:
#   PILOT=1 GPU=3 bash scripts/run_pathchain.sh                # seed 1 pilot only (~1.5h)
#   GPU=3 SEEDS="1 2 3 4 5 6" bash scripts/run_pathchain.sh    # six seeds in sequence (~9h)
#   GPU=0 SEEDS="1 2" bash scripts/run_pathchain.sh &          # split across three GPUs
#   GPU=1 SEEDS="3 4" bash scripts/run_pathchain.sh &
#   GPU=3 SEEDS="5 6" bash scripts/run_pathchain.sh &
set -u
ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"
GPU="${GPU:-3}"
CFG=configs/hdca_gdsc12_cross_pathchain.yaml
EVAL="data/matrices_ccle_2015 data/matrices_gcsi_2019"
if [ "${PILOT:-0}" = "1" ]; then SEEDS="1"; else SEEDS="${SEEDS:-1 2 3 4 5 6}"; fi
mkdir -p logs

# build the mask first if it is missing
[ -f data/matrices_gdsc12/hcdt_drug_path_chain.npy ] || python scripts/build_path_mask_chain.py

for s in $SEEDS; do
  OUT="results/hdca_gdsc12_pathchain_s${s}"
  echo "=========== pathchain seed $s (gpu $GPU) ==========="
  python src/train/train_hdca.py --config "$CFG" --mode cross --align both \
      --eval_dirs $EVAL --gpu "$GPU" --seed "$s" --tag "s${s}" \
      2>&1 | tee "logs/pathchain_s${s}.log"

  BEST="${OUT}/cross_dataset/gene_pathway/best.pt"
  if [ ! -f "$BEST" ]; then echo "!! best.pt missing: $BEST -- did training fail?"; continue; fi

  echo ">>> interpret seed $s"
  python src/analysis/interpret_hdca.py --config "$CFG" --model_path "$BEST" \
      --align both --out_dir "results/interpret_hdca/pathchain_s${s}" --gpu "$GPU" \
      2>&1 | tee "logs/interpret_pathchain_s${s}.log"

  echo ">>> check that the pathway constraint took effect (mass should be close to 1.0)"
  python scripts/check_path_attn_mass.py \
      --interp "results/interpret_hdca/pathchain_s${s}" --mask hcdt_drug_path_chain.npy
done

echo
echo "=== summary: cross-study accuracy (raw; soft comparator = CCLE 0.589 / gCSI 0.676) ==="
for s in $SEEDS; do
  f="results/hdca_gdsc12_pathchain_s${s}/cross_dataset/gene_pathway/cross_metrics.json"
  [ -f "$f" ] && python -c "
import json,sys
for r in json.load(open('$f')):
    print(f\"  seed $s {r['dataset']:10s} PCC {r['pcc']:.4f}  SCC {r['spearman']:.4f}  RMSE {r['rmse']:.3f}\")"
done
echo "=== score the word-boundary MoA comparison locally with src/analysis/moa_compare.py ==="
