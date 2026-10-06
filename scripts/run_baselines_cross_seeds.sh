#!/usr/bin/env bash
# Retrain the baselines (GraphDRP/DeepCDR/TGSA and PANCDR) cross-dataset over six seeds,
# so their error bars are symmetric with the six-seed HDCA-Net and DRPreter runs.
#
# Layout (compatible with the aggregator winsorize_cross_table.py):
#   - npz files share the model dir: results/baselines/cross_dataset/<model>/crosspred_<ds>_seed<s>.npz
#   - best.pt is isolated per seed: results/baselines/cross_dataset/<model>/seed<s>/best.pt (avoids races)
#   - one train_baselines.py call runs GraphDRP/DeepCDR/TGSA in sequence; PANCDR has its own script.
#
# Round-robin over three GPUs (0/1/2): three jobs per wave, wait, then the next wave.
#
# Usage:
#   cd /home/genetics/yiombi/HCP_Net
#   nohup bash scripts/run_baselines_cross_seeds.sh > logs/baselines_cross_all.log 2>&1 &
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

EVAL_DIRS="data/matrices_ccle_2015 data/matrices_gcsi_2019"
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)
mkdir -p logs

# -- 0) remove the old single-seed (seed42) npz so the aggregator picks up exactly seeds 1-6 --
echo ">> clearing old crosspred npz so that only seeds 1-6 remain"
find results/baselines/cross_dataset -name "crosspred_*.npz" -delete 2>/dev/null || true

# -- 1) GraphDRP / DeepCDR / TGSA via train_baselines.py (three models per call) --
i=0
for s in "${SEEDS[@]}"; do
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    log="logs/baselines_cross_seed${s}.log"
    echo ">> [baselines] seed=$s gpu=$g → $log"
    nohup python src/train/train_baselines.py \
        --split cross --eval_dirs $EVAL_DIRS \
        --seed "$s" --gpu "$g" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    (( i % ${#GPUS[@]} == 0 )) && { echo "   ...waiting for this wave ($i)"; wait; }
done
wait
echo "=== GraphDRP/DeepCDR/TGSA six seeds done ==="

# -- 2) PANCDR via its own script (hyperparameters kept at the argparse defaults used in training) --
#   If nz/d_dim/lam differed from the defaults during training, pass the same --nz --d_dim --lam here.
i=0
for s in "${SEEDS[@]}"; do
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    log="logs/pancdr_cross_seed${s}.log"
    echo ">> [PANCDR] seed=$s gpu=$g → $log"
    nohup python src/train/train_pancdr.py \
        --split cross --eval_dirs $EVAL_DIRS \
        --seed "$s" --gpu "$g" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    (( i % ${#GPUS[@]} == 0 )) && { echo "   ...waiting for this wave ($i)"; wait; }
done
wait
echo "=== PANCDR six seeds done ==="

echo ""
echo "=== all baselines, six seeds, done ==="
echo "output: results/baselines/cross_dataset/<model>/crosspred_<ds>_seed{1..6}.npz (12 per model)"
echo "check: each log tail should show CCLE and gCSI PCC within the seed-to-seed spread"
echo "next: copy the npz files locally and aggregate with winsorize_cross_table.py --subset"
