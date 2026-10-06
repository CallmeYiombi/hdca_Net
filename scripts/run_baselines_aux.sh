#!/usr/bin/env bash
# Baseline-fairness control: give the three deep-learning baselines the same drug--gene
# annotation HDCA-Net receives, and ask whether the cross-study gap is that difference
# rather than the architecture.
#
#   gated    x * M_dg -- the expression gated by the drug's targets, which is the signal
#            HDCA-Net's gene branch forms. Under a hard mask with one annotated target the
#            two are identical, and 195 of the 421 annotated compounds have exactly one,
#            so this is the strong form of the control.
#   concat   M_dg alone -- target identity without the expression it gates. Weak contrast.
#
# The mask is dropped entry-wise at the same rate HDCA-Net uses (--mask_dropout 0.1), so
# the control is not handed a cleaner mask than the model it is compared against.
#
# One invocation trains GraphDRP, DeepCDR and TGSA in sequence, so
#   2 arms x 6 seeds = 12 invocations = 36 model runs.
# Results land in results/baselines_aux{gated,concat}/ and never touch the published
# results/baselines/ tree.
#
# Usage (server):
#   nohup bash scripts/run_baselines_aux.sh > logs/baseline_aux.log 2>&1 &
#   SPLIT=random ARMS=gated bash scripts/run_baselines_aux.sh   # warm split, for the
#                                                              # structureless-compound test
# Then:
#   python scripts/eval_baseline_aux.py --out results/baseline_aux.json
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

EVAL_DIRS="data/matrices_ccle_2015 data/matrices_gcsi_2019"
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)
ARMS="${ARMS:-gated concat}"
SPLIT="${SPLIT:-cross}"        # cross | random  (random gives the warm-split per-pair dumps)
MASK="${MASK:-hcdt_drug_gene_pruned.npy}"
FORCE="${FORCE:-0}"
mkdir -p logs

# ---- the mask must be the generation the paper describes ---------------------
python - "$MASK" <<'EOF' || exit 1
import numpy as np, sys
f = "data/matrices_gdsc12/" + sys.argv[1]
m = np.load(f); nz = m.sum(1) > 0
n, med = int(nz.sum()), float(np.median(m.sum(1)[nz]))
single = int((m.sum(1) == 1).sum())
print(f"[mask gate] {sys.argv[1]}: {n}/542 annotated, median {med:.0f} genes, "
      f"{single} single-target compounds")
if n != 421:
    sys.exit(f"[mask gate] ABORT: expected 421 annotated compounds, got {n}.")
EOF

i=0
for arm in $ARMS; do
  for s in "${SEEDS[@]}"; do
    out="results/baselines_aux${arm}"
    marker="${out}/results_${SPLIT}_seed${s}.csv"
    if [ "$FORCE" = "0" ] && [ -e "$marker" ]; then
      echo "-- skip ${arm}/seed${s} (already done)"
      continue
    fi
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    log="logs/baselines_aux_${arm}_${SPLIT}_seed${s}.log"
    echo ">> [aux=$arm split=$SPLIT] seed=$s gpu=$g -> $log"
    if [ "$SPLIT" = "cross" ]; then
      EXTRA="--eval_dirs $EVAL_DIRS"
    else
      EXTRA=""
    fi
    nohup python src/train/train_baselines.py \
        --split "$SPLIT" $EXTRA \
        --aux "$arm" --drug_gene_file "$MASK" --mask_dropout 0.1 \
        --seed "$s" --gpu "$g" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    (( i % ${#GPUS[@]} == 0 )) && { echo "   ...wave wait ($i)"; wait; }
  done
done
wait

echo "=== done. aggregate with scripts/eval_baseline_aux.py ==="
