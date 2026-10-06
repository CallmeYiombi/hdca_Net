#!/usr/bin/env bash
# Forward-only pass over the existing 6-seed GDSC checkpoints to dump per-pair
# predictions. Nothing is retrained.
#
# Why: Table 1 is not a like-for-like comparison. HDCA-Net, GraphDRP, DeepCDR and TGSA
# are scored on all 542 compounds, while PANCDR drops the compounds whose molecular
# graph cannot be built and DRPreter is limited to what its featurizer covers -- both of
# which land on the same 433 compounds. The 109 excluded compounds are 18.9% of the warm
# test pairs and 12.7-25.3% of each unseen-drug fold, and they carry all-zero
# fingerprints, so the four full-panel models are penalised for attempting predictions
# the other two never make. This is the same alignment the cross-study table already
# applies (Sec. "Cross-study winsorization and matched evaluation"); it has simply never
# been applied to Table 1.
#
# PANCDR and DRPreter need no pass: their reported numbers are already subset-scored.
#
# Usage:
#   cd /home/genetics/yiombi/HCP_Net
#   nohup bash scripts/run_gdsc_dump_preds.sh > logs/gdsc_dump_all.log 2>&1 &
# Then:
#   python scripts/align_gdsc_subset.py
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

HDCA_CFG=configs/hdca_gdsc12.yaml
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)

mkdir -p logs
i=0
launch () {   # launch <gpu> <logname> <command...>
    local g=$1 log=$2; shift 2
    echo ">> gpu=$g  $log"
    nohup "$@" > "logs/$log" 2>&1 &
}

for split_hdca in random drug5; do
    split_base=$([ "$split_hdca" = "drug5" ] && echo drug || echo random)
    for s in "${SEEDS[@]}"; do
        g=${GPUS[$(( i % ${#GPUS[@]} ))]}
        launch "$g" "dump_hdca_${split_hdca}_seed${s}.log" \
            python src/train/train_hdca.py --config "$HDCA_CFG" \
                --mode "$split_hdca" --align both --seed "$s" --tag "seed${s}" \
                --gpu "$g" --eval_only
        i=$(( i + 1 )); (( i % ${#GPUS[@]} == 0 )) && wait

        g=${GPUS[$(( i % ${#GPUS[@]} ))]}
        launch "$g" "dump_baselines_${split_base}_seed${s}.log" \
            python src/train/train_baselines.py --split "$split_base" \
                --seed "$s" --gpu "$g" --eval_only
        i=$(( i + 1 )); (( i % ${#GPUS[@]} == 0 )) && wait
    done
done
wait

echo "=== done: 24 forward-only jobs ==="
echo "HDCA      -> results/hdca_gdsc12_seed{1..6}/{random,drug5}/gene_pathway/**/pred.npz"
echo "baselines -> results/baselines/{random,drug_fold}/<model>/seed{1..6}/**/pred.npz"
echo "next: python scripts/align_gdsc_subset.py"
echo "check: the recomputed full-panel PCC must reproduce results/gdsc_6seed.csv exactly,"
echo "       otherwise the dumped predictions do not correspond to the reported runs."
