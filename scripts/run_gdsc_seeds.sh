#!/usr/bin/env bash
# GDSC-internal (warm random split + drug 5-fold) 6-seed re-run for every model in
# Table 1. These are the only two settings in the paper still reported from a single
# training run per model; everything else (cross-study, branch, variants, MoA) is
# already six seeds.
#
# Why it matters: the drug-5-fold claim "best full-panel model on unseen drugs" rests
# on 5 folds at one seed with a fold spread of +/-0.056, while DRPreter alone was
# averaged over six seeds. Seeds 1-6 here match the seeds used for cross-study and for
# DRPreter, so Table 1 becomes seed-consistent with the rest of the paper.
#
# --seed already reseeds the SPLIT as well as initialisation
#   (train_hdca.py:254,289 / train_baselines.py:294,381 / train_pancdr.py:300,320),
#   so each seed is an independent split *and* an independent initialisation.
#
# Output paths are seed-isolated (train_baselines.py and train_pancdr.py were patched
# for this; previously only the cross split isolated by seed and parallel warm/drug5
# runs would have overwritten one another's best.pt and results CSV).
#
# Usage:
#   cd /home/genetics/yiombi/HCP_Net
#   nohup bash scripts/run_gdsc_seeds.sh > logs/gdsc_seeds_all.log 2>&1 &
#
# Then aggregate with:
#   python scripts/aggregate_gdsc_seeds.py
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

HDCA_CFG=configs/hdca_gdsc12.yaml     # lr 3e-4, 150 ep, patience 7, early stop 30,
                                      # broad mask, lambda_div=0 -- the Table 1 schedule
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)

# PANCDR: if the published run used non-default nz/d_dim/lam, re-state them here or the
# checkpoints will not correspond to the reported numbers.
PANCDR_ARGS="--nz 256 --d_dim 100 --lam 0.1"

mkdir -p logs

# ---- mask provenance gate -------------------------------------------------
# The broad HCDT mask exists in two generations: 240 annotated compounds before the
# GDSC/PubChem fallback and 435 after. Table 1's published warm/drug5 runs are dated
# 2026-08-04 16:12 and it is not recorded which generation they used. Re-running with a
# different mask would move the numbers for a reason that has nothing to do with seeds,
# so fail loudly here rather than discover it afterwards.
python - <<'EOF' || exit 1
import numpy as np, sys
m = np.load("data/matrices_gdsc12/hcdt_drug_gene.npy")
nz = m.sum(1) > 0
n, med = int(nz.sum()), float(np.median(m.sum(1)[nz]))
print(f"[mask gate] hcdt_drug_gene.npy: {n}/542 annotated, median {med:.0f} genes")
if n != 435:
    sys.exit(f"[mask gate] ABORT: expected the post-fallback mask (435 compounds), got {n}. "
             "Fetch the current mask before running, or the new seeds will not be "
             "comparable with the published seed-42 values.")
EOF

# ---- resume guard ---------------------------------------------------------
# HDCA writes to results/hdca_gdsc12_seed<s>/ and is seed-isolated even without the
# train_baselines/train_pancdr patch, so a restart should not redo work that already
# finished. Pass FORCE=1 to ignore existing outputs.
FORCE=${FORCE:-0}
done_already () {  # done_already <marker-path>
    [ "$FORCE" = "0" ] && [ -e "$1" ]
}

i=0
launch () {  # launch <gpu> <logname> <command...>
    # Do NOT set CUDA_VISIBLE_DEVICES here. The three trainers select their device
    # differently: train_hdca.py and train_pancdr.py set CUDA_VISIBLE_DEVICES from
    # --gpu themselves, but train_baselines.py does torch.device(f"cuda:{args.gpu}")
    # against the full device list. Restricting visibility from outside leaves that
    # process with a single device indexed 0, so --gpu 1 or 2 raises an invalid-device
    # error and the baseline jobs die immediately. Passing --gpu alone is what
    # run_hdca_cross_seeds.sh does and it works for all three.
    local g=$1 log=$2; shift 2
    echo ">> gpu=$g  $log"
    nohup "$@" > "logs/$log" 2>&1 &
}

for split_hdca in random drug5; do
    # train_baselines/train_pancdr call the unseen-drug split "drug"
    split_base=$([ "$split_hdca" = "drug5" ] && echo drug || echo random)

    for s in "${SEEDS[@]}"; do
        hdca_marker="results/hdca_gdsc12_seed${s}/${split_hdca}/gene_pathway"
        hdca_marker=$([ "$split_hdca" = "random" ] && echo "$hdca_marker/metrics.json" \
                                                  || echo "$hdca_marker/summary.json")
        if done_already "$hdca_marker"; then
            echo "-- skip hdca $split_hdca seed$s (already done)"
        else
            g=${GPUS[$(( i % ${#GPUS[@]} ))]}
            launch "$g" "hdca_${split_hdca}_seed${s}.log" \
                python src/train/train_hdca.py --config "$HDCA_CFG" \
                    --mode "$split_hdca" --align both --seed "$s" --tag "seed${s}" --gpu "$g"
            i=$(( i + 1 )); (( i % ${#GPUS[@]} == 0 )) && wait
        fi

        if done_already "results/baselines/results_${split_base}_seed${s}.csv"; then
            echo "-- skip baselines $split_base seed$s (already done)"
        else
            g=${GPUS[$(( i % ${#GPUS[@]} ))]}
            launch "$g" "baselines_${split_base}_seed${s}.log" \
                python src/train/train_baselines.py --split "$split_base" --seed "$s" --gpu "$g"
            i=$(( i + 1 )); (( i % ${#GPUS[@]} == 0 )) && wait
        fi

        if done_already "results/baselines/pancdr_results_${split_base}_seed${s}.csv"; then
            echo "-- skip pancdr $split_base seed$s (already done)"
        else
            g=${GPUS[$(( i % ${#GPUS[@]} ))]}
            launch "$g" "pancdr_${split_base}_seed${s}.log" \
                python src/train/train_pancdr.py --split "$split_base" --seed "$s" --gpu "$g" $PANCDR_ARGS
            i=$(( i + 1 )); (( i % ${#GPUS[@]} == 0 )) && wait
        fi
    done
done
wait

echo "=== done: 36 jobs (2 splits x 6 seeds x 3 launchers; baselines covers 3 models) ==="
echo "HDCA      -> results/hdca_gdsc12_seed{1..6}/{random,drug5}/gene_pathway/"
echo "baselines -> results/baselines/results_{random,drug}_seed{1..6}.csv"
echo "PANCDR    -> results/baselines/pancdr_results_{random,drug}_seed{1..6}.csv"
echo "sanity: the existing seed-42 values (warm PCC 0.865, drug5 0.407) should fall"
echo "        inside the new spread; if they do not, check that the config is unchanged."
