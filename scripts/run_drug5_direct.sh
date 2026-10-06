#!/usr/bin/env bash
# Unseen-drug split with the direct-target gene mask, six seeds.
#
# Single factor against the published drug-5-fold runs: only the gene mask changes
# (configs/hdca_gdsc12_drug5_direct.yaml). The schedule is deliberately left alone --
# the published runs stop at a mean best epoch of 10 out of 24, so they are not
# budget-limited and a schedule change would confound the comparison.
#
# Cost: a drug5 job is 5 folds and took 52-92 min (mean 67) in the 6-seed run, so
# 6 jobs over 3 GPUs is roughly 2.5 hours wall clock, longer if the box is shared.
#
# Usage:
#   cd /home/genetics/yiombi/HCP_Net
#   nohup bash scripts/run_drug5_direct.sh > logs/drug5_direct_all.log 2>&1 &
# Then:
#   python scripts/compare_drug5_mask.py
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

CFG=configs/hdca_gdsc12_drug5_direct.yaml
GPUS=(0 1 2)
SEEDS=(1 2 3 4 5 6)

mkdir -p logs

# ---- mask provenance gate -------------------------------------------------
# The comparison is only meaningful against the published broad-mask runs if both masks
# are the generation the paper describes: 435 annotated compounds for the broad mask and
# 421 for the direct-target one.
python - <<'EOF' || exit 1
import numpy as np, sys
for f, want in (("data/matrices_gdsc12/hcdt_drug_gene.npy", 435),
                ("data/matrices_gdsc12/hcdt_drug_gene_pruned.npy", 421)):
    m = np.load(f); nz = m.sum(1) > 0
    n, med = int(nz.sum()), float(np.median(m.sum(1)[nz]))
    print(f"[mask gate] {f.split('/')[-1]}: {n}/542 annotated, median {med:.0f} genes")
    if n != want:
        sys.exit(f"[mask gate] ABORT: expected {want} annotated compounds, got {n}.")
EOF

FORCE=${FORCE:-0}
i=0
for s in "${SEEDS[@]}"; do
    marker="results/hdca_gdsc12_drug5_direct_seed${s}/drug5/gene_pathway/summary.json"
    if [ "$FORCE" = "0" ] && [ -e "$marker" ]; then
        echo "-- skip seed$s (already done)"
        continue
    fi
    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    echo ">> gpu=$g  drug5_direct_seed${s}"
    nohup python src/train/train_hdca.py --config "$CFG" \
        --mode drug5 --align both --seed "$s" --tag "seed${s}" --gpu "$g" \
        > "logs/drug5_direct_seed${s}.log" 2>&1 &
    i=$(( i + 1 ))
    (( i % ${#GPUS[@]} == 0 )) && wait
done
wait

echo "=== done ==="
echo "out -> results/hdca_gdsc12_drug5_direct_seed{1..6}/drug5/gene_pathway/"
echo "next: python scripts/compare_drug5_mask.py"
