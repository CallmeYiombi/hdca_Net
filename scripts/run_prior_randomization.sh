#!/usr/bin/env bash
# Full prior randomization: how far does the negative control reach?
#
# The published sham randomizes only M_dg, the drug--gene mask. Two arms extend it, so the
# reader can see which prior carries the mechanism recovery:
#
#   drugside   M_dg + M_dp randomized  (both drug-side priors; the gene->pathway map is true)
#   allprior   M_dg + M_dp + M_gp      (nothing biological left in the forward pass)
#
# Every randomization preserves degree (scripts/build_random_prior.py), so each arm differs
# from the reference in which relations are asserted, not in how many. Architecture, schedule
# and seeds are untouched -- only the three matrix paths and out_dir change.
#
# Prerequisites:
#   python scripts/build_random_mask.py  --k 5     # hcdt_drug_gene_random_{1..5}.npy
#   python scripts/build_random_prior.py --k 5     # hcdt_drug_path_random_*, gene_pathway_random_*
#
# Usage (server):
#   nohup bash scripts/run_prior_randomization.sh > logs/prior_random_all.log 2>&1 &
# Then:
#   python scripts/eval_negative_control.py --k 5   # extend with the new out_dirs
set -u

ROOT=/home/genetics/yiombi/HCP_Net
cd "$ROOT"

BASE=configs/hdca_gdsc12_cross_pruned_div03.yaml
EVAL="data/matrices_ccle_2015 data/matrices_gcsi_2019"
GPUS=(0 1 2)
K="${K:-5}"
ARMS="${ARMS:-drugside allprior}"
FORCE="${FORCE:-0}"
mkdir -p logs configs

# ---- inputs must exist before anything is scheduled ------------------------
python - "$K" <<'EOF' || exit 1
import os, sys
k = int(sys.argv[1])
need = ([f"data/matrices_gdsc12/hcdt_drug_gene_random_{i}.npy" for i in range(1, k + 1)]
        + [f"data/matrices_gdsc12/hcdt_drug_path_random_{i}.npy" for i in range(1, k + 1)]
        + [f"data/matrices_gdsc12/gene_pathway_random_{i}.npy" for i in range(1, k + 1)])
missing = [f for f in need if not os.path.exists(f)]
if missing:
    sys.exit("[gate] ABORT: run build_random_mask.py and build_random_prior.py first; "
             f"missing {len(missing)}, e.g. {missing[0]}")
print(f"[gate] all {len(need)} randomized matrices present")
EOF

i=0
for arm in $ARMS; do
  for k in $(seq 1 "$K"); do
    out="results/hdca_gdsc12_prior_${arm}_${k}"
    marker="$out/cross_dataset/gene_pathway/best.pt"
    if [ "$FORCE" = "0" ] && [ -e "$marker" ]; then
      echo "-- skip $arm/$k (already done)"
      continue
    fi
    cfg="configs/_prior_${arm}_${k}.yaml"
    python - "$BASE" "$cfg" "$arm" "$k" "$out" <<'EOF'
import sys, yaml
base, out_cfg, arm, k, out_dir = sys.argv[1:6]
c = yaml.safe_load(open(base))
c["hcdt_drug_gene_file"] = f"hcdt_drug_gene_random_{k}.npy"
c["hcdt_drug_path_file"] = f"hcdt_drug_path_random_{k}.npy"
if arm == "allprior":
    c["gene_pathway_file"] = f"gene_pathway_random_{k}.npy"
c["out_dir"] = out_dir
yaml.safe_dump(c, open(out_cfg, "w"), sort_keys=False)
print(f"[cfg] {out_cfg}: M_dg+M_dp randomized" + (", M_gp permuted" if arm == "allprior" else ""))
EOF

    g=${GPUS[$(( i % ${#GPUS[@]} ))]}
    echo ">> gpu=$g  $arm realization $k"
    nohup python src/train/train_hdca.py --config "$cfg" --mode cross --align both \
        --eval_dirs $EVAL --gpu "$g" > "logs/prior_${arm}_${k}.log" 2>&1 &
    i=$(( i + 1 ))
    (( i % ${#GPUS[@]} == 0 )) && wait
  done
done
wait

# ---- interpretability read-out, one per finished run ----------------------
for arm in $ARMS; do
  for k in $(seq 1 "$K"); do
    cfg="configs/_prior_${arm}_${k}.yaml"
    best="results/hdca_gdsc12_prior_${arm}_${k}/cross_dataset/gene_pathway/best.pt"
    [ -e "$best" ] || { echo "!! missing $best, skipping interpret"; continue; }
    echo ">> interpret $arm/$k"
    python src/analysis/interpret_hdca.py --config "$cfg" --model_path "$best" \
        --align both --out_dir "results/interpret_hdca/prior_${arm}_${k}" --gpu 0 \
        > "logs/interpret_prior_${arm}_${k}.log" 2>&1
  done
done

echo "=== done. Compare against the M_dg-only sham in Table 5. ==="
