# HDCA-Net

Reference implementation of **HDCA-Net (Hierarchical Drug--Cell Cross-Alignment Network)**, a
mechanism-encoded model for anti-cancer drug response prediction. Drug--gene attention is
restricted to curated drug targets by a masked softmax and projected onto pathways by a
parameter-free membership map, so prediction and mechanistic read-out share a single forward
pass.

This repository contains the model, the training and evaluation pipeline, the baselines used
for comparison, and the analysis scripts that produce the reported tables and figures.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`rdkit` is needed only to rebuild Morgan fingerprints, and `seaborn` only for the
interpretability heatmaps.

## Data

Response and expression data are not redistributed here. Obtain them from their sources and
build the matrices:

```bash
# GDSC1 + GDSC2 training matrices
python src/data/build_gdsc12_matrices.py
python src/data/build_drug_graphs.py    # resolves SMILES; also builds the graphs PANCDR needs
python src/data/build_drug_fp.py        # rebuilds drug_fp.npy from those SMILES
python scripts/build_pruned_mask.py     # direct-target gene mask used by the reported model

# External evaluation sets (CCLE, gCSI) via PharmacoGx
Rscript scripts/download_pharmacodb.R
Rscript scripts/extract_pharmacodb.R
python src/data/build_cross_eval_matrices.py
```

This writes `data/matrices_gdsc12/` and `data/matrices_{ccle_2015,gcsi_2019}/`, each holding a
sample table together with the expression, fingerprint and prior matrices.

The fingerprints written by `build_gdsc12_matrices.py` are mostly empty: its SMILES lookup
misses most compounds. `build_drug_graphs.py` resolves SMILES properly (PubChem with a
property-name fallback, then ChEMBL) into `drug_smiles.csv`, and `build_drug_fp.py` rebuilds
`drug_fp.npy` from it. Both are required to reproduce the reported runs, which use 433/542
non-zero fingerprints; the remaining 109 compounds keep zero vectors. `build_drug_graphs.py`
also writes the atom-feature and adjacency arrays that the PANCDR baseline loads.

## Training

```bash
# Reported model: direct-target gene mask, cross-study evaluation
python src/train/train_hdca.py --config configs/hdca_gdsc12_cross_pruned_div03.yaml \
    --mode cross --align both --seed 1 --tag s1 \
    --eval_dirs data/matrices_ccle_2015 data/matrices_gcsi_2019

# Broad HCDT gene mask, for the mask-breadth comparison
python src/train/train_hdca.py --config configs/hdca_gdsc12_cross_diag.yaml --mode cross --align both ...

# Random split and unseen-drug folds
python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml --mode random
python src/train/train_hdca.py --config configs/hdca_gdsc12.yaml --mode drug5

# Baselines
python src/train/train_baselines.py --model graphdrp --mode cross ...
python src/train/train_pancdr.py --mode cross ...
```

`--align` selects the branches (`gene`, `pathway`, `both`) and drives the branch ablation.
`--eval_only` reloads `best.pt`, skips training and dumps per-pair predictions as
`crosspred_*.npz`.

## Evaluation

External IC50 values recomputed by PharmacoGx include extrapolated outliers outside the GDSC
training support; they are winsorized at that cap before any metric is computed.

```bash
python src/data/diag_cross_outliers.py                 # quantify the extrapolated values
python src/analysis/winsorize_cross_table.py \
    --model_dirs results/<run>/cross_dataset/gene_pathway \
    --drpreter_dir <drpreter_results> --subset --cap 13.82 \
    --out results/cross_table.csv
```

## Interpretability

```bash
# Per-drug alignment scores, MoA tables and heatmaps
python src/analysis/interpret_hdca.py --config configs/hdca_gdsc12_cross_pruned_div03.yaml \
    --model_path results/<run>/cross_dataset/gene_pathway/best.pt \
    --align both --out_dir results/interpret/<run>

# MoA agreement against an independent target database (DGIdb or ChEMBL)
python scripts/moa_expand_chembl.py --db_type dgidb --db data/interactions.tsv \
    --broad <broad_scores.npy> --direct <direct_scores.npy> --n_perm 10000

# Candidate-set-corrected comparison (percentile rank, enrichment, MRR)
python scripts/rank_metrics.py --hdca results/interpret/<run>/p_gene_align.npy
```

## Controls and ablations

```bash
# Randomized-mask negative control: count-matched sham masks, retrained from scratch
python scripts/build_random_mask.py --k 5
python scripts/eval_negative_control.py --k 5

# Pathway branch constrained by the causal-chain mask instead of an additive prior
python scripts/build_path_mask_chain.py
python src/train/train_hdca.py --config configs/hdca_gdsc12_cross_pathchain.yaml --mode cross --align both ...
python scripts/check_path_attn_mass.py --interp <interpret_dir> --mask hcdt_drug_path_chain.npy

# Branch ablation (gene-only / pathway-only / both)
python scripts/aggregate_branch_ablation.py
```

## Layout

```
src/models/      HDCA-Net and the baseline architectures
src/data/        matrix construction, datasets, splits
src/train/       training entry points
src/analysis/    interpretability, evaluation tables, figures
scripts/         controls, ablations and metric utilities
configs/         one YAML per reported configuration
```

## Notes on reproduction

Runs are seeded through `--seed`, and every reported number is the mean over six seeds unless
stated otherwise. `--tag` keeps per-seed outputs in separate directories. Metrics computed
before winsorization are not on the same scale as the reported tables and are used only for
comparisons internal to a single experiment.
