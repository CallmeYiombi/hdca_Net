
"""Build a causal-chain drug-pathway mask.

The curated drug-pathway annotation assigns a single pathway to most compounds, so using it
as a hard mask degenerates the pathway branch. This script instead derives the mask from the
chain the model encodes: every pathway containing one of a drug's direct target genes, unioned
with the curated annotation.

Usage: python scripts/build_path_mask_chain.py
"""
import os, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--drug_gene", default="hcdt_drug_gene_pruned.npy",
                    help="drug-gene mask the chain starts from")
    ap.add_argument("--out", default="hcdt_drug_path_chain.npy")
    ap.add_argument("--no_union", action="store_true",
                    help="use the chain only, without the curated drug-pathway annotation")
    args = ap.parse_args()

    load = lambda f: np.load(os.path.join(args.mat_dir, f))
    mdg = load(args.drug_gene) > 0
    gp  = load("gene_pathway.npy") > 0
    mdp = load("hcdt_drug_path_direct.npy") > 0

    reach = (mdg.astype(np.float32) @ gp.astype(np.float32)) > 0
    out = reach if args.no_union else (reach | mdp)
    out = out.astype(np.float32)

    n = out.sum(1); has = n > 0
    print(f"drug_gene={args.drug_gene}  D={out.shape[0]}  P={out.shape[1]}")
    print(f"  curated mask: drugs {int((mdp.sum(1)>0).sum())}/{len(n)}, "
          f"median {np.median(mdp.sum(1)[mdp.sum(1)>0]):.0f} pathways")
    print(f"  chain mask  : drugs {int(has.sum())}/{len(n)}, median {np.median(n[has]):.0f}, "
          f"IQR {np.percentile(n[has],25):.0f}-{np.percentile(n[has],75):.0f}, max {int(n.max())}")
    keep = (mdp & (out > 0)).sum() / max(mdp.sum(), 1) * 100
    print(f"  curated annotation retained: {keep:.1f}%")

    path = os.path.join(args.mat_dir, args.out)
    np.save(path, out)
    print(f"saved: {path}")

if __name__ == "__main__":
    main()
