#!/usr/bin/env python3
r"""
Build the causal-chain pathway mask, the prerequisite for hard-masking the pathway branch.

hcdt_drug_path_direct.npy holds the direct drug-pathway annotation: a median of one pathway per
compound, exactly one for 72% of them and none for 150. Hard-masking on it would degenerate the
pathway branch into a one-hot lookup and make the top-10 MoA metric a database identity.

Instead, build the mask from the drug -> target -> pathway chain the paper describes:

    M_path = (M_dg_direct @ M_gp > 0)  ∪  M_dp_direct

  = the pathways containing a compound's direct target genes, unioned with its existing direct
    annotation. That gives a median of about 44 pathways per compound (IQR 19-104) and covers
    413 of 542, so a hard mask does not degenerate and the top-10 evaluation stays non-trivial.

Usage:
  python scripts/build_path_mask_chain.py                      # default, from the direct-target mask
  python scripts/build_path_mask_chain.py --drug_gene hcdt_drug_gene.npy --out hcdt_drug_path_chain_broad.npy
"""
import os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--drug_gene", default="hcdt_drug_gene_pruned.npy",
                    help="the drug-gene mask the chain starts from")
    ap.add_argument("--out", default="hcdt_drug_path_chain.npy")
    ap.add_argument("--no_union", action="store_true",
                    help="use the chain alone, without unioning the existing direct annotation")
    args = ap.parse_args()

    load = lambda f: np.load(os.path.join(args.mat_dir, f))
    mdg = load(args.drug_gene) > 0                  # (D, G)
    gp  = load("gene_pathway.npy") > 0              # (G, P)
    mdp = load("hcdt_drug_path_direct.npy") > 0     # (D, P)

    reach = (mdg.astype(np.float32) @ gp.astype(np.float32)) > 0
    out = reach if args.no_union else (reach | mdp)
    out = out.astype(np.float32)

    n = out.sum(1); has = n > 0
    print(f"drug_gene={args.drug_gene}  D={out.shape[0]}  P={out.shape[1]}")
    print(f"  existing M_dp : {int((mdp.sum(1)>0).sum())}/{len(n)} compounds, "
          f"median {np.median(mdp.sum(1)[mdp.sum(1)>0]):.0f} pathway")
    print(f"  chain mask    : {int(has.sum())}/{len(n)} compounds, median {np.median(n[has]):.0f}, "
          f"IQR {np.percentile(n[has],25):.0f}-{np.percentile(n[has],75):.0f}, max {int(n.max())}")
    keep = (mdp & (out > 0)).sum() / max(mdp.sum(), 1) * 100
    print(f"  existing annotation retained: {keep:.1f}%")

    path = os.path.join(args.mat_dir, args.out)
    np.save(path, out)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
