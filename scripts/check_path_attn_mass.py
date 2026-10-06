#!/usr/bin/env python3
r"""
Quantify how far the pathway branch actually respects the HCDT constraint.

  - attention mass: softmax mass on annotated or masked pathways (mean and median over compounds)
  - top-10 agreement: the share of the top-10 pathways that lie inside the mask
  - enrichment: the ratio to a uniform distribution

Lets the soft-prior model (additive, gamma=2) and the hard-mask model be compared on one scale.

Usage:
  python scripts/check_path_attn_mass.py --interp results/interpret_hdca/pruned_div03 \
      --mask hcdt_drug_path_direct.npy
  python scripts/check_path_attn_mass.py --interp results/interpret_hdca/pathchain_s1 \
      --mask hcdt_drug_path_chain.npy
"""
import os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp", required=True, help="interpret output folder; must contain path_attn.npy")
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--mask", default="hcdt_drug_path_direct.npy",
                    help="the drug-pathway mask that model actually used")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    pa = np.load(os.path.join(args.interp, "path_attn.npy"))        # (D, P)
    m  = np.load(os.path.join(args.mat_dir, args.mask)) > 0         # (D, P)
    assert pa.shape == m.shape, f"shape mismatch {pa.shape} vs {m.shape}"

    has = m.sum(1) > 0
    idx = np.where(has)[0]
    mass = np.array([pa[d][m[d]].sum() for d in idx])
    unif = m.sum(1)[idx] / pa.shape[1]
    top = np.argsort(pa, 1)[:, ::-1][:, :args.topk]
    frac = np.array([m[d][top[d]].mean() for d in idx])

    print(f"interp = {args.interp}")
    print(f"mask   = {args.mask}  ({len(idx)}/{len(has)} compounds, "
          f"median {np.median(m.sum(1)[idx]):.0f} pathways each)")
    print(f"  attention mass on mask : mean {mass.mean():.4f}  median {np.median(mass):.4f}")
    print(f"  compounds with mass > 0.5 : {(mass > 0.5).mean()*100:.1f}%   > 0.9: {(mass > 0.9).mean()*100:.1f}%")
    print(f"  top-{args.topk} inside the mask  : mean {frac.mean()*100:.1f}%")
    print(f"  enrichment over uniform   : median {np.median(mass/unif):.1f}x")


if __name__ == "__main__":
    main()
