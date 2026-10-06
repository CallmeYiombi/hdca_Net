#!/usr/bin/env python3
"""Cardinality-matched broad mask (specificity control).

Reviewer concern this answers: the direct-target mask differs from the broad
HCDT mask in two ways at once -- it is *narrower*, and it comes from a
*different annotation source* (GDSC putative targets rather than HCDT).  A gain
from the direct-target mask could therefore be specificity or annotation source.

This control holds the source fixed and varies only the breadth: for every drug
we keep its HCDT row but subsample it, uniformly at random, down to exactly the
number of genes the direct-target mask assigns that drug.  Same source, same
cardinality as the direct-target arm, different content.

Reading of the outcome:
  * recovery stays near the broad arm  -> the gain is annotation content, not breadth
  * recovery rises toward direct-target -> breadth alone accounts for much of it

Usage:
    python scripts/build_cardinality_matched_mask.py --seed 101
writes data/matrices_gdsc12/hcdt_drug_gene_cardmatched_s<seed>.npy
"""
import argparse
import os

import numpy as np

BROAD = "hcdt_drug_gene.npy"
DIRECT = "hcdt_drug_gene_pruned.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    broad = np.load(os.path.join(args.mat_dir, BROAD))
    direct = np.load(os.path.join(args.mat_dir, DIRECT))
    assert broad.shape == direct.shape, (broad.shape, direct.shape)
    rng = np.random.default_rng(args.seed)

    out = np.zeros_like(broad)
    n_sub = n_kept = n_empty = 0
    for d in range(broad.shape[0]):
        b_idx = np.flatnonzero(broad[d])
        k = int(direct[d].sum())
        if b_idx.size == 0:
            n_empty += 1
            continue
        if k == 0 or k >= b_idx.size:
            # no direct row to match, or the broad row is already that small
            out[d, b_idx] = 1
            n_kept += 1
            continue
        pick = rng.choice(b_idx, size=k, replace=False)
        out[d, pick] = 1
        n_sub += 1

    b_sz = broad.sum(1)
    d_sz = direct.sum(1)
    o_sz = out.sum(1)
    ann = b_sz > 0
    print(f"drugs                : {broad.shape[0]}")
    print(f"  subsampled         : {n_sub}")
    print(f"  kept as-is         : {n_kept}  (no direct row, or broad already <= k)")
    print(f"  empty in broad     : {n_empty}")
    print(f"median genes/drug    : broad {np.median(b_sz[ann]):.0f} | "
          f"direct {np.median(d_sz[d_sz > 0]):.0f} | matched {np.median(o_sz[ann]):.0f}")
    both = (b_sz > 0) & (d_sz > 0)
    print(f"cardinality equals direct on {int((o_sz[both] == d_sz[both]).sum())}/"
          f"{int(both.sum())} drugs with both rows")
    # How much true-target content survives?  Averaging over all annotated drugs
    # is misleading: for most of them the broad row is already <= k, so nothing is
    # subsampled and the row is returned intact.  The control only bites on the
    # drugs that actually get subsampled -- which are exactly the wide-mask drugs
    # the MoA probe is made of.
    inter = (out * direct).sum(1)
    sub = both & (b_sz > d_sz) & (d_sz > 0)
    print(f"true targets retained (all annotated) : {inter[both].sum():.0f} of "
          f"{d_sz[both].sum():.0f}  -- inflated by the {int((b_sz[both] <= d_sz[both]).sum())} "
          f"drugs whose broad row was already small")
    print(f"true targets retained (subsampled only): {inter[sub].sum():.0f} of "
          f"{d_sz[sub].sum():.0f} over {int(sub.sum())} drugs "
          f"({100 * inter[sub].sum() / max(d_sz[sub].sum(), 1):.0f}%)")

    out_path = args.out or os.path.join(
        args.mat_dir, f"hcdt_drug_gene_cardmatched_s{args.seed}.npy")
    np.save(out_path, out.astype(broad.dtype))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
