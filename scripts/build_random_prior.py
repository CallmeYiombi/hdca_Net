#!/usr/bin/env python3
r"""Randomized counterparts of the two priors the existing sham control leaves intact.

scripts/build_random_mask.py randomizes only M_dg, the drug--gene mask. Two further
matrices carry biology into the forward pass, and a reader is entitled to ask whether the
result survives randomizing them too:

  M_dp  (D x P)  hcdt_drug_path_direct.npy -- the pathway branch's additive prior
  M_gp  (G x P)  gene_pathway.npy          -- the gene branch's pathway bottleneck

Both randomizations preserve degree, so the control differs from the reference in which
relations are asserted and in nothing else -- not in how many:

  M_dp  per drug, keep the number of annotated pathways and redraw their identity from
        the pathway universe, excluding that drug's true pathways.
  M_gp  permute the gene labels. This preserves every pathway's size exactly and the whole
        multiset of gene degrees, and destroys only the gene-to-pathway correspondence.
        Redrawing entries independently would not: pathway sizes span three orders of
        magnitude here, and a control that flattened them would be easier, not harder.

Usage:
  python scripts/build_random_prior.py --k 5
Then train with configs that point hcdt_drug_path_file / gene_pathway_file at the output.
"""
import argparse
import os

import numpy as np


def randomize_rows(m, rng, exclude_true=True):
    """Redraw each row's entries, keeping the row's count."""
    out = np.zeros_like(m)
    universe = np.arange(m.shape[1])
    for i in range(m.shape[0]):
        c = int(m[i].sum())
        if c == 0:
            continue
        true = np.where(m[i] > 0)[0]
        cand = np.setdiff1d(universe, true) if exclude_true else universe
        if len(cand) == 0:
            continue
        out[i, rng.choice(cand, size=min(c, len(cand)), replace=False)] = 1
    return out


def permute_rows(m, rng):
    """Relabel the rows. Column sums and the row-degree multiset are preserved exactly."""
    return m[rng.permutation(m.shape[0])]


def report(name, true, rand):
    overlap = int((true.astype(bool) & rand.astype(bool)).sum())

    def cmp(axis):
        a, b = true.sum(axis), rand.sum(axis)
        if np.array_equal(a, b):
            return "identical"
        # a permutation moves degrees between rows but keeps the multiset
        return "same multiset" if np.array_equal(np.sort(a), np.sort(b)) else "differ"

    print(f"  {name}: nnz {int(true.sum())} -> {int(rand.sum())}, "
          f"row degrees {cmp(1)}, col degrees {cmp(0)}, "
          f"entries shared with the true matrix {overlap} "
          f"({100 * overlap / max(int(true.sum()), 1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--k", type=int, default=5, help="number of realizations")
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()

    dp = np.load(os.path.join(args.mat_dir, "hcdt_drug_path_direct.npy"))
    gp = np.load(os.path.join(args.mat_dir, "gene_pathway.npy"))
    print(f"M_dp {dp.shape} nnz={int(dp.sum())} | M_gp {gp.shape} nnz={int(gp.sum())}")

    for k in range(1, args.k + 1):
        rng = np.random.default_rng(args.seed + k)
        dp_r = randomize_rows(dp, rng)
        gp_r = permute_rows(gp, rng)
        np.save(os.path.join(args.mat_dir, f"hcdt_drug_path_random_{k}.npy"), dp_r)
        np.save(os.path.join(args.mat_dir, f"gene_pathway_random_{k}.npy"), gp_r)
        print(f"realization {k}")
        report("M_dp", dp, dp_r)
        report("M_gp", gp, gp_r)


if __name__ == "__main__":
    main()
