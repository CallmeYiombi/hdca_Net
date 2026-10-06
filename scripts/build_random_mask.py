#!/usr/bin/env python3
r"""
Build the random gene masks used as the negative control.
Each sham mask keeps the per-compound gene count of the pruned (true direct-target) mask and
replaces only the identity of the genes, so sparsity is matched drug by drug and only the
biological content changes.

Nothing in the architecture changes: point hcdt_drug_gene_file in the config at one of these
npy files and retrain.

Usage: python scripts/build_random_mask.py --k 5 [--pool all|targets]
"""
import os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", default="data/matrices_gdsc12/hcdt_drug_gene_pruned.npy")
    ap.add_argument("--out_dir", default="data/matrices_gdsc12")
    ap.add_argument("--k", type=int, default=5, help="number of realizations")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--pool", choices=["all", "targets"], default="all",
                    help="all = draw from every gene, targets = draw from the target pool (a stricter control)")
    args = ap.parse_args()

    m = np.load(args.pruned)                       # (D, G) binary
    D, G = m.shape
    counts = m.sum(1).astype(int)
    target_universe = np.where(m.sum(0) > 0)[0] if args.pool == "targets" else np.arange(G)
    print(f"pruned: {D} drugs x {G} genes | {int((counts>0).sum())} compounds with >=1 target | pool={args.pool}({len(target_universe)})")

    for k in range(1, args.k + 1):
        rng = np.random.default_rng(args.seed + k)
        rand = np.zeros_like(m)
        for d in range(D):
            c = counts[d]
            if c == 0:
                continue
            true = np.where(m[d] > 0)[0]
            cand = np.setdiff1d(target_universe, true, assume_unique=False)
            if len(cand) == 0:
                continue
            chosen = rng.choice(cand, size=min(c, len(cand)), replace=False)
            rand[d, chosen] = 1
        out = os.path.join(args.out_dir, f"hcdt_drug_gene_random_{k}.npy")
        np.save(out, rand)
        ok = int(rand.sum()) == int(m.sum())       # the total gene count must match, preserving sparsity
        overlap = int((rand.astype(bool) & m.astype(bool)).sum())
        print(f"  realization {k}: saved {out} | count-matched={ok} | overlap with the true mask={overlap}")


if __name__ == "__main__":
    main()
