
"""Build count-matched random drug-gene masks for the negative control.

Each sham mask keeps the number of target genes the direct-target mask assigns to a drug and
draws that many genes at random from the gene universe, excluding the drug's true targets, so
attention sparsity is matched while gene identity is randomized.

Usage: python scripts/build_random_mask.py --k 5
"""
import os, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", default="data/matrices_gdsc12/hcdt_drug_gene_pruned.npy")
    ap.add_argument("--out_dir", default="data/matrices_gdsc12")
    ap.add_argument("--k", type=int, default=5, help="number of independent realizations")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--pool", choices=["all", "targets"], default="all",
                    help="all = draw from every gene, targets = draw from the target pool only")
    args = ap.parse_args()

    m = np.load(args.pruned)
    D, G = m.shape
    counts = m.sum(1).astype(int)
    target_universe = np.where(m.sum(0) > 0)[0] if args.pool == "targets" else np.arange(G)
    print(f"pruned: {D} drugs x {G} genes | drugs with >=1 target {int((counts>0).sum())} | pool={args.pool}({len(target_universe)})")

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
        ok = int(rand.sum()) == int(m.sum())
        overlap = int((rand.astype(bool) & m.astype(bool)).sum())
        print(f"  realization {k}: saved {out} | count-matched={ok} | overlap with true targets={overlap}")

if __name__ == "__main__":
    main()
