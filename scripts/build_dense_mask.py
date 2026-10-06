
"""Build the all-ones drug-gene mask used by the unrestricted-attention ablation.

With gate_mode: none the gene attention already ignores the mask, but the indicator in
the gene branch still switches the branch off for compounds whose mask row is empty.
Filling the mask keeps the branch active for every compound, so the ablation differs
from the reported model in exactly one respect: whether drug-gene attention is
restricted to annotated targets.

Usage: python scripts/build_dense_mask.py
"""
import os, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--like", default="hcdt_drug_gene.npy",
                    help="mask whose shape and dtype are copied")
    ap.add_argument("--out", default="hcdt_drug_gene_dense.npy")
    args = ap.parse_args()

    ref = np.load(os.path.join(args.mat_dir, args.like), mmap_mode="r")
    dense = np.ones(ref.shape, dtype=ref.dtype)
    out = os.path.join(args.mat_dir, args.out)
    np.save(out, dense)
    print(f"saved {out}  shape={dense.shape}  dtype={dense.dtype}")
    print(f"  every one of the {dense.shape[0]} compounds now attends over all "
          f"{dense.shape[1]} genes")

if __name__ == "__main__":
    main()
