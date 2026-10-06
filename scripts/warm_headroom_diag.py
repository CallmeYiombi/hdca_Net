"""Where does HDCA-Net lose accuracy on the warm split, and how much is recoverable?

HDCA-Net scores 0.8615 PCC on the warm split while an additive drug-plus-cell-line mean,
which models no interaction at all, scores 0.889. A model that is beaten by the additive
baseline is not failing to capture drug-cell interaction -- it is failing to reproduce the
main effects. This decomposes the warm test targets to bound what is recoverable and where.

No model or GPU needed: everything below is computed from the targets and the split.

Usage: python scripts/warm_headroom_diag.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.split import random_split  # noqa: E402

MD = "data/matrices_gdsc12"
REPORTED = {"HDCA-Net": 0.8615, "drug mean": 0.845, "drug+cell mean": 0.889,
            "DeepCDR": 0.8727, "PANCDR": 0.9283, "DRPreter": 0.9404}


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    st = pd.read_csv(os.path.join(MD, "sample_table.csv"))
    zero = set(np.load("zero_fp_idx.npy").tolist())          # the 109 corrected all-zero rows
    mask = np.load(os.path.join(MD, "hcdt_drug_gene.npy"))   # NOTE: local copy may be stale
    mask_empty = set(np.where(mask.sum(1) == 0)[0].tolist())

    tr, va, te = random_split(st, seed=42)
    trd, ted = st.iloc[tr], st.iloc[te]
    y = ted["ln_ic50"].values

    # --- variance decomposition of the warm test targets -------------------
    gmean = trd["ln_ic50"].mean()
    dmean = trd.groupby("drug_idx")["ln_ic50"].mean()
    cmean = trd.groupby("cell_idx")["ln_ic50"].mean()
    d_hat = ted["drug_idx"].map(dmean).fillna(gmean).values
    c_hat = ted["cell_idx"].map(cmean).fillna(gmean).values
    add = d_hat + c_hat - gmean

    print("warm test set: {:,} pairs\n".format(len(ted)))
    print("what the targets alone allow (fitted on the training split):")
    print(f"  drug mean only          PCC {pcc(y, d_hat):.4f}")
    print(f"  cell-line mean only     PCC {pcc(y, c_hat):.4f}")
    print(f"  additive drug+cell      PCC {pcc(y, add):.4f}   <- no interaction term at all")
    resid = y - add
    print(f"  residual (interaction+noise) share of variance: "
          f"{resid.var() / y.var() * 100:.1f}%")

    print("\nreported models:")
    for k, v in sorted(REPORTED.items(), key=lambda kv: -kv[1]):
        gap = v - pcc(y, add)
        print(f"  {k:16s} {v:.4f}   {'+' if gap >= 0 else ''}{gap:.4f} vs additive")

    # --- how much of the warm test set carries no drug-specific input ------
    print("\ndrug-side input coverage on the warm test pairs:")
    n = len(ted)
    for label, bad in (("zero fingerprint", zero),
                       ("empty drug--gene mask row", mask_empty),
                       ("zero fingerprint AND empty mask", zero & mask_empty)):
        k = int(ted["drug_idx"].isin(bad).sum())
        print(f"  {label:34s} {k:7,} pairs ({100*k/n:5.1f}%)")

    # --- ceiling if the drug-signal-less pairs were given their drug mean --
    blind = ted["drug_idx"].isin(zero & mask_empty).values
    if blind.any():
        hyb = np.where(blind, d_hat, y)          # oracle elsewhere, drug mean on blind pairs
        print(f"\nif every pair WITH drug input were predicted perfectly and the "
              f"{blind.sum():,} blind pairs\n  fell back to the drug mean, warm PCC would be "
              f"{pcc(y, hyb):.4f} -- so the blind pairs alone cap warm PCC there.")

    print("\nNOTE: hcdt_drug_gene.npy is stale locally (240 annotated vs 435 on the server),")
    print("      so the mask-derived rows above are upper bounds on the blind-pair count.")
    print("      Re-run on the server for the exact figures.")


if __name__ == "__main__":
    main()
