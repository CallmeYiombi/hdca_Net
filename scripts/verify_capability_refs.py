"""Recompute the two non-learned reference points quoted in Sec. "Prediction performance".

The paragraph on compounds with no resolvable structure bounds what is recoverable on
those 109 compounds with two predictors that use no model at all:

  floor    the cell line and nothing else -- a cell-line mean fitted on the training split
  ceiling  an oracle that knows the compound -- an additive drug + cell mean. On the warm
           split the drug mean comes from training; on the unseen-drug folds the compound
           has no training rows, so it comes from the held-out pairs themselves, which is
           exactly what makes it an oracle rather than a predictor.

Both are computed on seeds 1-6, the seeds the model scores they are compared against use.
An earlier version of these numbers was computed on a single seed-42 split while the model
scores were six-seed means; running this script with --seed42 reproduces those values and
shows the size of that inconsistency.

No model, GPU or checkpoint needed: everything comes from the targets and the split.

Usage: python scripts/verify_capability_refs.py [--seed42]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data.split import random_split, drug_kfold  # noqa: E402

MD = "data/matrices_gdsc12"

# Six-seed means from results/gdsc_6seed_aligned.csv, scale "structureless".
MODELS = {"warm":  {"DeepCDR": 0.2500, "GraphDRP": 0.2570, "TGSA": 0.2504, "HDCA-Net": 0.6454},
          "drug5": {"DeepCDR": 0.2784, "GraphDRP": 0.2741, "TGSA": 0.2725, "HDCA-Net": 0.3303}}

# What the manuscript states, for the matched (seeds 1-6) reading.
PAPER = {"warm": (0.260, 0.876, 63), "drug5": (None, None, 9)}


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def warm_refs(st, zero, seed):
    tr, _, te = random_split(st, seed=seed)
    trd, ted = st.iloc[tr], st.iloc[te]
    g = trd["ln_ic50"].mean()
    cm = trd.groupby("cell_idx")["ln_ic50"].mean()
    dm = trd.groupby("drug_idx")["ln_ic50"].mean()
    sub = ted[ted["drug_idx"].isin(zero)]
    c = sub["cell_idx"].map(cm).fillna(g).values
    d = sub["drug_idx"].map(dm).fillna(g).values
    return pcc(sub["ln_ic50"], c), pcc(sub["ln_ic50"], d + c - g), len(sub)


def drug5_refs(st, zero, seed):
    fl, ce, n = [], [], []
    for _, tri, _, tei in drug_kfold(st, n_splits=5, seed=seed):
        sub = st.iloc[tei]
        sub = sub[sub["drug_idx"].isin(zero)]
        trd = st.iloc[tri]
        g = trd["ln_ic50"].mean()
        cm = trd.groupby("cell_idx")["ln_ic50"].mean()
        c = sub["cell_idx"].map(cm).fillna(g).values
        d = sub.groupby("drug_idx")["ln_ic50"].transform("mean").values
        fl.append(pcc(sub["ln_ic50"], c))
        ce.append(pcc(sub["ln_ic50"], d + c - g))
        n.append(len(sub))
    return float(np.mean(fl)), float(np.mean(ce)), int(np.mean(n))


def report(title, floor, ceil, npairs, which):
    print(f"  {title}  (n={npairs:,} pairs)")
    print(f"    floor    cell mean only       {floor:.4f}")
    print(f"    ceiling  knows the compound   {ceil:.4f}")
    for m, v in MODELS[which].items():
        print(f"    {m:10s} {v:.4f}   recovers {100 * (v - floor) / (ceil - floor):6.1f}% of the gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed42", action="store_true",
                    help="reproduce the superseded single-split values instead")
    args = ap.parse_args()
    seeds = [42] if args.seed42 else [1, 2, 3, 4, 5, 6]

    st = pd.read_csv(os.path.join(MD, "sample_table.csv"))
    zero = set(np.load("zero_fp_idx.npy").tolist())
    print(f"{len(zero)} compounds with an all-zero fingerprint, "
          f"{int(st['drug_idx'].isin(zero).sum()):,} pairs of {len(st):,}")
    print(f"seeds: {seeds}\n")

    w = np.array([warm_refs(st, zero, s) for s in seeds], float)
    report("warm split", w[:, 0].mean(), w[:, 1].mean(), int(w[:, 2].mean()), "warm")
    d = np.array([drug5_refs(st, zero, s) for s in seeds], float)
    report("drug 5-fold, compound held out", d[:, 0].mean(), d[:, 1].mean(), int(d[:, 2].mean()), "drug5")

    # Share of pairs the 109 compounds take, quoted in the same paragraph.
    shares = []
    for s in seeds:
        for _, _, _, tei in drug_kfold(st, n_splits=5, seed=s):
            sub = st.iloc[tei]
            shares.append(100 * sub["drug_idx"].isin(zero).mean())
    print(f"\n  structureless share of each unseen-drug fold: "
          f"{min(shares):.1f}%--{max(shares):.1f}%  (manuscript: 11%--29%)")


if __name__ == "__main__":
    main()
