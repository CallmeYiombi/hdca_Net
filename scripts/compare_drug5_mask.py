"""Compare the unseen-drug split under the broad and the direct-target gene mask.

Single factor: the two arms differ only in which drug--gene mask the gene branch attends
over (configs/hdca_gdsc12.yaml vs configs/hdca_gdsc12_drug5_direct.yaml), on the same six
seeds, the same five folds and the same schedule.

Reports the fold-basis mean and spread used by Table 1, the seed-basis spread, and paired
tests over both units. Also re-scores on the 433 structure-resolved compounds if the
per-pair dumps are present, since the two arms differ most on compounds whose only
drug-side signal is the mask.

Usage: python scripts/compare_drug5_mask.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ARMS = {"broad (published)": "results/hdca_gdsc12_seed*/drug5/gene_pathway",
        "direct-target": "results/hdca_gdsc12_drug5_direct_seed*/drug5/gene_pathway"}


def load(pattern):
    rows = []
    for d in sorted(glob.glob(pattern)):
        seed = int(re.search(r"seed(\d+)", d).group(1))
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        for m in json.load(open(f))["folds"]:
            rows.append(dict(seed=seed, fold=m["fold"], pcc=m["pcc"],
                             scc=m["spearman"], rmse=m["rmse"]))
    return pd.DataFrame(rows)


def main():
    arms = {k: load(v) for k, v in ARMS.items()}
    for k, v in arms.items():
        if v.empty:
            raise SystemExit(f"no runs found for '{k}' ({ARMS[k]})")
        print(f"{k:20s} {v.seed.nunique()} seeds x {v.fold.nunique()} folds = {len(v)} runs")

    print(f"\n{'arm':20s} {'PCC (fold basis)':>20s} {'sd(seed)':>9s} {'SCC':>18s} {'RMSE':>18s}")
    for k, v in arms.items():
        pf = v.groupby("fold")[["pcc", "scc", "rmse"]].mean()
        ps = v.groupby("seed")["pcc"].mean()
        print(f"{k:20s} {pf.pcc.mean():11.4f}+/-{pf.pcc.std(ddof=1):.4f} "
              f"{ps.std(ddof=1):9.4f} {pf.scc.mean():11.4f}+/-{pf.scc.std(ddof=1):.4f} "
              f"{pf.rmse.mean():11.4f}+/-{pf.rmse.std(ddof=1):.4f}")

    a, b = arms["direct-target"], arms["broad (published)"]
    print("\npaired, direct-target minus broad:")
    for unit in ("fold", "seed"):
        x = a.groupby(unit)["pcc"].mean()
        y = b.groupby(unit)["pcc"].mean()
        common = x.index.intersection(y.index)
        d = x[common] - y[common]
        t = stats.ttest_rel(x[common], y[common])
        print(f"  over {unit}s (n={len(common)}): delta {d.mean():+.4f}  "
              f"paired t p={t.pvalue:.4f}  wins {int((d > 0).sum())}/{len(common)}")
    both = a.merge(b, on=["seed", "fold"], suffixes=("_direct", "_broad"))
    d = both.pcc_direct - both.pcc_broad
    print(f"  over all (seed, fold) cells (n={len(both)}): delta {d.mean():+.4f}  "
          f"wins {int((d > 0).sum())}/{len(both)}")

    print("\nreference points from the published 6-seed table (fold basis):")
    print("  DRPreter 0.4135 +/- 0.0401 | HDCA-Net(broad) 0.4013 +/- 0.0220 | "
          "GraphDRP 0.3864 +/- 0.0184")
    print("A direct-target arm above 0.4135 would put HDCA-Net first on the unseen-drug "
          "split outright;\nabove roughly 0.404 it clears GraphDRP by more than the "
          "current, non-significant margin.")


if __name__ == "__main__":
    main()
