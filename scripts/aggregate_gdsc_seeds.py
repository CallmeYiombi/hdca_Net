"""Aggregate the 6-seed GDSC-internal runs into the two panels of Table 1.

Reports both dispersions explicitly, because the published table conflated them: the
deep-learning rows carried a fold-to-fold spread while the DRPreter row carried a
seed-to-seed spread taken after fold-averaging (DRPreter-main/hdca/aggregate_hdca.py:3).

  * warm split: mean +/- sd over seeds.
  * drug 5-fold: fold-level mean +/- sd over the five folds (seeds averaged within each
    fold) AND seed-level sd (folds averaged within each seed), so the table can state
    which one it prints.

The three writers use different schemas, which this normalises:
  train_baselines.py  split,model,fold,best_ep,rmse,mae,r2,pcc,spearman   (fold "-" when warm)
  train_pancdr.py     rmse,...,model,split[,fold],best_epoch              (no fold column when warm,
                                                                          and split is "drug", not "drug_fold")
  train_hdca.py       metrics.json (warm) / summary.json {"folds": [...]} (drug5)

Runs on partial output, so it can be called while the drug-5-fold half is still training.

Usage: python scripts/aggregate_gdsc_seeds.py [--out results/gdsc_6seed.csv]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SPLIT_ALIAS = {"drug": "drug_fold", "drug5": "drug_fold", "random": "random"}


def row(split, model, seed, fold, m):
    return dict(split=SPLIT_ALIAS.get(split, split), model=model, seed=seed, fold=fold,
                pcc=m.get("pcc"), scc=m.get("spearman"), rmse=m.get("rmse"))


def hdca_rows():
    out = []
    for d in sorted(glob.glob("results/hdca_gdsc12_seed*/")):
        seed = int(re.search(r"seed(\d+)", d).group(1))
        f = os.path.join(d, "random", "gene_pathway", "metrics.json")
        if os.path.exists(f):
            out.append(row("random", "HDCA-Net", seed, None, json.load(open(f))))
        f = os.path.join(d, "drug5", "gene_pathway", "summary.json")
        if os.path.exists(f):
            for m in json.load(open(f)).get("folds", []):
                out.append(row("drug5", "HDCA-Net", seed, m.get("fold"), m))
    return out


def csv_rows():
    out = []
    for f in sorted(glob.glob("results/baselines/results_*_seed*.csv") +
                    glob.glob("results/baselines/pancdr_results_*_seed*.csv")):
        seed_from_name = int(re.search(r"seed(\d+)", f).group(1))
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            d = r.to_dict()
            fold = d.get("fold")
            if fold in ("-", "", None) or (isinstance(fold, float) and np.isnan(fold)):
                fold = None
            out.append(row(d.get("split", "random"),
                           d.get("model", "PANCDR"),
                           int(d.get("seed", seed_from_name)),
                           fold, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gdsc_6seed.csv")
    args = ap.parse_args()

    df = pd.DataFrame(hdca_rows() + csv_rows())
    if df.empty:
        raise SystemExit("no 6-seed GDSC runs found -- run scripts/run_gdsc_seeds.sh first")

    out = []
    for (split, model), g in df.groupby(["split", "model"]):
        if split == "random":
            per_seed = g.groupby("seed")["pcc"].mean()
            out.append(dict(split=split, model=model, n_seeds=g["seed"].nunique(),
                            pcc=per_seed.mean(), sd_seed=per_seed.std(ddof=1),
                            sd_fold=np.nan, scc=g["scc"].mean(), rmse=g["rmse"].mean()))
        else:
            per_fold = g.groupby("fold")["pcc"].mean()   # seeds averaged inside a fold
            per_seed = g.groupby("seed")["pcc"].mean()   # folds averaged inside a seed
            out.append(dict(split=split, model=model, n_seeds=g["seed"].nunique(),
                            pcc=per_fold.mean(),
                            sd_seed=per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
                            sd_fold=per_fold.std(ddof=1) if len(per_fold) > 1 else np.nan,
                            scc=g["scc"].mean(), rmse=g["rmse"].mean()))

    res = pd.DataFrame(out).sort_values(["split", "pcc"], ascending=[True, False])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False)

    fmt = lambda v: "    n/a" if (v is None or np.isnan(v)) else f"{v:.4f}"
    for split, g in res.groupby("split"):
        print(f"\n=== {split} ===")
        print(f"{'model':12s} {'seeds':>5s} {'PCC':>7s} {'sd(seed)':>9s} {'sd(fold)':>9s} "
              f"{'SCC':>7s} {'RMSE':>7s}")
        for _, r in g.iterrows():
            print(f"{r.model:12s} {int(r.n_seeds):5d} {r.pcc:7.4f} {fmt(r.sd_seed):>9s} "
                  f"{fmt(r.sd_fold):>9s} {r.scc:7.4f} {r.rmse:7.4f}")
    print(f"\nsaved {args.out}")
    print("Table 1 prints one dispersion per panel: sd(seed) for warm, sd(fold) for "
          "unseen-drug. Published single-run values were warm PCC 0.865 (HDCA-Net) and "
          "drug5 0.407; they should fall inside the new spreads.")


if __name__ == "__main__":
    main()
