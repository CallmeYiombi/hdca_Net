#!/usr/bin/env python3
"""Rebuild the cross-study table from the source per-pair predictions.

Recomputes every row from the npz/csv under the snapshot root rather than from a cached
summary, so the configuration each row comes from is explicit in the run directory name.
Scale matches Table 1 of the main text: winsorize at 13.82 and restrict to the pairs shared
with DRPreter.
"""
import argparse
import csv
import glob
import os
import statistics as st

import numpy as np
import pandas as pd
from scipy import stats

CAP = 13.82
COHORTS = {"CCLE": ("CCLE_2015", "matrices_ccle_2015", "IC_CCLE.csv"),
           "gCSI": ("gCSI_2019", "matrices_gcsi_2019", "IC_gCSI.csv")}

# runs to include: name -> (directory glob, description)
HDCA_RUNS = {
    "HDCA-Net (reference)": "results/hdca_gdsc12_pruned_div03_s*",
    "HDCA-Net-broad":       "results/hdca_gdsc12_broad_div03_s*",
    "HDCA-Net-dense":       "results/hdca_gdsc12_nomask_div03_s*",
    "HDCA-Net-sham":        "results/hdca_gdsc12_random_mask_*",
    "HDCA-Net-hardpath":    "results/hdca_gdsc12_pathchain_s*",
    "cardinality-matched":  "results/hdca_gdsc12_cardmatched_d*_s*",
    "diag2 (old Table 1)":  "results/hdca_gdsc12_diag2_seed*",
}
BASELINES = ("GraphDRP", "DeepCDR", "TGSA", "PANCDR")


def mask_for(root, cohort):
    _, mdir, ic = COHORTS[cohort]
    icp = os.path.join(root, "DRPreter-main", "Data_HDCA", ic)
    ref = set(zip(pd.read_csv(icp)["DepMap_ID"].astype(str),
                  pd.read_csv(icp)["Drug name"].astype(str)))
    tbl = pd.read_csv(os.path.join(root, "data", mdir, "sample_table.csv"))
    m = np.array([p in ref for p in zip(tbl["model_id"].astype(str),
                                        tbl["drug_name"].astype(str))])
    return m, tbl


def score(yt, yp, mask):
    yt = np.minimum(yt, CAP)
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    return np.corrcoef(yt, yp)[0, 1], stats.spearmanr(yt, yp).statistic


def collect(root, cohort):
    pset, _, _ = COHORTS[cohort]
    mask, _ = mask_for(root, cohort)
    out = {}

    for name, pattern in HDCA_RUNS.items():
        vals = []
        for d in sorted(glob.glob(os.path.join(root, pattern))):
            f = os.path.join(d, "cross_dataset", "gene_pathway",
                             f"crosspred_{pset}_seed*.npz")
            for g in sorted(glob.glob(f)):
                if "_gdscnorm_" in g:          # skip the standardization-sensitivity arm
                    continue
                z = np.load(g)
                if len(z["y_true"]) != len(mask):
                    print(f"  [warn] row count mismatch, skipping: {g}")
                    continue
                vals.append(score(z["y_true"], z["y_pred"], mask))
        if vals:
            out[name] = vals

    low = pset.split("_")[0].lower()
    for b in BASELINES:
        vals = []
        for g in sorted(glob.glob(os.path.join(
                root, "results/baselines/cross_dataset", b,
                f"crosspred_matrices_{low}_*_seed*.npz"))):
            if "_gdscnorm_" in g:
                continue
            z = np.load(g)
            if len(z["y_true"]) != len(mask):
                continue
            vals.append(score(z["y_true"], z["y_pred"], mask))
        if vals:
            out[b] = vals

    vals = []
    for g in sorted(glob.glob(os.path.join(
            root, "DRPreter-main/Result_HDCA", f"cross_{cohort}_seed*_pred.csv"))):
        df = pd.read_csv(g)
        vals.append(score(df["y_true"].values, df["y_pred"].values, None))  # the reference set
    if vals:
        out["DRPreter"] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="server_snapshot")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for coh in ("CCLE", "gCSI"):
        m, _ = mask_for(args.root, coh)
        print(f"\n=== {coh} === (winsorized at {CAP}, pairs shared with DRPreter "
              f"{int(m.sum())}/{len(m)})")
        print(f"{'run':24s} {'PCC':>18s} {'SCC':>18s}   n")
        res = collect(args.root, coh)
        for name, vals in sorted(res.items(), key=lambda x: -st.mean([v[0] for v in x[1]])):
            p = [v[0] for v in vals]
            s_ = [v[1] for v in vals]
            sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
            print(f"{name:24s} {st.mean(p):>9.4f}±{sd(p):<8.4f} "
                  f"{st.mean(s_):>9.4f}±{sd(s_):<8.4f}  {len(p)}")
            rows.append({"cohort": coh, "run": name, "n": len(p),
                         "pcc": round(st.mean(p), 4), "pcc_sd": round(sd(p), 4),
                         "scc": round(st.mean(s_), 4), "scc_sd": round(sd(s_), 4)})
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
