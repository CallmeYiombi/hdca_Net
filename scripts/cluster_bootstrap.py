"""Cluster bootstrap for the HDCA-Net minus DRPreter margin on the cross-study cohorts.

Pairs are not independent: each cell line and each compound contributes many of them, so a
pair-level interval understates the uncertainty. This resamples whole clusters instead.
Both models are scored on the same DRPreter-aligned, winsorized pairs, on the seed-averaged
predictions.

The HDCA-Net predictions must come from the reference configuration (direct-target mask,
lambda_div = 0.3); the flat `results/crosspred_*.npz` copies are a different configuration
(broad mask, lambda_div = 0). The default below points at the per-seed reference directories.

  python scripts/cluster_bootstrap.py [--boot 10000] [--hdca CCLE_GLOB GCSI_GLOB]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

CAP = 13.82
SETS = {
    "CCLE": dict(st="data/matrices_ccle_2015/sample_table.csv",
                 ic="DRPreter-main/IC_CCLE.csv",
                 hdca="server_snapshot/results/hdca_gdsc12_pruned_div03_s*/"
                      "cross_dataset/gene_pathway/crosspred_CCLE_2015_seed*.npz",
                 drp="DRPreter-main/Result_HDCA/cross_CCLE_seed*_pred.csv"),
    "gCSI": dict(st="data/matrices_gcsi_2019/sample_table.csv",
                 ic="DRPreter-main/IC_gCSI.csv",
                 hdca="server_snapshot/results/hdca_gdsc12_pruned_div03_s*/"
                      "cross_dataset/gene_pathway/crosspred_gCSI_2019_seed*.npz",
                 drp="DRPreter-main/Result_HDCA/cross_gCSI_seed*_pred.csv"),
}


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def seed_mean(paths, key):
    """Average predictions over seeds, keeping the file order of the rows."""
    out = []
    for p in sorted(paths, key=lambda x: int(re.search(r"seed(\d+)", x).group(1))):
        if p.endswith(".npz"):
            z = np.load(p)
            out.append(z["y_pred"])
        else:
            out.append(pd.read_csv(p)["y_pred"].values)
    return np.mean(np.stack(out), axis=0)


def load(name, cfg):
    st = pd.read_csv(cfg["st"])
    ic = pd.read_csv(cfg["ic"])
    ref = set(zip(ic["DepMap_ID"].astype(str), ic["Drug name"].astype(str)))
    mask = np.array([p in ref for p in zip(st["model_id"].astype(str),
                                           st["drug_name"].astype(str))])
    sub = st[mask].reset_index(drop=True)

    hd = seed_mean(glob.glob(cfg["hdca"]), "y_pred")[mask]
    dp = seed_mean(glob.glob(cfg["drp"]), "y_pred")
    if len(dp) != len(sub):
        raise SystemExit(f"{name}: DRPreter rows {len(dp)} != aligned rows {len(sub)}")
    y = np.minimum(sub["ln_ic50"].values, CAP)
    return sub, y, hd, dp


def boot(y, a, b, groups, n_boot, rng):
    """Resample whole clusters with replacement; return the PCC difference each time."""
    keys, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inv == g)[0] for g in range(len(keys))]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by_group[g] for g in pick])
        diffs[i] = pcc(a[idx], y[idx]) - pcc(b[idx], y[idx])
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--hdca", nargs=2, metavar=("CCLE_GLOB", "GCSI_GLOB"),
                    help="override the HDCA-Net prediction globs (CCLE, gCSI)")
    args = ap.parse_args()
    if args.hdca:
        SETS["CCLE"]["hdca"], SETS["gCSI"]["hdca"] = args.hdca
    rng = np.random.default_rng(0)

    for name, cfg in SETS.items():
        sub, y, hd, dp = load(name, cfg)
        print(f"  [source] {len(glob.glob(cfg['hdca']))} HDCA-Net seed files "
              f"from {cfg['hdca']}")
        obs = pcc(hd, y) - pcc(dp, y)
        print(f"\n{name}: {len(y)} aligned pairs, "
              f"{sub['model_id'].nunique()} cell lines, {sub['drug_name'].nunique()} compounds")
        print(f"  HDCA-Net {pcc(hd, y):.4f}   DRPreter {pcc(dp, y):.4f}   "
              f"observed difference {obs:+.4f}")
        for unit, g in (("pair (as published)", np.arange(len(y))),
                        ("cell line", sub["model_id"].values),
                        ("compound", sub["drug_name"].values)):
            d = boot(y, hd, dp, g, args.boot, rng)
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {unit:20s} 95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"P(diff<=0) = {float((d <= 0).mean()):.3f}")


if __name__ == "__main__":
    main()
