"""Aggregate the baseline-fairness control on the scale Table 2 uses.

Compares each deep-learning baseline with and without the drug--gene annotation, on the
winsorized, DRPreter-aligned cross-study pairs, so the rows sit beside HDCA-Net's .773/.761
without a scale caveat.

Reads the per-pair dumps the training script writes:
  results/baselines/cross_dataset/<model>/crosspred_<set>_seed<k>.npz        (fingerprint only)
  results/baselines_aux<arm>/cross_dataset/<model>/crosspred_<set>_seed<k>.npz

Usage (server): python scripts/eval_baseline_aux.py --out results/baseline_aux.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CAP = 13.82
MODELS = ("GraphDRP", "DeepCDR", "TGSA")
SETS = {
    "CCLE": dict(tag="matrices_ccle_2015", st="data/matrices_ccle_2015/sample_table.csv",
                 ic="DRPreter-main/IC_CCLE.csv"),
    "gCSI": dict(tag="matrices_gcsi_2019", st="data/matrices_gcsi_2019/sample_table.csv",
                 ic="DRPreter-main/IC_gCSI.csv"),
}
ARMS = {"fingerprint only": "results/baselines",
        "+ mask (gated)":   "results/baselines_auxgated",
        "+ mask (concat)":  "results/baselines_auxconcat"}


def aligned_mask(cfg):
    st = pd.read_csv(cfg["st"])
    ic = pd.read_csv(cfg["ic"])
    ref = set(zip(ic["DepMap_ID"].astype(str), ic["Drug name"].astype(str)))
    return np.array([p in ref for p in zip(st["model_id"].astype(str),
                                           st["drug_name"].astype(str))])


def score(root, model, cfg, mask):
    """Per-seed winsorized PCC/SCC on the aligned pairs."""
    # canonical layout is cross_dataset/<model>/; a flattened <model>/ copy is also accepted
    files = []
    for sub in (os.path.join("cross_dataset", model), model):
        files = glob.glob(os.path.join(root, sub, f"crosspred_{cfg['tag']}_seed*.npz"))
        if files:
            break
    out = []
    for f in sorted(files, key=lambda x: int(re.search(r"seed(\d+)", x).group(1))):
        z = np.load(f)
        yt, yp = z["y_true"], z["y_pred"]
        if len(yt) != len(mask):
            print(f"  [warn] {f}: {len(yt)} rows vs mask {len(mask)} — skipped")
            continue
        yt, yp = np.minimum(yt[mask], CAP), yp[mask]
        out.append((float(np.corrcoef(yp, yt)[0, 1]), float(spearmanr(yp, yt).statistic)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline_aux.json")
    args = ap.parse_args()

    masks = {k: aligned_mask(v) for k, v in SETS.items()}
    for k, m in masks.items():
        print(f"{k}: {int(m.sum())} aligned pairs of {len(m)}")
    print()

    result = {}
    header = f"{'model':10} {'arm':18}" + "".join(f"{k+' PCC':>16}{k+' SCC':>16}" for k in SETS)
    print(header)
    for model in MODELS:
        for arm, root in ARMS.items():
            row, cells = {}, []
            for k, cfg in SETS.items():
                v = score(root, model, cfg, masks[k])
                if not v:
                    cells += [f"{'--':>16}", f"{'--':>16}"]
                    continue
                a = np.array(v)
                row[k] = dict(n_seeds=len(v),
                              pcc=a[:, 0].mean(), pcc_sd=a[:, 0].std(ddof=1) if len(v) > 1 else 0.0,
                              scc=a[:, 1].mean(), scc_sd=a[:, 1].std(ddof=1) if len(v) > 1 else 0.0)
                cells += [f"{a[:,0].mean():10.4f}+/-{a[:,0].std(ddof=1) if len(v)>1 else 0:4.3f}",
                          f"{a[:,1].mean():10.4f}+/-{a[:,1].std(ddof=1) if len(v)>1 else 0:4.3f}"]
            if row:
                result[f"{model} | {arm}"] = row
            print(f"{model:10} {arm:18}" + "".join(cells))
        print()

    json.dump(result, open(args.out, "w"), indent=1)
    print(f"written to {args.out}")
    print("\nHDCA-Net for reference (Table 2): CCLE 0.773/0.795, gCSI 0.761/0.814")


if __name__ == "__main__":
    main()
