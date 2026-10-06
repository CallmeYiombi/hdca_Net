"""Re-score the 144-drug DGIdb panel with a lambda-matched broad arm.

The published comparison pairs the direct-target runs (prn_s*, lambda_div = 0.3)
against base_s* (lambda_div = 0), while the 16-drug probe pairs the same direct-target
runs against brd_s* (lambda_div = 0.3). Calling both "the broad mask" understates the
comparison, because the two differ in the regularization weight as well as the mask.

This script reports the panel under both broad arms, per seed, together with the
coverage-matched restriction (drugs whose gene branch is active under both masks) and
per-seed drug-level McNemar.

Usage: python scripts/moa_dgidb_lambda_matched.py [--out results/moa_dgidb_lambda_matched.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from moa_expand_chembl import build_masks, load_targets_dgidb, hit, mcnemar  # noqa: E402

MATRICES, DB = "data/matrices_gdsc12", "data/interactions.tsv"
TOPN, MAX_PATH_SIZE, TOPK = 2, 15, 10


def load(tag):
    return [np.load(p) for p in sorted(glob.glob(f"results/interpret_hdca/{tag}_s*/p_gene_align.npy"))]


def rate(P, masks, low2idx):
    h = sum(hit(P, low2idx[d], e, TOPK) for d, e in masks.items())
    return 100.0 * h / len(masks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the results to this JSON file")
    args = ap.parse_args()

    idm = json.load(open(os.path.join(MATRICES, "id_maps.json")))
    sym2idx = idm["gene_symbol_to_idx"]
    gp = np.load(os.path.join(MATRICES, "gene_pathway.npy"))
    psize = (gp > 0).sum(0)
    st = pd.read_csv(os.path.join(MATRICES, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    tgt, _ = load_targets_dgidb(DB, set(st["drug_name_lower"].unique()), sym2idx, TOPN)
    masks = build_masks(tgt, sym2idx, gp, psize, MAX_PATH_SIZE)

    direct = load("prn")
    arms = {"base_s* (lambda=0, as published)": load("base"),
            "brd_s*  (lambda-matched, 0.3)": load("brd")}

    # coverage-matched panel: gene branch active under both masks
    act_d = np.abs(direct[0]).max(axis=1) > 0
    print(f"full panel: {len(masks)} drugs\n")
    dump = dict(panel=dict(n_drugs=len(masks), topn=TOPN, max_path_size=MAX_PATH_SIZE,
                           topk=TOPK), arms={})

    for label, broad in arms.items():
        act_b = np.abs(broad[0]).max(axis=1) > 0
        both = {d: e for d, e in masks.items() if act_d[low2idx[d]] and act_b[low2idx[d]]}
        print(f"--- broad arm = {label} ---")
        for name, mk in (("full", masks), (f"coverage-matched (n={len(both)})", both)):
            br = [rate(P, mk, low2idx) for P in broad]
            dr = [rate(P, mk, low2idx) for P in direct]
            gains = [mcnemar(D, B, mk, low2idx, TOPK) for D, B in zip(direct, broad)]
            d_only = np.array([g[1] for g in gains]); b_only = np.array([g[0] for g in gains])
            ps = [g[2] for g in gains]
            print(f"  {name:28s} broad {np.mean(br):5.1f} +/- {np.std(br, ddof=1):4.1f}%   "
                  f"direct {np.mean(dr):5.1f} +/- {np.std(dr, ddof=1):4.1f}%   "
                  f"gain +{np.mean(dr)-np.mean(br):4.1f}pp")
            print(f"  {'':28s} McNemar {d_only.mean():4.1f} +/- {d_only.std(ddof=1):3.1f} vs "
                  f"{b_only.mean():4.1f} +/- {b_only.std(ddof=1):3.1f}   "
                  f"p {min(ps):.1e} to {max(ps):.1e}")
            dump["arms"].setdefault(label, {})[name] = dict(
                n_seeds=len(direct),
                broad_mean=float(np.mean(br)), broad_sd=float(np.std(br, ddof=1)),
                direct_mean=float(np.mean(dr)), direct_sd=float(np.std(dr, ddof=1)),
                gain_pp=float(np.mean(dr) - np.mean(br)),
                mcnemar_direct_only_mean=float(d_only.mean()),
                mcnemar_direct_only_sd=float(d_only.std(ddof=1)),
                mcnemar_broad_only_mean=float(b_only.mean()),
                mcnemar_broad_only_sd=float(b_only.std(ddof=1)),
                p_min=float(min(ps)), p_max=float(max(ps)),
                broad_per_seed=[float(x) for x in br],
                direct_per_seed=[float(x) for x in dr])
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dump, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
