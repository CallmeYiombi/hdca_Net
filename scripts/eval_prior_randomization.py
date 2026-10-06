"""Score the prior-randomization arms on the same 144-drug panel as the published control.

The published negative control randomizes M_dg only. scripts/run_prior_randomization.sh
adds two arms that randomize more of the prior, and this scores all of them against the
same DGIdb panel and the same metric, so the rows are directly comparable:

  reference   prn_s*             true M_dg, M_dp, M_gp
  sham        random_mask_*      M_dg randomized              (the published control)
  drugside    prior_drugside_*   M_dg + M_dp randomized
  allprior    prior_allprior_*   M_dg + M_dp + M_gp randomized

Expected pathways are always built from the TRUE gene--pathway matrix, whatever the model
was trained with: the question is whether a model given false biology still ranks the
pathway that real biology expects.

Accuracy is read from each run's cross_metrics.json. Those PCCs are raw -- not winsorized
or pair-aligned -- so they are comparable across the arms here but not with Table 2.

Run this on the server, where the interpretability outputs live, and keep the JSON:
  python scripts/eval_prior_randomization.py --out results/prior_randomization.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moa_expand_chembl import build_masks, load_targets_dgidb, hit  # noqa: E402

MATRICES, DB = "data/matrices_gdsc12", "data/interactions.tsv"
TOPN, MAX_PATH_SIZE, TOPK = 2, 15, 10

ARMS = {
    "reference (true prior)":        "results/interpret_hdca/prn_s*",
    "sham: M_dg":                    "results/interpret_hdca/random_mask_*",
    "drugside: M_dg+M_dp":           "results/interpret_hdca/prior_drugside_*",
    "allprior: M_dg+M_dp+M_gp":      "results/interpret_hdca/prior_allprior_*",
}
RUNDIR = {
    "reference (true prior)":   "results/hdca_gdsc12_pruned_div03_s*",
    "sham: M_dg":               "results/hdca_gdsc12_random_mask_*",
    "drugside: M_dg+M_dp":      "results/hdca_gdsc12_prior_drugside_*",
    "allprior: M_dg+M_dp+M_gp": "results/hdca_gdsc12_prior_allprior_*",
}


def rate(P, masks, low2idx):
    return 100.0 * sum(hit(P, low2idx[d], e, TOPK) for d, e in masks.items()) / len(masks)


def accuracy(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, "cross_dataset/gene_pathway/cross_metrics.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f)):
            out.setdefault(r["dataset"], []).append(r["pcc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/prior_randomization.json")
    args = ap.parse_args()

    idm = json.load(open(os.path.join(MATRICES, "id_maps.json")))
    gp = np.load(os.path.join(MATRICES, "gene_pathway.npy"))       # true, on purpose
    st = pd.read_csv(os.path.join(MATRICES, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    tgt, _ = load_targets_dgidb(DB, set(st["drug_name_lower"].unique()),
                                idm["gene_symbol_to_idx"], TOPN)
    masks = build_masks(tgt, idm["gene_symbol_to_idx"], gp, (gp > 0).sum(0), MAX_PATH_SIZE)
    print(f"panel: {len(masks)} drugs, top-{TOPK}\n")

    result = {}
    for name, pattern in ARMS.items():
        files = sorted(glob.glob(os.path.join(pattern, "p_gene_align.npy")))
        if not files:
            print(f"{name:28s} -- no runs found ({pattern})")
            continue
        rates = [rate(np.load(f), masks, low2idx) for f in files]
        acc = accuracy(RUNDIR[name])
        result[name] = dict(n=len(rates), moa_mean=float(np.mean(rates)),
                            moa_sd=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                            moa_per_run=rates,
                            accuracy={k: [float(x) for x in v] for k, v in acc.items()})
        a = "  ".join(f"{k} {np.mean(v):.4f}" for k, v in sorted(acc.items())) or "n/a"
        print(f"{name:28s} n={len(rates)}  MoA {np.mean(rates):5.1f}"
              f" +/- {np.std(rates, ddof=1) if len(rates) > 1 else 0:4.1f}%   raw PCC {a}")

    json.dump(result, open(args.out, "w"), indent=1)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
