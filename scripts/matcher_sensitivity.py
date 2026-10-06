"""How much of the MoA hit rate is the matcher's doing?

The hit rate depends on a keyword matcher, and a permissive one inflates it: "MET" matches
inside "methylation", "p53" inside "p53-independent". We report the word-boundary strict
matcher throughout, and a reviewer is entitled to know what that choice costs. This scores
every run on disk under three matchers, holding the top-10 cut-off and everything else fixed:

  strict     word boundary with an optional digit suffix -- the matcher used in the paper
  substring  plain case-insensitive containment, the permissive alternative
  prefix     token-initial match, an intermediate

Usage: python scripts/matcher_sensitivity.py [--channel p_gene_align] [--top_k 10]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moa_summarize import DRUG_MOA  # noqa: E402  (one source of truth for the keywords)

MATCHERS = {
    "strict":    lambda k, n: re.search(rf"\b{re.escape(k)}\d*\b", n, re.I) is not None,
    "prefix":    lambda k, n: re.search(rf"\b{re.escape(k)}", n, re.I) is not None,
    "substring": lambda k, n: k.lower() in n.lower(),
}


def build_probe(mat_dir):
    pathways = json.load(open(os.path.join(mat_dir, "id_maps.json")))["pathway_list"]
    tbl = pd.read_csv(os.path.join(mat_dir, "sample_table.csv")).drop_duplicates("drug_idx")
    probe = []
    for idx, name in zip(tbl["drug_idx"], tbl["drug_name_lower"]):
        for key, kws in DRUG_MOA.items():
            if key in name or name in key:
                probe.append((int(idx), kws))
                break
    return pathways, probe


def hit_rate(matrix, pathways, probe, match, top_k):
    hits = 0
    for idx, kws in probe:
        order = np.argsort(matrix[idx])[::-1][:top_k]
        hits += any(match(k, pathways[j]) for j in order for k in kws)
    return 100.0 * hits / len(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--interp_root", default="results/interpret_hdca")
    ap.add_argument("--channel", default="p_gene_align")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--tags", nargs="+", default=["prn", "brd", "base"],
                    help="reference (direct-target), broad mask, unpruned baseline")
    args = ap.parse_args()

    pathways, probe = build_probe(args.mat_dir)
    print(f"{len(probe)} probe drugs, {len(pathways)} pathways, top-{args.top_k}\n")

    rows = []
    for tag in args.tags:
        for d in sorted(glob.glob(os.path.join(args.interp_root, f"{tag}_s*"))):
            f = os.path.join(d, f"{args.channel}.npy")
            if not os.path.exists(f):
                continue
            m = np.load(f)
            seed = int(re.search(r"_s(\d+)$", d).group(1))
            for name, fn in MATCHERS.items():
                rows.append(dict(tag=tag, seed=seed, matcher=name,
                                 hit=hit_rate(m, pathways, probe, fn, args.top_k)))
    if not rows:
        sys.exit("no interpretability runs found")

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="tag", columns="matcher", values="hit", aggfunc=["mean", "std"])
    print(f"{'tag':>6}  " + "  ".join(f"{m:>16}" for m in MATCHERS))
    for tag in args.tags:
        sub = df[df.tag == tag]
        if sub.empty:
            continue
        cells = []
        for m in MATCHERS:
            v = sub[sub.matcher == m].hit
            cells.append(f"{v.mean():6.1f}+/-{v.std():4.1f}")
        print(f"{tag:>6}  " + "  ".join(f"{c:>16}" for c in cells))
    n = df[df.tag == args.tags[0]].seed.nunique()
    print(f"\nmean +/- sd over {n} seeds. The paper reports the strict column.")


if __name__ == "__main__":
    main()
