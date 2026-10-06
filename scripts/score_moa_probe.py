#!/usr/bin/env python3
"""Score the 16-drug MoA probe with the word-boundary strict matcher.

For each probe drug, rank its expected pathways in the `p_gene_align.npy` score matrix
(542 x 3143) that interpret_hdca writes, and count top-10 hits. This is the logic behind the
rates reported in Section 4.2 and Figure 2.

`--selftest` checks the scoring against the existing tags in moa_summary_all.csv (prn/brd/dns);
the per-drug ranks and hit counts reproduce exactly.

Usage:
    python scripts/score_moa_probe.py --selftest
    python scripts/score_moa_probe.py --runs results/interpret_hdca/cm101_s1 ... --label cardmatched
"""
import argparse
import csv
import json
import os
import re
import statistics as st

import numpy as np

TOP_K = 10
# Drug order of the `ranks` column. It is not the alphabetical order of
# results/moa_probe_drugs.csv; it was recovered from the published broad medians and
# cross-checked against the direct-target medians (16/16).
RANK_ORDER = ["trametinib", "dabrafenib", "erlotinib", "rucaparib", "lapatinib",
              "afatinib", "gefitinib", "vorinostat", "olaparib", "nilotinib",
              "crizotinib", "nutlin-3a", "selumetinib", "pictilisib", "imatinib",
              "panobinostat"]


def find_file(*candidates):
    """The file sits at the project root in one layout and under results/ in another; try both."""
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(" | ".join(candidates))


def build_matcher(kw, mode):
    """strict is the rule used in the paper: word boundaries, allowing a numeric suffix on a gene
    symbol. prefix and substring are the looser rules used in the sensitivity analysis."""
    k = re.escape(kw.strip())
    if mode == "strict":
        return re.compile(rf"\b{k}\d*\b", re.I)
    if mode == "prefix":
        return re.compile(rf"\b{k}", re.I)
    return re.compile(k, re.I)            # substring


def load_probe(mat_dir, probe_csv, matcher="strict"):
    paths = json.load(open(os.path.join(mat_dir, "id_maps.json")))["pathway_list"]
    probe_csv = find_file(probe_csv, os.path.basename(probe_csv),
                          os.path.join("results", os.path.basename(probe_csv)))
    rows = {r["drug"].replace(" (-)", ""): r
            for r in csv.DictReader(open(probe_csv))}
    masks, idx = {}, {}
    for d in RANK_ORDER:
        r = rows[d]
        pats = [build_matcher(k, matcher)
                for k in r["expected_kws"].split(",") if k.strip()]
        masks[d] = np.array([any(p.search(name) for p in pats) for name in paths])
        idx[d] = int(r["panel_idx"])
    return masks, idx


def score_run(run_dir, masks, idx):
    a = np.load(os.path.join(run_dir, "p_gene_align.npy"))
    ranks = []
    for d in RANK_ORDER:
        order = np.argsort(-a[idx[d]])
        hit = np.flatnonzero(masks[d][order])
        ranks.append(int(hit[0]) + 1 if len(hit) else 10 ** 6)
    hits = sum(1 for r in ranks if r <= TOP_K)
    return ranks, hits, 100.0 * hits / len(RANK_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--label", default="run")
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--probe_csv", default="results/moa_probe_drugs.csv")
    ap.add_argument("--summary_csv", default="moa_summary_all.csv",
                    help="path to the summary table (default: search the root and results/)")
    ap.add_argument("--matcher", default="strict",
                    choices=("strict", "prefix", "substring"),
                    help="strict is the rule used in the paper; the others are the sensitivity analysis")
    ap.add_argument("--selftest", action="store_true",
                    help="check the scoring against the existing tags in moa_summary_all.csv")
    args = ap.parse_args()
    masks, idx = load_probe(args.mat_dir, args.probe_csv, args.matcher)

    if args.selftest:
        summary = find_file(args.summary_csv,
                            os.path.join("results", os.path.basename(args.summary_csv)))
        print(f"summary table: {summary}")
        rec = list(csv.DictReader(open(summary)))
        ok = bad = 0
        for tag, folder in (("prn", "prn"), ("brd", "brd"), ("dns", "dns")):
            for s in range(1, 7):
                d = f"results/interpret_hdca/{folder}_s{s}"
                if not os.path.isdir(d):
                    continue
                ranks, hits, rate = score_run(d, masks, idx)
                ref = [r for r in rec if r["tag"] == tag and r["seed"] == str(s)]
                if not ref:
                    continue
                exp = [int(float(v)) for v in ref[0]["ranks"].split()]
                if ranks == exp and hits == int(ref[0]["hits"]):
                    ok += 1
                else:
                    bad += 1
                    print(f"  [MISMATCH] {tag}_s{s}")
        print(f"selftest: {ok} runs reproduced, {bad} mismatched")
        return

    rates = []
    print(f"{'run':28s} {'hits':>5s} {'hit rate':>9s}")
    for d in args.runs:
        ranks, hits, rate = score_run(d, masks, idx)
        rates.append(rate)
        print(f"{os.path.basename(d):28s} {hits:>3d}/16 {rate:>8.2f}%")
    if len(rates) > 1:
        print(f"\n{args.label}: {st.mean(rates):.2f} ± {st.stdev(rates):.2f} %  (n={len(rates)})")
        print("  reference -- broad 15.62 / direct 48.96 / dense 0.00 / sham 3.75")


if __name__ == "__main__":
    main()
