"""Re-run the DGIdb negative-control statistics over ALL five sham realizations.

The published figure "68 versus 1 (McNemar p = 2.4e-19)" comes from
moa_expand_chembl.py:196, which tests Pd[0] against Pb[0] -- the first file of each
list only. With five randomized realizations on disk that is one realization, not a
summary of them, and realization 1 happens to be the weakest sham (0.7% recovery).
This script reports every realization so the paper can quote a mean and a range.

It also re-scores the 144-drug panel's broad arm with the lambda-matched broad runs
(brd_s*), because the published broad arm uses base_s* (lambda_div = 0) while the
16-drug probe's broad arm uses brd_s* (lambda_div = 0.3).

Usage: python scripts/negctrl_mcnemar_all.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from moa_expand_chembl import build_masks, load_targets_dgidb, mcnemar, rate  # noqa: E402

MATRICES = "data/matrices_gdsc12"
DB = "data/interactions.tsv"
TOPN, MAX_PATH_SIZE, TOPK = 2, 15, 10


def load_panel():
    idm = json.load(open(os.path.join(MATRICES, "id_maps.json")))
    sym2idx = idm["gene_symbol_to_idx"]
    gp = np.load(os.path.join(MATRICES, "gene_pathway.npy"))
    psize = (gp > 0).sum(0)
    st = pd.read_csv(os.path.join(MATRICES, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    panel = set(st["drug_name_lower"].unique())
    tgt, n_matched = load_targets_dgidb(DB, panel, sym2idx, TOPN)
    masks = build_masks(tgt, sym2idx, gp, psize, MAX_PATH_SIZE)
    return masks, low2idx, n_matched, len(tgt)


def pct(P, masks, low2idx):
    h, n = rate(P, masks, low2idx, TOPK)
    return 100.0 * h / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--true_tag", default="pruned_div03",
                    help="interpret_hdca run used as the true-mask arm (default: the "
                         "reference configuration). Earlier versions of this script "
                         "(default: the reference configuration).")
    ap.add_argument("--out", default=None, help="write the results to this JSON file")
    args = ap.parse_args()

    masks, low2idx, n_matched, n_tgt = load_panel()
    print(f"panel 542 -> DB-matched {n_matched} -> in gene universe {n_tgt} "
          f"-> evaluable {len(masks)}\n")

    sham_paths = sorted(glob.glob("results/interpret_hdca/random_mask_*/p_gene_align.npy"))
    shams = {os.path.basename(os.path.dirname(p)): np.load(p) for p in sham_paths}

    # True-mask arm, named explicitly rather than matched by hit rate.
    print(f"true-mask arm: {args.true_tag}  (other runs listed for context)")
    real_path, real, arm_rates = None, None, {}
    for p in sorted(glob.glob("results/interpret_hdca/*/p_gene_align.npy")):
        tag = os.path.basename(os.path.dirname(p))
        if tag.startswith("random_mask"):
            continue
        P = np.load(p)
        r = pct(P, masks, low2idx)
        arm_rates[tag] = r
        if tag == args.true_tag:
            print(f"  {tag:16s} {r:6.2f}%   <-- selected")
            real_path, real = p, P
        elif r > 40:
            print(f"  {tag:16s} {r:6.2f}%")
    if real is None:
        sys.exit(f"run '{args.true_tag}' not found under results/interpret_hdca/")

    print(f"\nMcNemar, true mask ({os.path.basename(os.path.dirname(real_path))}) "
          f"vs each sham realization:")
    gains, ps, rates = [], [], []
    per_sham = []
    for name, P in shams.items():
        b01, b10, p = mcnemar(real, P, masks, low2idx, TOPK)
        r = pct(P, masks, low2idx)
        gains.append(b10); ps.append(p); rates.append(r)
        per_sham.append(dict(realization=name, sham_hit_pct=r,
                             true_only=int(b10), sham_only=int(b01), p=float(p)))
        print(f"  {name:16s} sham hit {r:4.1f}%   true-only {b10:3d}  sham-only {b01:2d}  p={p:.2e}")
    g = np.array(gains)
    print(f"\n  true-only compounds: {g.mean():.1f} +/- {g.std(ddof=1):.1f}  "
          f"(range {g.min()}-{g.max()})")
    print(f"  sham recovery      : {np.mean(rates):.1f} +/- {np.std(rates, ddof=1):.1f}%")
    print(f"  every realization p <= {max(ps):.2e}")

    # lambda-matched broad arm for the same 144-drug panel
    print("\n144-drug panel, broad arm by lambda:")
    lam = {}
    for tag, label in (("base", "base_s* (lambda_div = 0, as published)"),
                       ("brd", "brd_s*  (lambda_div = 0.3, lambda-matched)"),
                       ("prn", "prn_s*  (direct-target reference)")):
        paths = sorted(glob.glob(f"results/interpret_hdca/{tag}_s*/p_gene_align.npy"))
        if not paths:
            print(f"  {label:44s} (no local runs)")
            continue
        rs = [pct(np.load(p), masks, low2idx) for p in paths]
        lam[tag] = dict(mean=float(np.mean(rs)), sd=float(np.std(rs, ddof=1)),
                        n=len(rs), per_run=[float(x) for x in rs])
        print(f"  {label:44s} {np.mean(rs):5.1f} +/- {np.std(rs, ddof=1):4.1f}%  (n={len(rs)})")

    if args.out:
        out = dict(
            panel=dict(n_panel=542, db_matched=n_matched, in_gene_universe=n_tgt,
                       evaluable=len(masks), topk=TOPK),
            true_mask_arm=args.true_tag,
            true_mask_arm_rate_pct=arm_rates.get(args.true_tag),
            arm_rates_pct=arm_rates,
            sham_comparisons=per_sham,
            summary=dict(true_only_mean=float(g.mean()), true_only_sd=float(g.std(ddof=1)),
                         true_only_min=int(g.min()), true_only_max=int(g.max()),
                         sham_only_max=int(max(x["sham_only"] for x in per_sham)),
                         sham_recovery_mean=float(np.mean(rates)),
                         sham_recovery_sd=float(np.std(rates, ddof=1)),
                         max_p=float(max(ps)), n_comparisons=len(per_sham)),
            lambda_arms=lam)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
