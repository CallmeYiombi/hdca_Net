"""Score Table 1 on three compound sets, to separate a fair comparison from a capability.

Table 1 mixes two evaluation sets. HDCA-Net, GraphDRP, DeepCDR and TGSA are scored on all
542 compounds; PANCDR drops the compounds whose molecular graph cannot be built and
DRPreter is limited to its featurizer's coverage, and both land on the same 433. The 109
excluded compounds carry all-zero Morgan fingerprints and are 18.9% of the warm test pairs
and 10.9-28.5% of each unseen-drug fold (per (seed, fold) cell, measured from the dumps
below; an earlier estimate of 12.7-25.3% predated them).

Restricting to the 433 does NOT simply make the comparison fair -- it removes a real
difference between the models. The fingerprint-only baselines receive an all-zero input
for those compounds and can say nothing about them, so dropping them lifts their scores by
about 0.068 PCC on the warm split. HDCA-Net still reaches 85 of the 109 through the
drug--target mask, so it gains only about 0.034. The full panel is therefore the setting
in which HDCA-Net's prior does work the chemistry cannot, and the aligned subset is the
setting in which that advantage is deliberately removed.

Three scales are reported so both readings are available:
  full           all 542 compounds -- what the four full-panel rows of Table 1 use
  aligned        the 433 PANCDR and DRPreter can represent -- like-for-like with them
  structureless  the 109 with no resolvable structure -- the capability comparison

Requires the per-pair dumps from scripts/run_gdsc_dump_preds.sh.
Usage: python scripts/align_gdsc_subset.py [--out results/gdsc_6seed_aligned.csv]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

MD = "data/matrices_gdsc12"


def metrics(y, p):
    return dict(pcc=float(np.corrcoef(y, p)[0, 1]),
                scc=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))))


def collect(pattern, model, split):
    """Yield (model, split, seed, fold, y_true, y_pred, drug_idx) per dumped run."""
    st = pd.read_csv(os.path.join(MD, "sample_table.csv"))
    out = []
    for f in sorted(glob.glob(pattern)):
        d = os.path.dirname(f)
        idx_f = os.path.join(d, "test_idx.npz")
        if not os.path.exists(idx_f):
            continue
        seed = int(re.search(r"seed(\d+)", f).group(1))
        fm = re.search(r"fold(\d+)", f)
        z = np.load(f)
        ti = np.load(idx_f)["test_idx"]
        if len(ti) != len(z["y_true"]):
            print(f"  !! {f}: {len(z['y_true'])} predictions vs {len(ti)} test rows -- skipped")
            continue
        out.append((model, split, seed, int(fm.group(1)) if fm else None,
                    z["y_true"], z["y_pred"], st.iloc[ti]["drug_idx"].values))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gdsc_6seed_aligned.csv")
    ap.add_argument("--aux_root", default=None,
                    help="also score a baseline arm trained with the drug--gene mask, "
                         "e.g. results/baselines_auxgated; rows are named <model>+mask")
    args = ap.parse_args()

    zero = set(np.load("zero_fp_idx.npy").tolist())

    runs = []
    runs += collect("results/hdca_gdsc12_seed*/random/gene_pathway/pred.npz", "HDCA-Net", "random")
    runs += collect("results/hdca_gdsc12_seed*/drug5/gene_pathway/fold*/pred.npz", "HDCA-Net", "drug_fold")
    for m in ("GraphDRP", "DeepCDR", "TGSA"):
        runs += collect(f"results/baselines/random/{m}/seed*/pred.npz", m, "random")
        runs += collect(f"results/baselines/drug_fold/{m}/seed*/fold*/pred.npz", m, "drug_fold")
    if args.aux_root:
        # the same baselines retrained with the drug--gene mask: on the structureless
        # compounds this is the comparison that says whether the gap is the annotation
        for m in ("GraphDRP", "DeepCDR", "TGSA"):
            runs += collect(f"{args.aux_root}/random/{m}/seed*/pred.npz", m + "+mask", "random")
            runs += collect(f"{args.aux_root}/drug_fold/{m}/seed*/fold*/pred.npz",
                            m + "+mask", "drug_fold")
    if not runs:
        raise SystemExit("no per-pair dumps found -- run scripts/run_gdsc_dump_preds.sh first")

    rows = []
    for model, split, seed, fold, y, p, di in runs:
        keep = ~np.isin(di, list(zero))
        # "structureless" is the decisive scale: the 109 compounds with no resolvable
        # structure, which the fingerprint-only baselines cannot represent at all and
        # which HDCA-Net can still reach through the drug--target mask.
        for scale, sel in (("full", np.ones(len(y), bool)),
                           ("aligned", keep),
                           ("structureless", ~keep)):
            if sel.sum() < 10:
                continue
            rows.append(dict(model=model, split=split, seed=seed, fold=fold, scale=scale,
                             n=int(sel.sum()), **metrics(y[sel], p[sel])))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    for split in ("random", "drug_fold"):
        g0 = df[df.split == split]
        if g0.empty:
            continue
        print(f"\n=== {split} ===")
        print(f"{'model':10s} {'scale':8s} {'n pairs':>9s} {'PCC':>16s} {'SCC':>16s}")
        for model in sorted(g0.model.unique()):
            for scale in ("full", "aligned", "structureless"):
                g = g0[(g0.model == model) & (g0.scale == scale)]
                if g.empty:
                    continue
                unit = "fold" if split == "drug_fold" else "seed"
                per = g.groupby(unit)[["pcc", "scc"]].mean()
                sd = per.std(ddof=1)
                print(f"{model:10s} {scale:8s} {int(g.n.mean()):9,} "
                      f"{per.pcc.mean():9.4f}+/-{sd.pcc:.4f} {per.scc.mean():9.4f}+/-{sd.scc:.4f}")
        # what the alignment is worth, per model
        print("  alignment effect (aligned - full), PCC:")
        for model in sorted(g0.model.unique()):
            a = g0[(g0.model == model) & (g0.scale == "aligned")].pcc.mean()
            f = g0[(g0.model == model) & (g0.scale == "full")].pcc.mean()
            print(f"    {model:10s} {a - f:+.4f}")
        print("  on the 109 structureless compounds alone, PCC:")
        for model in sorted(g0.model.unique()):
            v = g0[(g0.model == model) & (g0.scale == "structureless")]
            if not v.empty:
                print(f"    {model:10s} {v.pcc.mean():+.4f}   (n={int(v.n.mean()):,} pairs)")

    print(f"\nsaved {args.out}")
    print("CHECK: the 'full' rows must match results/gdsc_6seed.csv; if they do not, the "
          "dumps do not correspond to the reported runs.")
    print("PANCDR and DRPreter need no dump -- their published numbers are already "
          "scored on the same 433 compounds.")


if __name__ == "__main__":
    main()
