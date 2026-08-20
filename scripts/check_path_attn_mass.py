
"""Quantify how tightly the pathway branch follows the HCDT annotation.

Reports the softmax mass the trained model places on annotated pathways, the share of its
top-K that is annotated and the enrichment over a uniform distribution. Used to compare the
additive-prior gate with the hard-masked variant.

Usage:
  python scripts/check_path_attn_mass.py --interp <interpret_dir> --mask <mask.npy>
"""
import os, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp", required=True, help="interpretability output directory (must contain path_attn.npy)")
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--mask", default="hcdt_drug_path_direct.npy",
                    help="the drug-pathway mask that model was trained with")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    pa = np.load(os.path.join(args.interp, "path_attn.npy"))
    m  = np.load(os.path.join(args.mat_dir, args.mask)) > 0
    assert pa.shape == m.shape, f"shape mismatch {pa.shape} vs {m.shape}"

    has = m.sum(1) > 0
    idx = np.where(has)[0]
    mass = np.array([pa[d][m[d]].sum() for d in idx])
    unif = m.sum(1)[idx] / pa.shape[1]
    top = np.argsort(pa, 1)[:, ::-1][:, :args.topk]
    frac = np.array([m[d][top[d]].mean() for d in idx])

    print(f"interp = {args.interp}")
    print(f"mask   = {args.mask}  (drugs {len(idx)}/{len(has)}, "
          f"median {np.median(m.sum(1)[idx]):.0f} pathways per drug)")
    print(f"  attention mass on mask : mean {mass.mean():.4f}  median {np.median(mass):.4f}")
    print(f"  drugs with mass > 0.5 : {(mass > 0.5).mean()*100:.1f}%   > 0.9: {(mass > 0.9).mean()*100:.1f}%")
    print(f"  share of top-{args.topk} inside mask: mean {frac.mean()*100:.1f}%")
    print(f"  enrichment over uniform : median {np.median(mass/unif):.1f}x")

if __name__ == "__main__":
    main()
