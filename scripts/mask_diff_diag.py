"""Verify every drug--gene mask number quoted in Methods (sec:pruned) against the masks
themselves. Run on the server, where both masks are the current generation.

The manuscript states:
    the broad mask annotates 435 of 542 compounds with a median of 4 genes;
    the direct-target mask annotates 421;
    13 compounds are covered only by the direct-target mask and 27 only by the broad one;
    209 of the 542 rows change in total;
    the GDSC annotation maps to at least one gene for 346 compounds, which then carry a
    mean of 2 genes each, and the remaining 196 keep their broad row.

Only 435 and 421 can be checked away from the server (they follow from the all-zero rows
of p_gene_align.npy). Everything else needs both mask files, and the local copy of the
broad mask is a stale pre-fallback generation (240 compounds), so running this locally
reproduces the wrong answers on purpose -- it will say so.

Usage (server):
    cd /home/genetics/yiombi/HCP_Net && python scripts/mask_diff_diag.py
CPU only, a few seconds, safe to run alongside training.
"""
import os

import numpy as np

MD = "data/matrices_gdsc12"
BROAD, DIRECT = "hcdt_drug_gene.npy", "hcdt_drug_gene_pruned.npy"
CLAIMED = dict(broad=435, direct=421, broad_median=4, direct_only=13, broad_only=27,
               rows_changed=209)


def main():
    b = np.load(os.path.join(MD, BROAD))
    d = np.load(os.path.join(MD, DIRECT))
    nb, nd = b.sum(1), d.sum(1)
    hb, hd = nb > 0, nd > 0

    got = dict(
        broad=int(hb.sum()),
        direct=int(hd.sum()),
        broad_median=float(np.median(nb[hb])),
        direct_only=int((hd & ~hb).sum()),
        broad_only=int((hb & ~hd).sum()),
        rows_changed=int((b != d).any(1).sum()),
    )

    if got["broad"] != 435:
        print(f"!! {BROAD} has {got['broad']}/542 annotated compounds, not 435.\n"
              "   This is the stale pre-fallback generation; the numbers below are not\n"
              "   the manuscript's. Fetch the current mask before trusting them.\n")

    print(f"{'quantity':22s} {'computed':>9s} {'manuscript':>11s}")
    for k in CLAIMED:
        flag = "" if abs(got[k] - CLAIMED[k]) < 1e-9 else "   <-- MISMATCH"
        print(f"{k:22s} {got[k]:>9} {CLAIMED[k]:>11}{flag}")

    # Not the manuscript's "mean of 2 genes": that is taken over the 346 compounds the
    # GDSC annotation maps, which cannot be identified from the mask files alone -- it
    # needs the build step. Reported here only as a sanity range.
    print(f"\ndirect-target rows: mean {nd[hd].mean():.2f}, median {np.median(nd[hd]):.0f} "
          f"genes over all {int(hd.sum())} non-empty rows")
    print(f"compounds empty under both masks: {int((~hb & ~hd).sum())}")
    print(f"compounds annotated by both:      {int((hb & hd).sum())}")


if __name__ == "__main__":
    main()
