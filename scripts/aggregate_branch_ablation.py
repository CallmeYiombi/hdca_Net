#!/usr/bin/env python3
"""
Aggregate the branch ablation: cross-study PCC and SCC for gene-only, pathway-only and both.
Reads results/hdca_gdsc12_pruned_div03_s{N}/cross_dataset/<align>/cross_metrics.json for each
seed (a list of CCLE_2015 and gCSI_2019 dicts, keyed pcc/spearman/rmse/mae) and reports
mean +/- standard deviation.

The pcc in cross_metrics.json is raw, before winsorization, which is enough for comparing the
branches with each other. For winsorized absolute values, recompute from the crosspred npz.

Usage: python scripts/aggregate_branch_ablation.py
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results")
SEED_GLOB = "hdca_gdsc12_pruned_div03_s*"
ALIGNS = [("gene", "gene-only"), ("pathway", "pathway-only"), ("gene_pathway", "both (full)")]
DSETS = ["CCLE_2015", "gCSI_2019"]


def collect(align):
    """align -> {dataset: [pcc...], dataset+'_scc': [...]}"""
    out = {d: [] for d in DSETS}
    out.update({d + "_scc": [] for d in DSETS})
    for sd in sorted(glob.glob(os.path.join(BASE, SEED_GLOB))):
        f = os.path.join(sd, "cross_dataset", align, "cross_metrics.json")
        if not os.path.exists(f):
            continue
        for row in json.load(open(f)):
            d = row.get("dataset")
            if d in DSETS:
                out[d].append(row["pcc"])
                out[d + "_scc"].append(row["spearman"])
    return out


def main():
    print("=" * 62)
    print(" Branch ablation — cross-dataset PCC/SCC (raw, mean±std over seeds)")
    print(" config: pruned_div03 (direct-target mask, final model)")
    print("=" * 62)
    print(f"{'branch':16s} | {'CCLE PCC':16s} | {'gCSI PCC':16s} | n_seed")
    rows = []
    for align, label in ALIGNS:
        c = collect(align)
        ccle = np.array(c["CCLE_2015"]); gcsi = np.array(c["gCSI_2019"])
        if len(ccle) == 0:
            print(f"{label:16s} | (no results; not trained yet?)")
            continue
        print(f"{label:16s} | {ccle.mean():.3f} ± {ccle.std(ddof=1) if len(ccle)>1 else 0:.3f}    "
              f"| {gcsi.mean():.3f} ± {gcsi.std(ddof=1) if len(gcsi)>1 else 0:.3f}    | {len(ccle)}")
        rows.append((label, ccle, gcsi,
                     np.array(c["CCLE_2015_scc"]), np.array(c["gCSI_2019_scc"])))

    # relative contribution: what each branch alone loses against both
    both = next((r for r in rows if r[0].startswith("both")), None)
    if both is not None:
        print("\n=== loss of each branch alone against both (delta PCC) ===")
        for label, ccle, gcsi, *_ in rows:
            if label.startswith("both"):
                continue
            print(f"  {label:14s}: CCLE {ccle.mean()-both[1].mean():+.3f}, gCSI {gcsi.mean()-both[2].mean():+.3f}")
    print("\nNote: these are raw, unwinsorized PCC values, on a different scale from the winsorized\n"
          "numbers in the main text; use them for the relative comparison between branches.")


if __name__ == "__main__":
    main()
