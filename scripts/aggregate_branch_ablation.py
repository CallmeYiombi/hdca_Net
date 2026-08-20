
"""Aggregate the branch ablation.

Collects cross-study PCC and SCC for the gene-only, pathway-only and full models across
seeds and reports the change relative to the full model.

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
    print(" Branch ablation -- cross-dataset PCC/SCC (raw, mean+/-std over seeds)")
    print(" config: pruned_div03 (direct-target mask, final model)")
    print("=" * 62)
    print(f"{'branch':16s} | {'CCLE PCC':16s} | {'gCSI PCC':16s} | n_seed")
    rows = []
    for align, label in ALIGNS:
        c = collect(align)
        ccle = np.array(c["CCLE_2015"]); gcsi = np.array(c["gCSI_2019"])
        if len(ccle) == 0:
            print(f"{label:16s} | (no results found)")
            continue
        print(f"{label:16s} | {ccle.mean():.3f} +/- {ccle.std(ddof=1) if len(ccle)>1 else 0:.3f}    "
              f"| {gcsi.mean():.3f} +/- {gcsi.std(ddof=1) if len(gcsi)>1 else 0:.3f}    | {len(ccle)}")
        rows.append((label, ccle, gcsi,
                     np.array(c["CCLE_2015_scc"]), np.array(c["gCSI_2019_scc"])))

    both = next((r for r in rows if r[0].startswith("both")), None)
    if both is not None:
        print("\n=== change relative to the full model (delta PCC) ===")
        for label, ccle, gcsi, *_ in rows:
            if label.startswith("both"):
                continue
            print(f"  {label:14s}: CCLE {ccle.mean()-both[1].mean():+.3f}, gCSI {gcsi.mean()-both[2].mean():+.3f}")
    print("\nNote: PCC values are raw (not winsorized) and are meant for the\n"
          "relative comparison between branches only.")

if __name__ == "__main__":
    main()
