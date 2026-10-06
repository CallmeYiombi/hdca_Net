
"""Summarise the 16-drug MoA hit rate for every interpretability run on disk.

Scores each results/interpret_hdca/<tag>_s<seed>/p_gene_align.npy with the same
word-boundary matcher and top-10 cut-off used elsewhere, and writes one row per
(tag, seed) so the per-seed values can be aggregated and tested downstream. The
output is a few kilobytes, which avoids moving the score matrices themselves.

Usage: python scripts/moa_summarize.py --out results/moa_summary_all.csv
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

DRUG_MOA = {
    "erlotinib": ["EGFR", "ErbB"], "gefitinib": ["EGFR", "ErbB"],
    "lapatinib": ["EGFR", "ErbB", "HER2"], "afatinib": ["EGFR", "ErbB"],
    "vemurafenib": ["MAPK", "BRAF", "ERK"], "dabrafenib": ["MAPK", "BRAF", "ERK"],
    "trametinib": ["MAPK", "MEK", "ERK"], "selumetinib": ["MAPK", "MEK", "ERK"],
    "bkm120": ["PI3K", "AKT", "mTOR"], "gdc-0941": ["PI3K", "AKT"],
    "pictilisib": ["PI3K", "AKT"], "vorinostat": ["HDAC", "histone", "acetyl"],
    "panobinostat": ["HDAC", "histone", "acetyl"], "imatinib": ["BCR-ABL", "ABL", "KIT"],
    "nilotinib": ["BCR-ABL", "ABL"], "crizotinib": ["ALK", "MET"],
    "nutlin-3a": ["p53", "MDM2", "TP53"], "olaparib": ["PARP", "DNA repair", "BRCA"],
    "rucaparib": ["PARP", "DNA repair"],
}

def word_boundary(keyword, name):
    return re.search(rf"\b{re.escape(keyword)}\d*\b", name, re.I) is not None

def build_probe(mat_dir):
    pathways = json.load(open(os.path.join(mat_dir, "id_maps.json")))["pathway_list"]
    tbl = pd.read_csv(os.path.join(mat_dir, "sample_table.csv")).drop_duplicates("drug_idx")
    probe = []
    for idx, name in zip(tbl["drug_idx"], tbl["drug_name_lower"]):
        for key, keywords in DRUG_MOA.items():
            if key in name or name in key:
                probe.append((int(idx), name, keywords))
                break
    return pathways, probe

def score(matrix, pathways, probe, top_k):
    hits, ranks = 0, []
    for idx, _, keywords in probe:
        order = np.argsort(matrix[idx])[::-1]
        rank = len(pathways)
        for position, j in enumerate(order, 1):
            if any(word_boundary(k, pathways[j]) for k in keywords):
                rank = position
                break
        ranks.append(rank)
        hits += rank <= top_k
    return hits, ranks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--interp_root", default="results/interpret_hdca")
    ap.add_argument("--channel", default="p_gene_align")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out", default="results/moa_summary_all.csv")
    args = ap.parse_args()

    pathways, probe = build_probe(args.mat_dir)
    print(f"probe drugs: {len(probe)}; pathways: {len(pathways)}")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.interp_root, "*", f"{args.channel}.npy"))):
        run = os.path.basename(os.path.dirname(path))
        m = re.match(r"(.+)_s(\d+)$", run)
        tag, seed = (m.group(1), int(m.group(2))) if m else (run, -1)
        matrix = np.load(path)
        hits, ranks = score(matrix, pathways, probe, args.top_k)
        empty = int((np.abs(matrix).max(axis=1) == 0).sum())
        rows.append({"tag": tag, "seed": seed, "run": run,
                     "hits": hits, "n_probe": len(probe),
                     "hit_rate": 100.0 * hits / len(probe),
                     "empty_mask_rows": empty,
                     "median_rank": float(np.median(ranks)),
                     "ranks": " ".join(str(r) for r in ranks)})
        print(f"  {run:16s} hits {hits:2d}/{len(probe)}  ({100.0*hits/len(probe):5.1f}%)  "
              f"empty-mask rows {empty}")

    df = pd.DataFrame(rows).sort_values(["tag", "seed"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}  ({len(df)} runs)")
    print("\nper-tag mean over seeds:")
    for tag, g in df.groupby("tag"):
        sd = g["hit_rate"].std(ddof=1) if len(g) > 1 else 0.0
        print(f"  {tag:12s} n={len(g)}  {g['hit_rate'].mean():5.1f} +/- {sd:4.1f}%")

if __name__ == "__main__":
    main()
