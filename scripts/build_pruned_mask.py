
"""Build the direct-target drug-gene mask (hcdt_drug_gene_pruned.npy).

The broad HCDT drug-gene mask assigns hundreds of genes to well-studied targeted agents,
which dilutes attention over the true targets. This script replaces a drug's mask row with
the genes named by the GDSC PUTATIVE_TARGET annotation of that compound, expanding target
families to their member symbols (MEK -> MAP2K1/2, PI3K -> PIK3CA/B/D/G, HDAC -> HDAC1-11,
HER2 -> ERBB2, and so on). Drugs whose annotation maps to no gene in the universe keep their
original broad row.

Usage: python scripts/build_pruned_mask.py
"""
import os, re, json, argparse
import numpy as np
import pandas as pd

FAMILY = {
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"], "PI3KBETA": ["PIK3CB"],
    "HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
             "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11"],
    "PDGFR": ["PDGFRA", "PDGFRB"], "AKT": ["AKT1", "AKT2", "AKT3"],
    "MEK": ["MAP2K1", "MAP2K2"], "MEK1": ["MAP2K1"], "MEK2": ["MAP2K2"],
    "ERK": ["MAPK1", "MAPK3"], "ERK1": ["MAPK3"], "ERK2": ["MAPK1"],
    "ABL": ["ABL1"], "BCR-ABL": ["ABL1"], "TAK1": ["MAP3K7"], "HER2": ["ERBB2"],
    "BCL-XL": ["BCL2L1"], "BCL-W": ["BCL2L2"], "BCL-B": ["BCL2L10"], "BFL1": ["BCL2A1"],
    "TGFB": ["TGFBR1"], "P53": ["TP53"], "JNK": ["MAPK8", "MAPK9", "MAPK10"],
    "P38": ["MAPK14"], "IGF1R": ["IGF1R"], "AURORA": ["AURKA", "AURKB"],
    "CDK": ["CDK1", "CDK2", "CDK4", "CDK6"], "GSK3": ["GSK3A", "GSK3B"],
    "RAF": ["BRAF", "RAF1", "ARAF"], "SRC": ["SRC"], "MTOR": ["MTOR"],
}

def clean(t):
    t = re.sub(r"\(.*?\)", "", t).strip()
    t = re.sub(r"\b(inhibitor|class|activity|antagonist|agonist)\b.*", "", t, flags=re.I)
    return t.strip()

def map_target(raw, gsym):
    t = clean(raw)
    up = t.upper()
    out = set()
    if t in gsym:
        out.add(t)
    elif up in gsym:
        out.add(up)
    if up in FAMILY:
        out.update(g for g in FAMILY[up] if g in gsym)
    return out

def load_gdsc_targets(gdsc_files):
    cols = ["DRUG_NAME", "PUTATIVE_TARGET"]
    gdsc = pd.concat([pd.read_excel(f, usecols=cols) for f in gdsc_files]).dropna()
    d2t = {}
    for _, r in gdsc.iterrows():
        key = str(r.DRUG_NAME).strip().lower()
        vals = [x.strip() for x in str(r.PUTATIVE_TARGET).replace(";", ",").split(",")
                if x.strip() and x.strip() != "-"]
        d2t.setdefault(key, set()).update(vals)
    return d2t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--gdsc", nargs="+", default=[
        "data/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "data/GDSC2_fitted_dose_response_27Oct23.xlsx"])
    ap.add_argument("--out", default="hcdt_drug_gene_pruned.npy")
    args = ap.parse_args()

    idm = json.load(open(os.path.join(args.mat_dir, "id_maps.json")))
    gsym2idx = idm["gene_symbol_to_idx"]
    gsym = set(gsym2idx)
    drug2idx = idm["drug_name_to_idx"]

    broad = np.load(os.path.join(args.mat_dir, "hcdt_drug_gene.npy"))
    G = broad.shape[1]
    d2t = load_gdsc_targets(args.gdsc)

    pruned = broad.copy()
    n_sharp, n_fallback, sizes = 0, 0, []
    for dn, idx in drug2idx.items():
        dl = dn.strip().lower()
        tg = d2t.get(dl)
        if tg is None:
            for k, v in d2t.items():
                if k == dl or dl in k or k in dl:
                    tg = v
                    break
        genes = {g for raw in (tg or []) for g in map_target(raw, gsym)}
        genes = {g for g in genes if g in gsym2idx}
        if genes:
            row = np.zeros(G, dtype=broad.dtype)
            row[[gsym2idx[g] for g in genes]] = 1.0
            pruned[idx] = row
            n_sharp += 1
            sizes.append(len(genes))
        else:
            n_fallback += 1

    out_path = os.path.join(args.mat_dir, args.out)
    np.save(out_path, pruned)

    changed = int((~np.all(broad == pruned, axis=1)).sum())
    gained = int(((broad.sum(1) == 0) & (pruned.sum(1) > 0)).sum())
    print(f"saved {out_path}  shape={pruned.shape}")
    print(f"  remapped from GDSC annotation : {n_sharp}/{len(drug2idx)} "
          f"(median {int(np.median(sizes))}, mean {np.mean(sizes):.2f} genes)")
    print(f"  no mappable annotation, kept broad row : {n_fallback}")
    print(f"  rows that differ from the broad mask : {changed}")
    print(f"    of which previously had no target row : {gained}")
    print(f"  drugs with no target row : broad {int((broad.sum(1)==0).sum())} "
          f"-> pruned {int((pruned.sum(1)==0).sum())}")

if __name__ == "__main__":
    main()
