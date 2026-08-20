"""Mechanism-of-action agreement on an independent target database.

Extends the hand-curated 16-drug probe to every panel compound annotated in DGIdb or
ChEMBL. For each drug the expected set is the specific pathways (membership <= max_path_size)
containing an annotated target; a hit requires one of them in the top-K of the gene branch.
Reports seed reproducibility, a drug-level McNemar test, a permutation null and a
sensitivity grid over the target count, pathway size and K.

Usage:
  python scripts/moa_expand_chembl.py --db_type dgidb --db data/interactions.tsv \
      --broad <broad_mask_scores.npy> --direct <direct_mask_scores.npy> --n_perm 10000
"""
import os, json, argparse, sqlite3, random
import numpy as np
import pandas as pd
from scipy.stats import binomtest

DGIDB_GOOD = {"ChEMBL", "GuideToPharmacology", "TTD", "DTC"}
DGIDB_DIR = {"inhibitor", "antagonist", "blocker", "agonist", "activator",
             "modulator", "negative modulator", "positive modulator"}
Q_MECH = """
SELECT DISTINCT md.molregno, cs.component_synonym AS gene
FROM drug_mechanism dm
JOIN molecule_dictionary md ON dm.molregno = md.molregno
JOIN target_dictionary   td ON dm.tid = td.tid
JOIN target_components   tc ON td.tid = tc.tid
JOIN component_synonyms  cs ON tc.component_id = cs.component_id
WHERE cs.syn_type = 'GENE_SYMBOL' AND td.organism = 'Homo sapiens';
"""

def load_targets_dgidb(tsv, panel, sym2idx, topn):
    dg = pd.read_csv(tsv, sep="\t", skiprows=2)
    dg["dl"] = dg["drug_name"].astype(str).str.lower().str.strip()
    dg = dg[dg["gene_name"].notna() & dg["dl"].isin(panel)
            & dg["interaction_source_db_name"].isin(DGIDB_GOOD)
            & dg["interaction_types"].isin(DGIDB_DIR)]
    n_matched = dg["dl"].nunique()
    tgt = {}
    for d, g in dg.groupby("dl"):
        gg = g.sort_values("interaction_score", ascending=False)
        genes = [x for x in dict.fromkeys(gg["gene_name"]) if x in sym2idx]
        genes = genes if topn in (0, None) else genes[:topn]
        if genes:
            tgt[d] = genes
    return tgt, n_matched

def load_targets_chembl(db, panel, sym2idx, topn):
    con = sqlite3.connect(db)
    mech = pd.read_sql(Q_MECH, con)
    name = pd.concat([
        pd.read_sql("SELECT molregno, LOWER(pref_name) AS nm FROM molecule_dictionary WHERE pref_name IS NOT NULL;", con),
        pd.read_sql("SELECT molregno, LOWER(synonyms) AS nm FROM molecule_synonyms WHERE synonyms IS NOT NULL;", con),
    ], ignore_index=True)
    con.close()
    mech = mech[mech["gene"].isin(sym2idx)]
    mol2genes = mech.groupby("molregno")["gene"].apply(lambda s: list(dict.fromkeys(s))).to_dict()
    nm2mol = dict(zip(name["nm"].str.strip(), name["molregno"]))
    tgt = {}
    n_matched = 0
    for d in panel:
        mol = nm2mol.get(d)
        if mol is None:
            continue
        n_matched += 1
        if mol in mol2genes:
            genes = mol2genes[mol]
            tgt[d] = genes if topn in (0, None) else genes[:topn]
    return tgt, n_matched

def build_masks(tgt, sym2idx, gp, psize, max_path_size):
    masks = {}
    for d, genes in tgt.items():
        idx = [sym2idx[g] for g in genes if g in sym2idx]
        if not idx:
            continue
        m = (gp[idx] > 0).any(0) & (psize <= max_path_size)
        if m.sum() >= 1:
            masks[d] = np.where(m)[0]
    return masks

def hit(P, di, exp_idx, K):
    return len(set(np.argsort(P[di])[::-1][:K].tolist()) & set(exp_idx.tolist())) > 0

def rate(P, masks, low2idx, K):
    h = sum(hit(P, low2idx[d], e, K) for d, e in masks.items())
    return h, len(masks)

def mcnemar(Pa, Pb, masks, low2idx, K):
    b01 = b10 = 0
    for d, e in masks.items():
        a = hit(Pa, low2idx[d], e, K); b = hit(Pb, low2idx[d], e, K)
        b01 += (b and not a); b10 += (a and not b)
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
    return b01, b10, p

def perm_null(P, masks, low2idx, K, n_perm, seed=0):
    """Permutation null: reassign expected pathway sets across drugs n_perm times."""
    rng = random.Random(seed)
    drugs = list(masks); exps = list(masks.values()); n = len(drugs)
    real = rate(P, masks, low2idx, K)[0] / n * 100
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = exps[:]; rng.shuffle(perm)
        null[t] = sum(hit(P, low2idx[drugs[i]], perm[i], K) for i in range(n)) / n * 100
    p = (np.sum(null >= real) + 1) / (n_perm + 1)
    return real, float(null.mean()), float(null.std()), float(p)

def mean_over_seeds(P_list, masks, low2idx, K):
    r = [rate(P, masks, low2idx, K)[0] / len(masks) * 100 for P in P_list]
    return float(np.mean(r)), float(np.std(r, ddof=1) if len(r) > 1 else 0.0), r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_type", choices=["dgidb", "chembl"], required=True)
    ap.add_argument("--db", required=True, help="DGIdb interactions.tsv or ChEMBL .db")
    ap.add_argument("--broad", nargs="+", required=True, help="gene-branch scores for the broad mask (repeat for seeds)")
    ap.add_argument("--direct", nargs="+", required=True, help="gene-branch scores for the direct-target mask (repeat for seeds)")
    ap.add_argument("--matrices", default="data/matrices_gdsc12")
    ap.add_argument("--topn", type=int, default=2, help="maximum number of targets per drug (0 = all)")
    ap.add_argument("--max_path_size", type=int, default=15)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--out", default="results/moa_expand.json")
    args = ap.parse_args()

    idm = json.load(open(os.path.join(args.matrices, "id_maps.json")))
    sym2idx, pl = idm["gene_symbol_to_idx"], idm["pathway_list"]
    gp = np.load(os.path.join(args.matrices, "gene_pathway.npy")); psize = (gp > 0).sum(0)
    st = pd.read_csv(os.path.join(args.matrices, "sample_table.csv"))
    low2idx = dict(zip(st["drug_name_lower"], st["drug_idx"]))
    panel = set(st["drug_name_lower"].unique())

    loader = load_targets_dgidb if args.db_type == "dgidb" else load_targets_chembl
    tgt, n_matched = loader(args.db, panel, sym2idx, args.topn)
    n_univ = sum(1 for g in tgt.values() if any(x in sym2idx for x in g))
    masks = build_masks(tgt, sym2idx, gp, psize, args.max_path_size)

    print("=" * 66)
    print(f" MoA agreement on an independent database [{args.db_type}]  (topn={args.topn}, size<={args.max_path_size}, K={args.topk})")
    print("=" * 66)
    print("[#5 selection flow]")
    print(f"  panel compounds              : {len(panel)}")
    print(f"  -> matched in database       : {n_matched}")
    print(f"  -> targets in gene universe  : {len(tgt)}")
    print(f"  -> with a specific pathway (size<={args.max_path_size}) = evaluated : {len(masks)}")

    Pb = [np.load(f) for f in args.broad]; Pd = [np.load(f) for f in args.direct]

    bmean, bstd, br = mean_over_seeds(Pb, masks, low2idx, args.topk)
    dmean, dstd, dr = mean_over_seeds(Pd, masks, low2idx, args.topk)
    print(f"\n[main] gene MoA hit  (n={len(masks)}, {len(Pb)}/{len(Pd)} seed)")
    print(f"  broad  : {bmean:.1f} +/- {bstd:.1f}%   {[round(x,1) for x in br]}")
    print(f"  direct : {dmean:.1f} +/- {dstd:.1f}%   {[round(x,1) for x in dr]}")

    real, nmean, nstd, pperm = perm_null(Pd[0], masks, low2idx, args.topk, args.n_perm)
    print(f"\n[permutation null, {args.n_perm} draws]  direct real {real:.1f}% vs "
          f"null {nmean:.1f} +/- {nstd:.1f}%  -> empirical p={pperm:.2e}")

    b_gain, d_gain, pmc = mcnemar(Pd[0], Pb[0], masks, low2idx, args.topk)
    print(f"[drug-level McNemar] direct-gain={d_gain}, broad-gain={b_gain}, p={pmc:.2e}")

    result = {"db_type": args.db_type, "n_panel": len(panel), "n_matched": n_matched,
              "n_target": len(tgt), "n_eval": len(masks),
              "broad_mean": bmean, "broad_std": bstd, "direct_mean": dmean, "direct_std": dstd,
              "perm_real": real, "perm_null_mean": nmean, "perm_p": pperm,
              "mcnemar_direct_gain": d_gain, "mcnemar_broad_gain": b_gain, "mcnemar_p": pmc,
              "params": {"topn": args.topn, "max_path_size": args.max_path_size, "topk": args.topk}}

    if args.sensitivity:
        print("\n[#1/#4 sensitivity: broad->direct hit% (McNemar p)]")
        grid = []
        for tn in [1, 2, 3]:
            tg, _ = loader(args.db, panel, sym2idx, tn)
            for ms in [10, 15, 20, 30]:
                mk = build_masks(tg, sym2idx, gp, psize, ms)
                for K in [5, 10, 20]:
                    hb = rate(Pb[0], mk, low2idx, K)[0] / len(mk) * 100
                    hd = rate(Pd[0], mk, low2idx, K)[0] / len(mk) * 100
                    _, _, pp = mcnemar(Pd[0], Pb[0], mk, low2idx, K)
                    grid.append(dict(topn=tn, size=ms, topk=K, n=len(mk),
                                     broad=round(hb, 1), direct=round(hd, 1), p=pp))
                    print(f"  topn={tn} size<={ms:2d} K={K:2d} (n={len(mk):3d}): "
                          f"broad {hb:4.1f}% -> direct {hd:4.1f}%  p={pp:.1e}")
        result["sensitivity"] = grid

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nsaved: {args.out}")
    print("\nNote: the shuffled null tests drug specificity; it does not remove the\n"
          "structural circularity of scoring a mask against its own targets.")

if __name__ == "__main__":
    main()
