r"""
Replication of the MoA read-out on a larger, database-derived target panel.
Extends the 16 hand-curated probe drugs to hundreds. No retraining: it reads existing
p_gene_align npy files.

Metric: the expected set is the specific (small, size <= max_path_size) pathways that contain a
        drug's target genes; a hit is an overlap with the top-K pathways of the model's
        gene-branch score. This is gene-to-pathway membership, not name matching.

Checks included:
  1. sensitivity to topn, the number of targets kept per drug   (--topn, --sensitivity)
  2. broad and direct are passed npy files at the same lambda, so only the mask differs
  3. a random-target permutation null over 1k-10k draws          (--n_perm; targets shuffled across drugs)
  #4 topK / pathway-size / target# sensitivity  (--sensitivity)
  5. the selection flow from 542 compounds down to the evaluated set (always printed)
  6. seed reproducibility and a drug-level McNemar test           (pass several npy files to --broad/--direct)
  7. replication on a second database                              (--db_type dgidb|chembl)

Scope: the shuffled null tests drug specificity, that is, that the result is not a collapse onto
   generic pathways. It does not remove the structural circularity by which the direct mask
   supplies the very targets whose pathways are expected. Quantifying that circularity requires
   the randomized-mask retraining, which is a separate experiment.

Usage (DGIdb, local):
  python scripts/moa_expand_chembl.py --db_type dgidb --db data/interactions.tsv \
     --broad results/interpret_hdca_v1/fixA/p_gene_align.npy \
     --direct results/interpret_hdca/pruned/p_gene_align.npy \
     --n_perm 10000 --sensitivity
Usage (ChEMBL, server):
  python scripts/moa_expand_chembl.py --db_type chembl \
     --db data/chembl_37/chembl_37_sqlite/chembl_37.db \
     --broad results/interpret_hdca_v1/fixA/p_gene_align.npy \
     --direct results/interpret_hdca/pruned/p_gene_align.npy --n_perm 10000 --sensitivity
"""
import os, json, argparse, sqlite3, random
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# -- target sources ----------------------------------------------------------
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


# ── metric ───────────────────────────────────────────────────────────────────
def build_masks(tgt, sym2idx, gp, psize, max_path_size):
    masks = {}
    for d, genes in tgt.items():
        idx = [sym2idx[g] for g in genes if g in sym2idx]
        if not idx:
            continue
        m = (gp[idx] > 0).any(0) & (psize <= max_path_size)
        if m.sum() >= 1:
            masks[d] = np.where(m)[0]        # expected pathway index array
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
    """Permutation null: reassign the expected target-pathway sets across drugs n_perm times."""
    rng = random.Random(seed)
    drugs = list(masks); exps = list(masks.values()); n = len(drugs)
    real = rate(P, masks, low2idx, K)[0] / n * 100
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = exps[:]; rng.shuffle(perm)
        null[t] = sum(hit(P, low2idx[drugs[i]], perm[i], K) for i in range(n)) / n * 100
    p = (np.sum(null >= real) + 1) / (n_perm + 1)          # one-sided empirical p, add-one
    return real, float(null.mean()), float(null.std()), float(p)


def mean_over_seeds(P_list, masks, low2idx, K):
    r = [rate(P, masks, low2idx, K)[0] / len(masks) * 100 for P in P_list]
    return float(np.mean(r)), float(np.std(r, ddof=1) if len(r) > 1 else 0.0), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_type", choices=["dgidb", "chembl"], required=True)
    ap.add_argument("--db", required=True, help="dgidb interactions.tsv or a chembl .db")
    ap.add_argument("--broad", nargs="+", required=True, help="broad p_gene_align.npy (several files = several seeds)")
    ap.add_argument("--direct", nargs="+", required=True, help="direct p_gene_align.npy (several files = several seeds)")
    ap.add_argument("--matrices", default="data/matrices_gdsc12")
    ap.add_argument("--topn", type=int, default=2, help="maximum targets kept per drug (0 = all)")
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

    # ── #5 selection flow ──
    print("=" * 66)
    print(f" MoA database replication [{args.db_type}]  (topn={args.topn}, size<={args.max_path_size}, K={args.topk})")
    print("=" * 66)
    print("[#5 selection flow]")
    print(f"  panel compounds                     : {len(panel)}")
    print(f"  -> matched in the database          : {n_matched}")
    print(f"  -> with a target in the gene universe: {len(tgt)}")
    print(f"  -> with a specific pathway (size<={args.max_path_size}) = evaluated : {len(masks)}")

    Pb = [np.load(f) for f in args.broad]; Pd = [np.load(f) for f in args.direct]

    # -- 6. seed reproducibility (mean +/- sd when several npy files are given) --
    bmean, bstd, br = mean_over_seeds(Pb, masks, low2idx, args.topk)
    dmean, dstd, dr = mean_over_seeds(Pd, masks, low2idx, args.topk)
    print(f"\n[main] gene MoA hit  (n={len(masks)}, {len(Pb)}/{len(Pd)} seed)")
    print(f"  broad  : {bmean:.1f} ± {bstd:.1f}%   {[round(x,1) for x in br]}")
    print(f"  direct : {dmean:.1f} ± {dstd:.1f}%   {[round(x,1) for x in dr]}")

    # -- 3. permutation null (direct arm, first npy) --
    real, nmean, nstd, pperm = perm_null(Pd[0], masks, low2idx, args.topk, args.n_perm)
    print(f"\n[permutation null, {args.n_perm} draws]  direct observed {real:.1f}% vs "
          f"null {nmean:.1f} ± {nstd:.1f}%  → empirical p={pperm:.2e}")

    # -- drug-level McNemar (broad vs direct, first npy) --
    # mcnemar(Pa=direct, Pb=broad) -> b01 = hit only under broad, b10 = hit only under direct
    b_gain, d_gain, pmc = mcnemar(Pd[0], Pb[0], masks, low2idx, args.topk)
    print(f"[drug-level McNemar] direct-gain={d_gain}, broad-gain={b_gain}, p={pmc:.2e}")

    result = {"db_type": args.db_type, "n_panel": len(panel), "n_matched": n_matched,
              "n_target": len(tgt), "n_eval": len(masks),
              "broad_mean": bmean, "broad_std": bstd, "direct_mean": dmean, "direct_std": dstd,
              "perm_real": real, "perm_null_mean": nmean, "perm_p": pperm,
              "mcnemar_direct_gain": d_gain, "mcnemar_broad_gain": b_gain, "mcnemar_p": pmc,
              "params": {"topn": args.topn, "max_path_size": args.max_path_size, "topk": args.topk}}

    # ── #1 #4 sensitivity grid ──
    if args.sensitivity:
        print("\n[#1/#4 sensitivity: broad→direct hit% (McNemar p)]")
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
                          f"broad {hb:4.1f}% → direct {hd:4.1f}%  p={pp:.1e}")
        result["sensitivity"] = grid

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nsaved: {args.out}")
    print("\nNote: the shuffled null tests drug specificity, not the structural circularity by which "
          "the direct mask supplies the targets whose pathways are expected; that requires the "
          "randomized-mask retraining.")


if __name__ == "__main__":
    main()
