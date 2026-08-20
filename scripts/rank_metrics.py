
"""Scale-free comparison of mechanism recovery across models.

A fixed top-K cut-off is a weaker filter for a model that ranks 34 pathways than for one that
ranks 3,143, so this script reports, in addition to the hit rate, the percentile of the expected
pathway in each model's own ranking, the enrichment over the chance level implied by its
candidate set, and the mean reciprocal rank.

Usage:
  python scripts/rank_metrics.py --hdca <scores.npy> [...] [--per_drug out.csv]
  python scripts/rank_metrics.py --drpreter <scores.npz> [...]
"""
import os, re, sys, json, argparse
from math import comb
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(os.getcwd(), "src", "analysis"),
           os.path.join(_here, "..", "src", "analysis"),
           os.path.join(_here, "src", "analysis")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)
try:
    from interpret_hdca import DRUG_MOA
except ModuleNotFoundError:
    sys.exit("Could not locate src/analysis/interpret_hdca.py; run this from the project root.")

def wb(kw, text):
    return re.search(rf"\b{re.escape(kw)}\d*\b", str(text), re.IGNORECASE) is not None

def per_drug(scores, names, kws, K=10):
    E = [i for i, n in enumerate(names) if any(wb(k, n) for k in kws)]
    if not E:
        return None
    order = np.argsort(scores)[::-1]
    pos = {int(d): r + 1 for r, d in enumerate(order)}
    r, N = min(pos[i] for i in E), len(names)
    chance = 1 - comb(N - len(E), K) / comb(N, K) if N - len(E) >= K else 1.0
    return r, N, len(E), int(r <= K), chance

def summarize(rows, label):
    r = np.array([x[0] for x in rows]); N = rows[0][1]
    hit = np.array([x[3] for x in rows]); ch = np.array([x[4] for x in rows])
    pct = r / N * 100
    print(f"  {label:26s} n={len(rows):2d}  N={N:5d}  hit@10 {hit.mean()*100:5.1f}%  "
          f"chance {ch.mean()*100:5.2f}%  enrich {hit.mean()/ch.mean():5.1f}x  "
          f"median pct {np.median(pct):5.1f}%  MRR {np.mean(1/r):.3f}")
    return dict(n=len(rows), hit=hit.mean()*100, chance=ch.mean()*100,
                enrich=hit.mean()/ch.mean(), med_pct=float(np.median(pct)), mrr=float(np.mean(1/r)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdca", nargs="*", default=[], help="HDCA-Net interpretability score files, one per seed")
    ap.add_argument("--drpreter", nargs="*", default=[], help="DRPreter scores.npz files, one per seed")
    ap.add_argument("--matrices", default="data/matrices_gdsc12")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_drug", default=None,
                    help="dump per-drug, per-seed ranks and hits to CSV")
    args = ap.parse_args()
    res = {"hdca": [], "drpreter": []}
    pd_rows = []

    if args.hdca:
        ids = json.load(open(os.path.join(args.matrices, "id_maps.json")))
        names = [None] * ids["num_pathways"]
        for n, i in ids["pathway_to_idx"].items():
            names[i] = n
        d2i = {k.lower(): v for k, v in ids["drug_name_to_idx"].items()}
        print("HDCA:")
        for f in args.hdca:
            S = np.load(f); rows = []
            for d, kws in DRUG_MOA.items():
                idx = d2i.get(d) or d2i.get(d + " (-)")
                if idx is None:
                    cand = [k for k in d2i if k.startswith(d)]
                    idx = d2i[cand[0]] if cand else None
                if idx is None:
                    continue
                m = per_drug(S[idx], names, kws, args.topk)
                if m:
                    rows.append(m)
                    pd_rows.append({"run": os.path.basename(os.path.dirname(f)), "drug": d,
                                    "rank": m[0], "n_expected": m[2], "hit": m[3]})
            if rows:
                res["hdca"].append(summarize(rows, os.path.basename(os.path.dirname(f))))

    if args.drpreter:
        print("DRPreter:")
        for f in args.drpreter:
            z = np.load(f, allow_pickle=True)
            names = [str(x) for x in z["pathways"]]; drugs = [str(x) for x in z["drugs"]]
            rows = []
            for i, d in enumerate(drugs):
                kws = DRUG_MOA.get(d.lower().replace(" (-)", ""))
                if kws is None:
                    continue
                m = per_drug(z["path_scores"][i], names, kws, args.topk)
                if m:
                    rows.append(m)
            if rows:
                res["drpreter"].append(summarize(rows, os.path.basename(os.path.dirname(f))))

    for k, v in res.items():
        if v:
            print(f"\n[{k}] mean over {len(v)} run(s): "
                  f"hit@10 {np.mean([x['hit'] for x in v]):.1f}%  "
                  f"enrich {np.mean([x['enrich'] for x in v]):.1f}x  "
                  f"median pct {np.mean([x['med_pct'] for x in v]):.1f}%  "
                  f"MRR {np.mean([x['mrr'] for x in v]):.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("saved:", args.out)
    if args.per_drug and pd_rows:
        import csv
        with open(args.per_drug, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pd_rows[0].keys())); w.writeheader(); w.writerows(pd_rows)
        print("saved:", args.per_drug)

if __name__ == "__main__":
    main()
