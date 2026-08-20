
"""Aggregate the randomized-mask negative control.

Compares the model trained with the true direct-target mask against models trained with
count-matched random masks, on two axes: cross-study predictive accuracy and the recovery
of each model's own mask-derived pathways. A comparable recovery rate indicates that the
effect is structural; a much higher rate for the true mask indicates that the encoded
targets carry information.

Usage: python scripts/eval_negative_control.py --k 5
"""
import os, glob, json, argparse
import numpy as np

def read_metrics(path):
    """cross_metrics.json -> {dataset: row}"""
    if not os.path.exists(path):
        return {}
    return {r["dataset"]: r for r in json.load(open(path))}

def perf_compare(args):
    """Predictive accuracy, true mask vs sham masks.

    The PCC values here are raw (not winsorized), so their absolute level is not on the
    scale of the reported cross-study table; use them for the true-vs-sham contrast only.
    """
    real = read_metrics(os.path.join(args.real_run, "cross_dataset/gene_pathway/cross_metrics.json"))
    if not real:
        print(f"\n[perf] no cross_metrics.json under {args.real_run}; skipping accuracy comparison")
        return {}

    seed = {}
    for sp in sorted(glob.glob(args.real_seed_glob)):
        for ds, r in read_metrics(sp).items():
            seed.setdefault(ds, []).append(r["pcc"])

    sham_runs = []
    for k in range(1, args.k + 1):
        m = read_metrics(f"results/hdca_gdsc12_random_mask_{k}/cross_dataset/gene_pathway/cross_metrics.json")
        if m:
            sham_runs.append(m)
        else:
            print(f"  [skip] realization {k}: no cross_metrics.json")

    out = {}
    print("\n=== Predictive accuracy (cross-study, raw ln IC50) ===")
    for ds, r in real.items():
        pv = [m[ds]["pcc"] for m in sham_runs if ds in m]
        if not pv:
            print(f"  {ds}: real PCC {r['pcc']:.4f} | no sham runs")
            continue
        rv = [m[ds]["rmse"] for m in sham_runs if ds in m]
        sm, ss = float(np.mean(pv)), float(np.std(pv, ddof=1) if len(pv) > 1 else 0.0)
        sstd = float(np.std(seed[ds], ddof=1)) if len(seed.get(ds, [])) > 1 else None
        line = (f"  {ds}: real PCC {r['pcc']:.4f} | sham {sm:.4f} +/- {ss:.4f} (n={len(pv)})"
                f" | delta {r['pcc'] - sm:+.4f}")
        if sstd is not None:
            line += f"   [real seed std {sstd:.4f}]"
        print(line)
        print(f"          RMSE  real {r['rmse']:.3f} | sham {float(np.mean(rv)):.3f}")
        out[ds] = {"real_pcc": r["pcc"], "sham_pcc_mean": sm, "sham_pcc_std": ss,
                   "sham_pcc_each": pv, "delta_pcc": r["pcc"] - sm,
                   "real_seed_std": sstd,
                   "real_rmse": r["rmse"], "sham_rmse_mean": float(np.mean(rv))}

    if out:
        gaps = [v["delta_pcc"] for v in out.values()]
        noise = [v["real_seed_std"] for v in out.values() if v["real_seed_std"]]
        thr = (2 * max(noise)) if noise else 0.01
        if max(gaps) < thr:
            pv = ("accuracy: real ~= sham. The true-target mask barely affects predictive accuracy, "
                  "so the mask changes what the model attends to rather than how well it predicts.")
        else:
            pv = ("accuracy: real > sham. The true targets also contribute to prediction.")
        print("accuracy verdict:", pv)
        out["verdict"] = pv
    return out

def exp_masks(mask_npy, gp, psize, max_path_size):
    m = np.load(mask_npy)
    exp = {}
    for d in range(m.shape[0]):
        genes = np.where(m[d] > 0)[0]
        if len(genes) == 0:
            continue
        em = (gp[genes] > 0).any(0) & (psize <= max_path_size)
        if em.sum() >= 1:
            exp[d] = np.where(em)[0]
    return exp

def hit_rate(P, exp, K):
    h = sum(len(set(np.argsort(P[d])[::-1][:K].tolist()) & set(e.tolist())) > 0
            for d, e in exp.items())
    return 100.0 * h / len(exp), len(exp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", default="data/matrices_gdsc12")
    ap.add_argument("--real_scores", default="results/interpret_hdca/pruned_div03/p_gene_align.npy")
    ap.add_argument("--real_mask", default="data/matrices_gdsc12/hcdt_drug_gene_pruned.npy")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_path_size", type=int, default=15)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--real_run", default="results/hdca_gdsc12_pruned_div03",
                    help="output directory of the true-mask run (read cross_metrics.json)")
    ap.add_argument("--real_seed_glob",
                    default="results/hdca_gdsc12_pruned_div03_s*/cross_dataset/gene_pathway/cross_metrics.json",
                    help="six-seed true-mask runs, used as the seed-noise baseline")
    ap.add_argument("--out", default="results/negative_control.json")
    args = ap.parse_args()

    gp = np.load(os.path.join(args.matrices, "gene_pathway.npy")); psize = (gp > 0).sum(0)

    Preal = np.load(args.real_scores)
    real_exp = exp_masks(args.real_mask, gp, psize, args.max_path_size)
    real_rate, nreal = hit_rate(Preal, real_exp, args.topk)
    print(f"real (true-target mask): {real_rate:.1f}%  (n={nreal})")

    sham = []
    for k in range(1, args.k + 1):
        sp = f"results/interpret_hdca/random_mask_{k}/p_gene_align.npy"
        mp = f"{args.matrices}/hcdt_drug_gene_random_{k}.npy"
        if not (os.path.exists(sp) and os.path.exists(mp)):
            print(f"  [skip] realization {k}: missing file ({sp})"); continue
        Pk = np.load(sp); ek = exp_masks(mp, gp, psize, args.max_path_size)
        r, n = hit_rate(Pk, ek, args.topk)
        sham.append(r)
        print(f"  sham {k} (random mask): {r:.1f}%  (n={n})")

    res = {"params": {"max_path_size": args.max_path_size, "topk": args.topk},
           "moa": {"real": real_rate, "n_real": nreal, "sham_each": sham}}
    if sham:
        sm, ss = float(np.mean(sham)), float(np.std(sham, ddof=1) if len(sham) > 1 else 0.0)
        print(f"\nsham mean: {sm:.1f} +/- {ss:.1f}%   |   real: {real_rate:.1f}%")
        gap = real_rate - sm
        print(f"real - sham = {gap:+.1f}pp")
        if gap < ss + 3:
            verdict = "real ~= sham: the recovery is largely structural; report it as propagation of the prior, not discovery."
        else:
            verdict = "real >> sham: the true targets recover substantially more than any sharp mask would."
        print("MoA verdict:", verdict)
        res["moa"].update({"sham_mean": sm, "sham_std": ss, "gap_pp": gap, "verdict": verdict})
    else:
        print("\nNo sham MoA results found; train the randomized-mask models first.")

    res["perf"] = perf_compare(args)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved: {args.out}")

if __name__ == "__main__":
    main()
