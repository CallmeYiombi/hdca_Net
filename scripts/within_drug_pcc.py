#!/usr/bin/env python3
r"""Within-drug (per-compound) cross-study PCC for the models of Table 1 and Table S11.

A pooled PCC is inflated by between-drug sensitivity differences; correlating within each
compound over the cell lines it was tested on removes that component.

Recipe: winsorize the targets at the GDSC training support (13.82), keep compounds tested on
at least --min_cells cell lines, take Pearson r within each compound, average over compounds,
then over the six seeds. Reported on two pair scales: native (each model on the pairs its own
featurizer covers) and aligned (all models on the pairs DRPreter covers).

  python scripts/within_drug_pcc.py [--scale aligned] [--out results/within_drug_pcc.csv]
"""
import os
import argparse
import numpy as np
import pandas as pd

CAP = 13.82
SEEDS = range(1, 7)

COHORTS = {  # cohort -> (sample_table dir, HDCA npz tag, baseline npz tag, DRPreter tag)
    "CCLE": ("matrices_ccle_2015", "CCLE_2015", "matrices_ccle_2015", "CCLE"),
    "gCSI": ("matrices_gcsi_2019", "gCSI_2019", "matrices_gcsi_2019", "gCSI"),
}
BASELINES = ["GraphDRP", "DeepCDR", "TGSA", "PANCDR"]


def within_drug(y_true, y_pred, drugs, cap=CAP, min_cells=10):
    """Macro-averaged per-compound Pearson r.  Returns (mean_r, n_compounds_used)."""
    y = np.minimum(np.asarray(y_true, dtype=float), cap)
    p = np.asarray(y_pred, dtype=float)
    rs = []
    for d in pd.unique(drugs):
        idx = np.where(drugs == d)[0]
        if len(idx) < min_cells:
            continue
        yy, pp = y[idx], p[idx]
        if yy.std() == 0 or pp.std() == 0:
            continue
        rs.append(np.corrcoef(yy, pp)[0, 1])
    return (float(np.mean(rs)), len(rs)) if rs else (np.nan, 0)


def load_sample_table(root, st_dir):
    st = pd.read_csv(os.path.join(root, "data", st_dir, "sample_table.csv"))
    return st["drug_name"].astype(str).values, st["model_id"].astype(str).values


def load_hdca(root, tag):
    out = []
    for s in SEEDS:
        f = os.path.join(root, "results", f"hdca_gdsc12_pruned_div03_s{s}",
                         "cross_dataset", "gene_pathway", f"crosspred_{tag}_seed{s}.npz")
        z = np.load(f)
        out.append((z["y_true"], z["y_pred"]))
    return out


def load_baseline(root, model, tag):
    out = []
    for s in SEEDS:
        f = os.path.join(root, "results", "baselines", "cross_dataset", model,
                         f"crosspred_{tag}_seed{s}.npz")
        z = np.load(f)
        out.append((z["y_true"], z["y_pred"]))
    return out


def load_drpreter(drp_root, tag):
    """DRPreter dumps y_pred/y_true only; row order matches its IC_<cohort>.csv."""
    ic = pd.read_csv(os.path.join(drp_root, "Data_HDCA", f"IC_{tag}.csv"))
    drugs = ic["Drug name"].astype(str).values
    cells = ic["DepMap_ID"].astype(str).values
    out = []
    for s in SEEDS:
        df = pd.read_csv(os.path.join(drp_root, "Result_HDCA", f"cross_{tag}_seed{s}_pred.csv"))
        assert len(df) == len(ic), f"DRPreter {tag} seed{s}: {len(df)} rows vs IC {len(ic)}"
        out.append((df["y_true"].values, df["y_pred"].values))
    return out, drugs, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="server_snapshot",
                    help="snapshot root holding data/ and results/ (default: server_snapshot)")
    ap.add_argument("--drpreter_root", default="DRPreter-main")
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--min_cells", type=int, default=10)
    ap.add_argument("--scale", choices=["native", "aligned", "both"], default="both")
    ap.add_argument("--out", default="results/within_drug_pcc.csv")
    a = ap.parse_args()

    rows = []
    for coh, (st_dir, hdca_tag, base_tag, drp_tag) in COHORTS.items():
        st_drugs, st_cells = load_sample_table(a.root, st_dir)
        drp_seeds, drp_drugs, drp_cells = load_drpreter(a.drpreter_root, drp_tag)

        # pair set DRPreter covers, expressed as a mask over sample_table row order
        ref = set(zip(drp_cells, drp_drugs))
        align_mask = np.array([pc in ref for pc in zip(st_cells, st_drugs)], dtype=bool)
        print(f"[{coh}] sample_table {len(st_drugs)} rows -> DRPreter-aligned "
              f"{int(align_mask.sum())} (DRPreter set {len(ref)})")

        models = {"HDCA-Net": (load_hdca(a.root, hdca_tag), st_drugs, True)}
        for m in BASELINES:
            models[m] = (load_baseline(a.root, m, base_tag), st_drugs, True)
        models["DRPreter"] = (drp_seeds, drp_drugs, False)  # already its own pair set

        for name, (seeds, drugs, maskable) in models.items():
            for scale in (["native", "aligned"] if a.scale == "both" else [a.scale]):
                vals, ns = [], []
                skipped = False
                for y_true, y_pred in seeds:
                    d, yt, yp = drugs, y_true, y_pred
                    if scale == "aligned" and maskable:
                        if len(yt) != len(align_mask):
                            skipped = True
                            break
                        d, yt, yp = drugs[align_mask], yt[align_mask], yp[align_mask]
                    r, n = within_drug(yt, yp, d, a.cap, a.min_cells)
                    vals.append(r)
                    ns.append(n)
                if skipped:
                    print(f"  [warn] {name}/{coh}/{scale}: npz length != sample_table "
                          f"({len(seeds[0][0])} vs {len(align_mask)}), aligned scale skipped")
                    continue
                rows.append(dict(cohort=coh, scale=scale, model=name,
                                 n_drugs=ns[0], n_seeds=len(vals),
                                 within_pcc=np.mean(vals), sd=np.std(vals, ddof=1)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    for scale in df["scale"].unique():
        for coh in COHORTS:
            sub = df[(df.scale == scale) & (df.cohort == coh)].sort_values(
                "within_pcc", ascending=False)
            print(f"\n=== within-drug PCC | {coh} | scale={scale} "
                  f"(cap {a.cap}, >={a.min_cells} cell lines) ===")
            for _, r in sub.iterrows():
                print(f"  {r.model:<10s} {r.within_pcc:.4f} +/- {r.sd:.4f}   "
                      f"({int(r.n_drugs)} drugs, {int(r.n_seeds)} seeds)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
