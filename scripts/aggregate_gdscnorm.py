#!/usr/bin/env python3
"""Sensitivity of the cross-study results to how the evaluation cohort is standardized.

For each model, on the same checkpoint and the same evaluation pairs, compare
  (a) cohort-own z-score     = crosspred_<pset>_seed<N>.npz           (the setting used in the paper)
  (b) GDSC-statistics z-score = crosspred_<pset>_gdscnorm_seed<N>.npz  (the sensitivity analysis)
Because the pair set is identical, the difference isolates the standardization choice.

The question is not what the absolute values are but whether the ranking of models survives.

Usage (on the server):
    python scripts/aggregate_gdscnorm.py
    python scripts/aggregate_gdscnorm.py --cap 13.82 --csv results/gdscnorm_table.csv
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

import numpy as np

BASELINES = ("GraphDRP", "DeepCDR", "TGSA", "PANCDR")

# Models that re-standardize the evaluation cohort inside their own pipeline. The intervention
# never reaches them, so reporting a difference of zero would be misleading; mark them n/a.
#   train_pancdr.py:199        expr = zscore(np.load(p / "cell_expr.npy"))
#   DRPreter build_hdca_data.py:117  sub = StandardScaler().fit_transform(sub)
SELF_STANDARDIZING = {"PANCDR", "DRPreter"}


# There are many HDCA variant runs (broad/dense/cardmatched/gamma/sham), so the substring
# 'hdca' does not identify the reference model. Count a file as HDCA-Net only when it sits in
# the reference config directory (pruned_div03) or directly under results/.
HDCA_REF = "pruned_div03"


def model_of(path):
    """Derive the model name from the path.

    The results tree mixes different experiments that share a model name, so matching on the
    name alone is unsafe. Two cases have to be excluded explicitly:
      - results/baselines_aux{gated,concat}/  : the arm that supplies the mask to a baseline
      - .../cross_dataset/{gene,pathway}/     : the single-branch ablations of HDCA-Net
    Only the reference configuration is counted.
    """
    parts = path.split(os.sep)
    if any(p.startswith("baselines_aux") for p in parts):
        return None                       # the aux arm is a different experiment
    for b in BASELINES:
        if b in parts:
            return b
    if HDCA_REF in path and "gene_pathway" in parts:
        return "HDCA-Net"                 # only the reference, which uses both branches
    return None


def load_subset_mask(sample_table, ic_csv):
    """True for the (cell, drug) pairs DRPreter covers. npz row order equals sample_table row
    order; same convention as build_masks in src/analysis/winsorize_cross_table.py."""
    import pandas as pd
    ref = set(zip(pd.read_csv(ic_csv)["DepMap_ID"].astype(str),
                  pd.read_csv(ic_csv)["Drug name"].astype(str)))
    tbl = pd.read_csv(sample_table)
    pairs = zip(tbl["model_id"].astype(str), tbl["drug_name"].astype(str))
    return np.array([p in ref for p in pairs], dtype=bool)


def cohort_of(name):
    n = name.lower()
    if "ccle" in n:
        return "CCLE"
    if "gcsi" in n:
        return "gCSI"
    return None


def collect_npz(root="results"):
    """(model, cohort, seed) -> {'cohort': path, 'gdsc': path}"""
    out = {}
    for f in glob.glob(os.path.join(root, "**", "crosspred_*.npz"), recursive=True):
        base = os.path.basename(f)
        m = re.match(r"crosspred_(.+)_seed(\d+)\.npz$", base)
        if not m:
            continue
        pset, seed = m.group(1), int(m.group(2))
        kind = "gdsc" if pset.endswith("_gdscnorm") else "cohort"
        mdl, coh = model_of(f), cohort_of(pset)
        if not mdl or not coh:
            continue
        slot = out.setdefault((mdl, coh, seed), {})
        # The same prediction can exist both in a run directory and in a flat collection folder.
        # Prefer the reference config directory so configurations do not get mixed.
        if kind in slot and HDCA_REF in slot[kind] and HDCA_REF not in f:
            continue
        slot[kind] = f
    return out


def pcc_scc(f, cap, mask=None):
    d = np.load(f)
    yt = np.minimum(d["y_true"], cap)
    yp = d["y_pred"]
    if mask is not None:
        if len(yt) != len(mask):
            raise SystemExit(f"[subset] {f}: npz has {len(yt)} rows but the mask has {len(mask)} (row order mismatch)")
        yt, yp = yt[mask], yp[mask]
    r = np.corrcoef(yt, yp)[0, 1]
    from scipy import stats
    return r, stats.spearmanr(yt, yp).statistic, len(yt)


def drpreter(cap, base="DRPreter-main"):
    """DRPreter writes pred.csv (y_pred, y_true), under Result_HDCA or Result_HDCA_gdscnorm."""
    from scipy import stats
    res = {}
    for kind, d in (("cohort", "Result_HDCA"), ("gdsc", "Result_HDCA_gdscnorm")):
        for f in glob.glob(os.path.join(base, d, "cross_*_seed*_pred.csv")):
            m = re.search(r"cross_(CCLE|gCSI)_seed(\d+)_pred\.csv$", os.path.basename(f))
            if not m:
                continue
            coh, seed = m.group(1), int(m.group(2))
            yt, yp = [], []
            for row in csv.DictReader(open(f)):
                yt.append(min(float(row["y_true"]), cap))
                yp.append(float(row["y_pred"]))
            yt, yp = np.array(yt), np.array(yp)
            res.setdefault(("DRPreter", coh, seed), {})[kind] = (
                np.corrcoef(yt, yp)[0, 1], stats.spearmanr(yt, yp).statistic, len(yt))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=13.82, help="winsorize cap (the GDSC support)")
    ap.add_argument("--root", default="results")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--subset", action="store_true",
                    help="same scale as Table 1: restrict to the pairs DRPreter covers")
    ap.add_argument("--matrices_root", default="data")
    ap.add_argument("--drpreter_ic_dir", default="DRPreter-main/Data_HDCA")
    ap.add_argument("--drpreter_ic_dir_gdsc", default="DRPreter-main/Data_HDCA_gdscnorm")
    ap.add_argument("--verbose", action="store_true",
                    help="for seed 1, report which files were paired, the pair count, and whether the predictions are identical")
    args = ap.parse_args()

    # subset mask: built per arm from its own sample_table and IC csv
    masks = {}
    if args.subset:
        spec = {("CCLE", "cohort"): ("matrices_ccle_2015", args.drpreter_ic_dir, "IC_CCLE.csv"),
                ("gCSI", "cohort"): ("matrices_gcsi_2019", args.drpreter_ic_dir, "IC_gCSI.csv"),
                ("CCLE", "gdsc"): ("matrices_ccle_2015_gdscnorm", args.drpreter_ic_dir_gdsc, "IC_CCLE.csv"),
                ("gCSI", "gdsc"): ("matrices_gcsi_2019_gdscnorm", args.drpreter_ic_dir_gdsc, "IC_gCSI.csv")}
        for key, (mdir, icdir, ic) in spec.items():
            st_path = os.path.join(args.matrices_root, mdir, "sample_table.csv")
            m = load_subset_mask(st_path, os.path.join(icdir, ic))
            masks[key] = m
            print(f"  [subset] {key[0]}/{key[1]}: {len(m)} rows -> {int(m.sum())} shared")
        print()

    pairs = collect_npz(args.root)
    if args.verbose:
        print("=== seed 1 pairing diagnostic ===")
        for (mdl, coh, seed), d in sorted(pairs.items(), key=lambda x: (x[0][0], x[0][1])):
            if seed != 1:
                continue
            print(f"[{mdl} / {coh}]")
            for kind in ("cohort", "gdsc"):
                print(f"   {kind:7s}: {d.get(kind, '(none)')}")
            if "cohort" in d and "gdsc" in d:
                a, b = np.load(d["cohort"]), np.load(d["gdsc"])
                same = np.allclose(a["y_pred"], b["y_pred"])
                print(f"   n_pairs: {len(a['y_true'])} / {len(b['y_true'])}   "
                      f"predictions identical: {same}{'   <== the standardization did not take effect' if same else ''}")
        print()
    scored = {}
    for key, d in pairs.items():
        if "gdsc" not in d or "cohort" not in d:
            continue
        mdl, coh, _ = key
        scored[key] = {k: pcc_scc(v, args.cap, masks.get((coh, k)))
                       for k, v in d.items()}
    scored.update({k: v for k, v in drpreter(args.cap).items() if len(v) == 2})

    agg = {}
    for (mdl, coh, _), v in scored.items():
        agg.setdefault((mdl, coh), {"cohort": [], "gdsc": []})
        agg[(mdl, coh)]["cohort"].append(v["cohort"][0])
        agg[(mdl, coh)]["gdsc"].append(v["gdsc"][0])

    rows = []
    for coh in ("CCLE", "gCSI"):
        entries = [(m, a) for (m, c), a in agg.items() if c == coh]
        if not entries:
            continue
        print(f"\n=== {coh}  (winsorized PCC, cap {args.cap}) ===")
        print(f"{'model':12s} {'cohort-norm':>18s} {'GDSC-norm':>18s} {'Δ':>9s}  n")
        entries.sort(key=lambda x: -st.mean(x[1]["cohort"]))
        for mdl, a in entries:
            c, g = a["cohort"], a["gdsc"]
            sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
            delta = ("      n/a" if mdl in SELF_STANDARDIZING
                     else f"{st.mean(g)-st.mean(c):>+9.4f}")
            print(f"{mdl:12s} {st.mean(c):>10.4f}±{sd(c):<7.4f} "
                  f"{st.mean(g):>10.4f}±{sd(g):<7.4f} {delta}  {len(c)}")
            rows.append({"cohort": coh, "model": mdl, "n_runs": len(c),
                         "pcc_cohortnorm": round(st.mean(c), 4),
                         "pcc_gdscnorm": round(st.mean(g), 4),
                         "delta": "" if mdl in SELF_STANDARDIZING
                                  else round(st.mean(g) - st.mean(c), 4),
                         "self_standardizing": mdl in SELF_STANDARDIZING})
        rank_c = [m for m, _ in entries]
        rank_g = [m for m, _ in sorted(entries, key=lambda x: -st.mean(x[1]["gdsc"]))]
        print(f"  ranking (cohort standardization): {' > '.join(rank_c)}")
        print(f"  ranking (GDSC standardization)  : {' > '.join(rank_g)}")
        print(f"  -> ranking {'unchanged' if rank_c == rank_g else 'CHANGED'}")
        if any(m in SELF_STANDARDIZING for m, _ in entries):
            print("  n/a = the model re-standardizes the evaluation cohort itself, so this intervention does not reach it")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
