"""Cross-study evaluation table with winsorized targets.

Collects the per-pair predictions dumped by each model (crosspred_*.npz) and DRPreter's
pred.csv, winsorizes the external log-IC50 targets at the GDSC training support cap and
recomputes PCC, SCC, RMSE and MAE. With --subset every model is restricted to the
(drug, cell) pairs DRPreter covers, so all rows are scored on an identical test set.

Usage:
  python src/analysis/winsorize_cross_table.py --model_dirs <dir> [<dir> ...] \
      [--drpreter_dir <dir>] [--subset] --cap 13.82 --out results/cross_table.csv
"""
import os, re, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def metrics(y_true, y_pred, cap=None):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    if cap is not None:
        yt = np.minimum(yt, cap)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    return dict(n=int(m.sum()),
                PCC=float(pearsonr(yt, yp)[0]),
                SCC=float(spearmanr(yt, yp)[0]),
                RMSE=float(np.sqrt(np.mean((yt - yp) ** 2))),
                MAE=float(np.mean(np.abs(yt - yp))))

def which_cset(name):
    n = name.lower()
    if "ccle" in n:  return "CCLE"
    if "gcsi" in n:  return "gCSI"
    return None

def model_name(d):
    b = os.path.basename(os.path.normpath(d))
    if b in ("gene_pathway", "both") or "hdca" in d.lower():
        return "HDCA"
    return b

def collect_npz(model_dirs):
    """(model, cset) -> list of (y_true, y_pred) per seed."""
    rows = {}
    for d in model_dirs:
        mdl = model_name(d)
        files = sorted(glob.glob(os.path.join(d, "crosspred_*.npz")))
        for f in files:
            cset = which_cset(os.path.basename(f))
            if cset is None:
                print(f"  [skip] could not infer dataset from {f}"); continue
            z = np.load(f)
            rows.setdefault((mdl, cset), []).append((z["y_true"], z["y_pred"]))
    return rows

def collect_drpreter(drp_dir):
    rows = {}
    for cset in ["CCLE", "gCSI"]:
        for f in sorted(glob.glob(os.path.join(drp_dir, f"cross_{cset}_seed*_pred.csv"))):
            df = pd.read_csv(f)
            rows.setdefault(("DRPreter", cset), []).append(
                (df["y_true"].values, df["y_pred"].values))
    return rows

def load_ref_pairs(ic_path):
    """Read the (cell, drug) pairs DRPreter evaluates from IC_<dataset>.csv."""
    df = pd.read_csv(ic_path)
    return set(zip(df["DepMap_ID"].astype(str), df["Drug name"].astype(str)))

def build_masks(subset_specs):
    """Build a boolean mask per dataset, in sample-table row order."""
    masks = {}
    for cset, (st_path, ic_path) in subset_specs.items():
        ref = load_ref_pairs(ic_path)
        st  = pd.read_csv(st_path)
        pairs = list(zip(st["model_id"].astype(str), st["drug_name"].astype(str)))
        mask = np.array([p in ref for p in pairs], dtype=bool)
        masks[cset] = mask
        print(f"  [subset] {cset}: sample table {len(mask)} rows -> {int(mask.sum())} shared "
              f"(DRPreter set {len(ref)})")
    return masks

def apply_masks(rows, masks):
    """Restrict every model except DRPreter to the shared pairs (npz rows follow the sample table)."""
    out = {}
    for (mdl, cset), seeds in rows.items():
        if mdl == "DRPreter" or cset not in masks:
            out[(mdl, cset)] = seeds
            continue
        m = masks[cset]
        new = []
        for yt, yp in seeds:
            if len(yt) != len(m):
                print(f"  [subset][warn] {mdl}/{cset}: npz len {len(yt)} != mask len {len(m)} "
                      f"-> skipping the mask (row alignment mismatch)")
                new.append((yt, yp))
            else:
                new.append((yt[m], yp[m]))
        out[(mdl, cset)] = new
    return out

def summarize(rows, cap):
    out = []
    for (mdl, cset), seeds in rows.items():
        raw = pd.DataFrame([metrics(yt, yp, cap=None) for yt, yp in seeds])
        win = pd.DataFrame([metrics(yt, yp, cap=cap)  for yt, yp in seeds])
        out.append(dict(
            model=mdl, cset=cset, n_seeds=len(seeds), n_test=int(raw["n"].iloc[0]),
            PCC_raw=raw["PCC"].mean(), PCC_win=win["PCC"].mean(), PCC_win_std=win["PCC"].std(),
            SCC_win=win["SCC"].mean(),
            RMSE_raw=raw["RMSE"].mean(), RMSE_win=win["RMSE"].mean(),
            MAE_win=win["MAE"].mean(),
        ))
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--drpreter_dir", default=None)
    ap.add_argument("--cap", type=float, default=13.82)
    ap.add_argument("--out", default="results/cross_winsorized_table.csv")
    ap.add_argument("--subset", action="store_true",
                    help="restrict every model to the pairs shared with DRPreter")
    ap.add_argument("--matrices_root", default="data",
                    help="parent directory of the per-dataset sample tables")
    ap.add_argument("--drpreter_ic_dir", default="DRPreter-main/Data_HDCA",
                    help="directory holding DRPreter's IC_CCLE.csv / IC_gCSI.csv")
    args = ap.parse_args()

    rows = collect_npz(args.model_dirs)
    if args.drpreter_dir:
        rows.update(collect_drpreter(args.drpreter_dir))
    if not rows:
        raise SystemExit("No crosspred_*.npz found; check the paths or run the models with --eval_only.")

    order = {"HDCA": 0, "DRPreter": 1, "PANCDR": 2, "DeepCDR": 3, "GraphDRP": 4, "TGSA": 5}
    pd.set_option("display.width", 160, "display.max_columns", 20)
    cols = ["model", "n_seeds", "n_test", "PCC_raw", "PCC_win",
            "SCC_win", "RMSE_raw", "RMSE_win", "MAE_win"]

    def _order_sort(d):
        d = d.copy()
        d["_o"] = d["model"].map(lambda m: order.get(m, 9))
        return d.sort_values(["cset", "_o"]).drop(columns="_o").reset_index(drop=True)

    def _print(d, title):
        for cset in ["CCLE", "gCSI"]:
            sub = d[d.cset == cset]
            if len(sub):
                print(f"\n===== cross_{cset}  {title}  (cap={args.cap}) =====")
                print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df = _order_sort(summarize(rows, args.cap))
    _print(df, "[FULL native test]")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")

    if args.subset:
        print("\n" + "=" * 70)
        subset_specs = {
            "CCLE": (os.path.join(args.matrices_root, "matrices_ccle_2015", "sample_table.csv"),
                     os.path.join(args.drpreter_ic_dir, "IC_CCLE.csv")),
            "gCSI": (os.path.join(args.matrices_root, "matrices_gcsi_2019", "sample_table.csv"),
                     os.path.join(args.drpreter_ic_dir, "IC_gCSI.csv")),
        }
        masks   = build_masks(subset_specs)
        df_sub  = _order_sort(summarize(apply_masks(rows, masks), args.cap))
        _print(df_sub, "[SUBSET: pairs shared with DRPreter]")
        sub_out = args.out.replace(".csv", "_subset.csv")
        df_sub.to_csv(sub_out, index=False)
        print(f"\nsaved: {sub_out}")

    print("\nNote: raw = no cap applied; win = winsorized. FULL uses each model's native\n"
          "test set, SUBSET the pairs shared with DRPreter.")

if __name__ == "__main__":
    main()
