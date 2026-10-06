"""
Diagnose extreme ln_ic50 outliers in the cross-study cohorts.

The CCLE and gCSI targets reach ln_ic50 of +143.10 and +73.24, which correspond to responses of
e^143 and are not physically meaningful. This script reports how many such records there are,
which drug and cell line they belong to, whether they concentrate on particular compounds, and
how far they sit outside the normal GDSC range of roughly [-12, +14].

Usage:
  python src/data/diag_cross_outliers.py --sample_table data/matrices_ccle_2015/sample_table.csv --tag CCLE
  python src/data/diag_cross_outliers.py --sample_table data/matrices_gcsi_2019/sample_table.csv --tag gCSI
  # baseline check on GDSC12:
  python src/data/diag_cross_outliers.py --sample_table data/matrices_gdsc12/sample_table.csv --tag GDSC12
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_table", required=True)
    ap.add_argument("--tag", default="")
    # approximate upper end of the normal GDSC ln_ic50 range; anything above is a candidate
    ap.add_argument("--hi", type=float, default=15.0)
    args = ap.parse_args()

    df = pd.read_csv(args.sample_table)
    y = df["ln_ic50"].astype(float)
    print(f"\n===== [{args.tag}] {args.sample_table}  (n={len(df):,}) =====")

    # --- quantiles ---
    qs = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0]
    print("ln_ic50 quantiles:")
    for q in qs:
        print(f"  {q*100:7.2f}% : {y.quantile(q):+10.4f}")

    # --- counts above the threshold ---
    print(f"\ncounts above the threshold (normal GDSC upper end ~{args.hi}):")
    for thr in [args.hi, 20, 30, 50, 100]:
        n = int((y > thr).sum())
        print(f"  ln_ic50 > {thr:>5.0f} : {n:>6d}  ({100*n/len(df):.4f}%)   [response=e^{thr:.0f} ≈ {np.exp(thr):.2e}]")

    # --- the extreme records ---
    ext = df.loc[y > args.hi].copy()
    ext["response"] = np.exp(ext["ln_ic50"])
    ext = ext.sort_values("ln_ic50", ascending=False)
    cols = [c for c in ["drug_name", "model_id", "cell_name", "ln_ic50", "response"] if c in ext.columns]
    print(f"\n{len(ext)} extreme records (ln_ic50 > {args.hi}); top 15:")
    if len(ext):
        print(ext[cols].head(15).to_string(index=False))

        # --- per-compound counts, to catch a unit error on one compound ---
        print(f"\ncompounds the extreme values concentrate on (top 15):")
        by_drug = ext.groupby("drug_name").agg(
            n_extreme=("ln_ic50", "size"),
            max_ln=("ln_ic50", "max"),
        ).sort_values("n_extreme", ascending=False)
        print(by_drug.head(15).to_string())

        # does that compound also have in-range records? a unit error would affect all of them
        top_drug = by_drug.index[0]
        d_all = df.loc[df["drug_name"] == top_drug, "ln_ic50"]
        print(f"\nfull ln_ic50 distribution for the worst compound '{top_drug}': "
              f"n={len(d_all)}  min={d_all.min():+.3f}  median={d_all.median():+.3f}  max={d_all.max():+.3f}")
    else:
        print("  (none; this dataset has no extreme outliers)")


if __name__ == "__main__":
    main()
