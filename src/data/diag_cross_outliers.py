"""Diagnose extrapolated IC50 values in the external datasets.

PharmacoGx recomputes IC50 by extrapolation when a compound never reaches 50% inhibition,
which produces a small number of extreme values outside the GDSC training support. This script
quantifies them and their effect on Pearson correlation.
"""
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_table", required=True)
    ap.add_argument("--tag", default="")

    ap.add_argument("--hi", type=float, default=15.0)
    args = ap.parse_args()

    df = pd.read_csv(args.sample_table)
    y = df["ln_ic50"].astype(float)
    print(f"\n===== [{args.tag}] {args.sample_table}  (n={len(df):,}) =====")

    qs = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0]
    print("ln_ic50 quantiles:")
    for q in qs:
        print(f"  {q*100:7.2f}% : {y.quantile(q):+10.4f}")

    print(f"\nCounts above the GDSC support cap (~{args.hi}):")
    for thr in [args.hi, 20, 30, 50, 100]:
        n = int((y > thr).sum())
        print(f"  ln_ic50 > {thr:>5.0f} : {n:>6d}  ({100*n/len(df):.4f}%)   [response=e^{thr:.0f} ~ {np.exp(thr):.2e}]")

    ext = df.loc[y > args.hi].copy()
    ext["response"] = np.exp(ext["ln_ic50"])
    ext = ext.sort_values("ln_ic50", ascending=False)
    cols = [c for c in ["drug_name", "model_id", "cell_name", "ln_ic50", "response"] if c in ext.columns]
    print(f"\nExtreme records (ln_ic50 > {args.hi}): {len(ext)} total, top 15:")
    if len(ext):
        print(ext[cols].head(15).to_string(index=False))

        print(f"\nDrugs carrying the extreme values (top 15):")
        by_drug = ext.groupby("drug_name").agg(
            n_extreme=("ln_ic50", "size"),
            max_ln=("ln_ic50", "max"),
        ).sort_values("n_extreme", ascending=False)
        print(by_drug.head(15).to_string())

        top_drug = by_drug.index[0]
        d_all = df.loc[df["drug_name"] == top_drug, "ln_ic50"]
        print(f"\nFull ln_ic50 distribution for '{top_drug}': "
              f"n={len(d_all)}  min={d_all.min():+.3f}  median={d_all.median():+.3f}  max={d_all.max():+.3f}")
    else:
        print("  (none: no extreme outliers in this dataset)")

if __name__ == "__main__":
    main()
