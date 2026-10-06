"""Empirical leakage audit for the splits and the cross-study evaluation.

Checks, in order:
  1. duplicate (drug, cell) rows in the GDSC sample table -- random_split is
     row-level disjoint, so the paper's "no identical pair in more than one fold"
     claim holds only if the table is already de-duplicated;
  2. train/val/test index disjointness for the reported random split;
  3. drug disjointness for every drug-5-fold fold;
  4. how much of each external cohort's (cell, drug) combinations, cells and
     drugs recur in GDSC training -- the recurrence the paper discloses;
  5. whether any external pair carries a target identical to a GDSC one, which
     would be genuine label leakage rather than mere combination recurrence.

Usage: python scripts/leakage_audit.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.split import drug_kfold, random_split  # noqa: E402

GDSC = "data/matrices_gdsc12"
EXTERNAL = {"CCLE": "data/matrices_ccle_2015", "gCSI": "data/matrices_gcsi_2019"}


def main():
    st = pd.read_csv(os.path.join(GDSC, "sample_table.csv"))
    print(f"GDSC sample table: {len(st):,} rows\n")

    # 1. duplicate (drug, cell) pairs
    key = ["drug_name_lower", "model_id"] if "model_id" in st else ["drug_idx", "cell_idx"]
    dup = st.duplicated(subset=key).sum()
    print(f"1. duplicate (drug, cell) rows on {key}: {dup}"
          f"   -> {'OK, table is de-duplicated' if dup == 0 else 'LEAKAGE RISK'}")

    # 2. random split disjointness (the reported configuration)
    tr, va, te = random_split(st, seed=42)
    print(f"2. random split sizes {len(tr):,}/{len(va):,}/{len(te):,}   "
          f"overlaps tr&te {len(set(tr) & set(te))}, tr&va {len(set(tr) & set(va))}, "
          f"va&te {len(set(va) & set(te))}")
    pair = st[key].apply(tuple, axis=1).values
    shared = set(pair[tr]) & set(pair[te])
    print(f"   (drug, cell) combinations shared between train and test: {len(shared)}")

    # 3. drug 5-fold disjointness
    print("3. drug 5-fold:")
    for fold, tr_i, va_i, te_i in drug_kfold(st, n_splits=5, seed=42):
        d = lambda ix: set(st.loc[ix, "drug_name_lower"])
        print(f"   fold {fold}: drugs tr {len(d(tr_i))} va {len(d(va_i))} te {len(d(te_i))} | "
              f"tr&te {len(d(tr_i) & d(te_i))}  va&te {len(d(va_i) & d(te_i))}  "
              f"tr&va {len(d(tr_i) & d(va_i))}")

    # 4/5. external cohort recurrence and target identity
    gd_pairs = dict(zip(zip(st["drug_name_lower"], st["model_id"]), st["ln_ic50"])) \
        if "model_id" in st else {}
    gd_drugs, gd_cells = set(st["drug_name_lower"]), set(st.get("model_id", []))
    print("\n4/5. external cohorts vs GDSC training data:")
    for name, path in EXTERNAL.items():
        f = os.path.join(path, "sample_table.csv")
        if not os.path.exists(f):
            print(f"   {name}: {f} not present locally -- skipped")
            continue
        ex = pd.read_csv(f)
        n = len(ex)
        drug_rec = ex["drug_name_lower"].isin(gd_drugs).mean() * 100
        cell_rec = ex["model_id"].isin(gd_cells).mean() * 100
        combos = list(zip(ex["drug_name_lower"], ex["model_id"]))
        combo_rec = np.mean([c in gd_pairs for c in combos]) * 100
        same = sum(1 for c, y in zip(combos, ex["ln_ic50"])
                   if c in gd_pairs and np.isclose(gd_pairs[c], y, atol=1e-9))
        print(f"   {name}: {n:,} pairs | drug recurs {drug_rec:.0f}%  "
              f"cell recurs {cell_rec:.0f}%  (cell,drug) combo recurs {combo_rec:.0f}%")
        print(f"      pairs whose target value is IDENTICAL to the GDSC one: {same}"
              f"   -> {'OK, different assay values' if same == 0 else 'LABEL LEAKAGE'}")


if __name__ == "__main__":
    main()
