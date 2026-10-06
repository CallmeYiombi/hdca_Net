#!/usr/bin/env python3
"""Recompute the overlap figures behind the Limitations statement that the 144-compound panel
is not independent of the supplied annotation.

Definitions
  gene-level: of the union of DGIdb target genes over the 144 evaluated compounds, the share
              that also appears somewhere in the drug--gene mask given to the model
  drug-level: of the 144 compounds, the share whose own mask row shares at least one gene with
              its DGIdb targets

The panel selection (542 -> 144) reuses the logic of moa_expand_chembl.py, so the compound set
matches the one used in the paper. Run with --probe to print the function signatures first.

    python scripts/dgidb_overlap.py --probe
    python scripts/dgidb_overlap.py --db data/interactions.tsv
"""
import argparse
import inspect
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def probe():
    import moa_expand_chembl as M
    print("=== public functions of moa_expand_chembl ===")
    for name, fn in vars(M).items():
        if callable(fn) and not name.startswith("_") and getattr(fn, "__module__", "") == M.__name__:
            try:
                print(f"  {name}{inspect.signature(fn)}")
            except (TypeError, ValueError):
                print(f"  {name}(?)")
    print("\n=== call sites in eval_prior_randomization.py ===")
    for cand in ("scripts/eval_prior_randomization.py", "results/eval_prior_randomization.py"):
        if os.path.exists(cand):
            for i, line in enumerate(open(cand), 1):
                if "load_targets" in line or "build_masks" in line or "max_path_size" in line:
                    print(f"  {cand}:{i}: {line.rstrip()}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="print the internal function signatures and exit")
    ap.add_argument("--db", default="data/interactions.tsv")
    ap.add_argument("--matrices", default="data/matrices_gdsc12")
    ap.add_argument("--mask", default="hcdt_drug_gene_pruned.npy",
                    help="the drug--gene mask actually supplied to the model")
    ap.add_argument("--topn", type=int, default=2)
    ap.add_argument("--max_path_size", type=int, default=15)
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    import json
    import moa_expand_chembl as M

    st = pd.read_csv(os.path.join(args.matrices, "sample_table.csv"))
    col = "drug_name_lower" if "drug_name_lower" in st.columns else None
    panel = (set(st[col].astype(str).unique()) if col
             else set(st["drug_name"].astype(str).str.lower().str.strip().unique()))

    ids = json.load(open(os.path.join(args.matrices, "id_maps.json")))
    sym2idx = ids["gene_symbol_to_idx"]
    drug2idx = {k.lower(): v for k, v in ids["drug_name_to_idx"].items()}

    # signature: load_targets_dgidb(tsv, panel, sym2idx, topn), as in eval_prior_randomization
    tgt, _ = M.load_targets_dgidb(args.db, panel, sym2idx, args.topn)

    # the 144 evaluated compounds are those with a specific pathway, per the original logic
    gp = np.load(os.path.join(args.matrices, "gene_pathway.npy"))
    masks = M.build_masks(tgt, sym2idx, gp, (gp > 0).sum(0), args.max_path_size)
    evaluable = [d for d, v in masks.items() if len(v) > 0]
    print(f"selection: {len(tgt)} matched in the database -> {len(evaluable)} with a specific pathway")

    supplied = np.load(os.path.join(args.matrices, args.mask))
    anywhere = supplied.sum(0) > 0

    def idx_of(g):
        return g if isinstance(g, (int, np.integer)) else sym2idx[g]

    genes = sorted({idx_of(g) for d in evaluable for g in tgt[d]})
    in_any = sum(1 for gi in genes if anywhere[gi])
    print(f"\n[gene-level] {in_any} of {len(genes)} target genes appear in the mask "
          f"= {100 * in_any / len(genes):.1f}%")

    own = miss = 0
    for d in evaluable:
        if d not in drug2idx:
            miss += 1
            continue
        row = supplied[drug2idx[d]]
        if any(row[idx_of(g)] for g in tgt[d]):
            own += 1
    n = len(evaluable) - miss
    print(f"[drug-level] {own} of {n} compounds overlap their own mask row = {100 * own / n:.1f}%"
          + (f"   ({miss} compounds absent from id_maps were excluded)" if miss else ""))

    print("\nLimitations in the paper quote 69% of genes and 84% of compounds.")


if __name__ == "__main__":
    main()
