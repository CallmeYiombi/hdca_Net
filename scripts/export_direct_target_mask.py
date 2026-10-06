#!/usr/bin/env python3
r"""Export the direct-target gene mask as a compact, checkable text artifact.

`build_pruned_mask.py` documents how the mask is derived from the GDSC putative-target
annotation. This script exports the matrix the reported models were actually trained on, as a
(drug, gene) edge list of a few thousand rows rather than a 42 MB dense array, and verifies
that the edge list rebuilds that matrix bit for bit.

  python scripts/export_direct_target_mask.py [--verify]
"""
import os
import json
import hashlib
import argparse
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild(edges, drug2idx, gene2idx, shape, dtype):
    m = np.zeros(shape, dtype=dtype)
    rows = edges["drug_name"].map(drug2idx).to_numpy()
    cols = edges["gene_symbol"].map(gene2idx).to_numpy()
    m[rows, cols] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--mask", default="hcdt_drug_gene_pruned.npy")
    ap.add_argument("--out", default="data/direct_target_mask.csv")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing export instead of writing one")
    a = ap.parse_args()

    idm = json.load(open(os.path.join(a.mat_dir, "id_maps.json")))
    gene2idx = idm["gene_symbol_to_idx"]
    drug2idx = idm["drug_name_to_idx"]
    idx2gene = {v: k for k, v in gene2idx.items()}
    idx2drug = {v: k for k, v in drug2idx.items()}

    mask_path = os.path.join(a.mat_dir, a.mask)
    mask = np.load(mask_path)
    print(f"{mask_path}\n  shape {mask.shape}  sha256 {sha256(mask_path)}")
    print(f"  annotated compounds {int((mask.sum(1) > 0).sum())}, "
          f"edges {int(mask.sum())}")

    if a.verify:
        edges = pd.read_csv(a.out)
    else:
        r, c = np.nonzero(mask)
        edges = pd.DataFrame({"drug_name": [idx2drug[i] for i in r],
                              "gene_symbol": [idx2gene[j] for j in c]})
        edges = edges.sort_values(["drug_name", "gene_symbol"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        edges.to_csv(a.out, index=False)
        print(f"wrote {a.out}  ({len(edges)} edges, "
              f"{os.path.getsize(a.out) / 1024:.0f} KB)")

    same = np.array_equal(rebuild(edges, drug2idx, gene2idx, mask.shape, mask.dtype), mask)
    print(f"round-trip rebuilds the trained mask exactly: {same}")
    if not same:
        raise SystemExit("export does NOT reproduce the mask -- do not release it")


if __name__ == "__main__":
    main()
