"""
Rebuild 2048-bit Morgan fingerprints (drug_fp.npy) from an existing
drug_smiles.csv (produced by build_drug_graphs.py).

Motivation
----------
The original drug_fp.npy was built with a broken PubChem SMILES fetcher and had
~72% (394/542) all-zero fingerprints, which silently crippled the fingerprint-only
baselines. Reusing the SMILES already recovered for the drug graphs keeps FP and
graph inputs consistent and restores coverage.

Usage
-----
PYTHONPATH=. python src/data/build_drug_fp.py --mat_dir data/matrices_gdsc12
# writes drug_fp.npy (radius=2, 2048 bits); backs up the old one to drug_fp_oldzero.npy
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def morgan_bits(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", default="data/matrices_gdsc12")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=2048)
    args = ap.parse_args()

    smiles_csv = os.path.join(args.mat_dir, "drug_smiles.csv")
    fp_path = os.path.join(args.mat_dir, "drug_fp.npy")
    if not os.path.exists(smiles_csv):
        raise SystemExit(f"{smiles_csv} not found -- run build_drug_graphs.py first")

    if os.path.exists(fp_path):
        n_drugs = np.load(fp_path).shape[0]
    else:
        st = pd.read_csv(os.path.join(args.mat_dir, "sample_table.csv"))
        n_drugs = int(st["drug_idx"].max()) + 1

    df = pd.read_csv(smiles_csv).fillna({"smiles": ""})
    out = np.zeros((n_drugs, args.n_bits), dtype=np.float32)
    ok, bad = 0, []
    for _, row in df.iterrows():
        idx = int(row["drug_idx"])
        smi = str(row["smiles"]).strip()
        if not smi:
            bad.append(row.get("drug_name", idx))
            continue
        arr = morgan_bits(smi, args.radius, args.n_bits)
        if arr is None:
            bad.append(row.get("drug_name", idx))
            continue
        out[idx] = arr
        ok += 1

    if os.path.exists(fp_path):
        backup = os.path.join(args.mat_dir, "drug_fp_oldzero.npy")
        if not os.path.exists(backup):
            shutil.copy2(fp_path, backup)
            print(f"backed up old drug_fp.npy -> {backup}")

    np.save(fp_path, out)
    nonzero = int((np.abs(out).sum(1) > 0).sum())
    print(f"saved {fp_path}  shape={out.shape}  radius={args.radius}")
    print(f"nonzero fingerprints: {nonzero}/{n_drugs}  (resolved SMILES: {ok})")
    if bad:
        print(f"still empty ({len(bad)}): {bad[:20]}{' ...' if len(bad) > 20 else ''}")

if __name__ == "__main__":
    main()
