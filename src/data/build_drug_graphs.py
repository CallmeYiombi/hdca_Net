"""Resolve compound SMILES and build molecular graph arrays.

Drug names from sample_table.csv are resolved to canonical SMILES through the PubChem REST
API (name -> SMILES, then name -> CID -> SMILES) with ChEMBL as a fallback, trying several
name variants per compound. PubChem periodically renames its SMILES properties, so each
lookup tries IsomericSMILES, CanonicalSMILES, SMILES and ConnectivitySMILES in turn; an
earlier fetcher that queried IsomericSMILES alone silently failed for most compounds and
produced the all-zero fingerprints described in build_drug_fp.py.

Each resolved molecule is converted with RDKit to atom features and a row-normalised
adjacency matrix, padded to --max_atoms, and written alongside the SMILES table:

    drug_atom_feats.npy  (D, MAX_ATOMS, 49)
    drug_adj_norm.npy    (D, MAX_ATOMS, MAX_ATOMS)
    drug_mask.npy        (D, MAX_ATOMS)  bool
    drug_smiles.csv      SMILES lookup table, consumed by build_drug_fp.py

Only drug_smiles.csv is needed for HDCA-Net and the fingerprint baselines; the graph arrays
serve the graph-based baselines.

Usage:
    python src/data/build_drug_graphs.py --mat_dir data/matrices_gdsc12 [--max_atoms 100]
    python src/data/build_drug_graphs.py --resume   # re-fetch only the unresolved compounds
"""

import argparse
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

_ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
_DEGREES = list(range(11))
_FORMAL_CHG = [-3, -2, -1, 0, 1, 2, 3]
_NUM_HS = [0, 1, 2, 3, 4]
_HYBRID = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]

ATOM_FEAT_DIM = ((len(_ATOM_TYPES) + 1) + (len(_DEGREES) + 1)
                 + (len(_FORMAL_CHG) + 1) + (len(_NUM_HS) + 1)
                 + (len(_HYBRID) + 1) + 1)


def _one_hot(val, allowed):
    """One-hot over `allowed` with a trailing bucket for unseen values."""
    vec = [0] * (len(allowed) + 1)
    if val in allowed:
        vec[allowed.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def _atom_features(atom):
    return (
        _one_hot(atom.GetAtomicNum(), _ATOM_TYPES)
        + _one_hot(atom.GetDegree(), _DEGREES)
        + _one_hot(int(atom.GetFormalCharge()), _FORMAL_CHG)
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)
        + _one_hot(atom.GetHybridization(), _HYBRID)
        + [int(atom.GetIsAromatic())]
    )


def mol_to_arrays(mol, max_atoms):
    """Returns (atom_feats, adj_norm, mask) with self-loops and row normalisation."""
    n = min(mol.GetNumAtoms(), max_atoms)

    feats = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        feats[i] = _atom_features(atom)

    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            adj[i, j] = adj[j, i] = 1.0
    for i in range(n):
        adj[i, i] = 1.0

    row_sum = adj.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = (adj / row_sum).astype(np.float32)

    mask = np.zeros(max_atoms, dtype=bool)
    mask[:n] = True
    return feats, adj_norm, mask


_SMILES_PROPS = ["IsomericSMILES", "CanonicalSMILES", "SMILES", "ConnectivitySMILES"]
_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def _fetch_json(url, retries=4):
    """GET JSON with exponential backoff. None on 400/404; retries on throttling."""
    delay = 0.5
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
        except Exception:
            pass
        time.sleep(delay)
        delay *= 2
    return None


def _pubchem_smiles(kind, ident):
    """kind is 'name' or 'cid'; tries each SMILES property name in turn."""
    base = f"{_PUBCHEM}/{kind}/{urllib.parse.quote(str(ident))}/property"
    for prop in _SMILES_PROPS:
        data = _fetch_json(f"{base}/{prop}/JSON")
        if not data:
            continue
        try:
            val = data["PropertyTable"]["Properties"][0].get(prop)
            if val:
                return val
        except Exception:
            pass
    return None


def _pubchem_cid_by_name(name):
    data = _fetch_json(f"{_PUBCHEM}/name/{urllib.parse.quote(name)}/cids/JSON")
    try:
        return str(data["IdentifierList"]["CID"][0])
    except Exception:
        return None


def _get_from_chembl(name):
    encoded = urllib.parse.quote(name)
    data = _fetch_json(f"https://www.ebi.ac.uk/chembl/api/data/molecule"
                       f"?pref_name__iexact={encoded}&format=json&limit=1")
    if not data:
        return None
    mols = data.get("molecules", [])
    if not mols:
        return None
    struct = mols[0].get("molecule_structures") or {}
    return struct.get("canonical_smiles")


_ALIASES = {
    "venotoclax": "venetoclax",
    "picolinici-acid": "picolinic acid",
}


def _name_candidates(name):
    """Ordered name variants: known typos, parenthetical suffixes, comma synonyms."""
    name = name.strip().lower()
    cands = []

    def add(x):
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in cands:
            cands.append(x)

    add(name)
    if name in _ALIASES:
        add(_ALIASES[name])
    add(re.sub(r"\s*\([^)]*\)\s*", " ", name))
    for part in name.split(","):
        add(part)
        add(re.sub(r"\s*\([^)]*\)\s*", " ", part))
    for inner in re.findall(r"\(([^)]*)\)", name):
        add(inner)
    return cands


def _resolve_one(name):
    name = name.strip()
    if name.isdigit():
        s = _pubchem_smiles("cid", name)
        if s:
            return s
    s = _pubchem_smiles("name", name)
    if s:
        return s
    cid = _pubchem_cid_by_name(name)
    if cid:
        s = _pubchem_smiles("cid", cid)
        if s:
            return s
    return _get_from_chembl(name)


def fetch_smiles(name):
    for cand in _name_candidates(name):
        s = _resolve_one(cand)
        if s:
            return s
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_dir", default="data/matrices_gdsc12")
    parser.add_argument("--max_atoms", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="keep drugs already resolved in drug_smiles.csv and "
                             "only re-fetch the failed/empty ones")
    parser.add_argument("--sleep", type=float, default=0.34,
                        help="delay between API calls (s)")
    args = parser.parse_args()

    sample_table = pd.read_csv(os.path.join(args.mat_dir, "sample_table.csv"))
    drug_map = (
        sample_table[["drug_idx", "drug_name_lower"]]
        .drop_duplicates()
        .sort_values("drug_idx")
        .reset_index(drop=True)
    )
    n_drugs = len(drug_map)
    max_atoms = args.max_atoms

    ff = os.path.join(args.mat_dir, "drug_atom_feats.npy")
    fa = os.path.join(args.mat_dir, "drug_adj_norm.npy")
    fm = os.path.join(args.mat_dir, "drug_mask.npy")
    fs = os.path.join(args.mat_dir, "drug_smiles.csv")

    prev_smiles = {}
    if args.resume and all(os.path.exists(p) for p in (ff, fa, fm, fs)):
        atom_feats_all = np.load(ff)
        adj_norm_all = np.load(fa)
        mask_all = np.load(fm)
        prev = pd.read_csv(fs).fillna({"smiles": ""})
        prev_smiles = dict(zip(prev["drug_idx"].astype(int), prev["smiles"].astype(str)))
        n_have = sum(1 for v in prev_smiles.values() if v)
        print(f"[resume] loaded {ff} + {fs}: {n_have}/{n_drugs} already resolved; "
              f"re-fetching the rest.")
    else:
        atom_feats_all = np.zeros((n_drugs, max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
        adj_norm_all = np.zeros((n_drugs, max_atoms, max_atoms), dtype=np.float32)
        mask_all = np.zeros((n_drugs, max_atoms), dtype=bool)

    records, failed = [], []
    print(f"Fetching SMILES for {n_drugs} drugs (PubChem -> ChEMBL)...")

    for _, row in drug_map.iterrows():
        idx = int(row["drug_idx"])
        name = row["drug_name_lower"]

        prev = prev_smiles.get(idx, "")
        if prev and Chem.MolFromSmiles(prev) is not None:
            records.append({"drug_idx": idx, "drug_name": name, "smiles": prev})
            continue

        smiles = fetch_smiles(name)
        time.sleep(args.sleep)

        if smiles is None:
            print(f"  [FAIL]  {name}")
            failed.append(name)
            records.append({"drug_idx": idx, "drug_name": name, "smiles": ""})
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [BAD SMILES]  {name}  {smiles}")
            failed.append(name)
            records.append({"drug_idx": idx, "drug_name": name, "smiles": smiles})
            continue

        feats, adj_norm, mask = mol_to_arrays(mol, max_atoms)
        atom_feats_all[idx] = feats
        adj_norm_all[idx] = adj_norm
        mask_all[idx] = mask
        print(f"  [OK]  {name:30s}  atoms={mol.GetNumAtoms()}")
        records.append({"drug_idx": idx, "drug_name": name, "smiles": smiles})

    np.save(ff, atom_feats_all)
    np.save(fa, adj_norm_all)
    np.save(fm, mask_all)
    pd.DataFrame(records).to_csv(fs, index=False)

    resolved = sum(1 for r in records if r["smiles"])
    print(f"\nSaved to {args.mat_dir}/")
    print(f"  drug_atom_feats.npy : {atom_feats_all.shape}")
    print(f"  drug_adj_norm.npy   : {adj_norm_all.shape}")
    print(f"  drug_mask.npy       : {mask_all.shape}")
    print(f"  drug_smiles.csv     : {resolved}/{n_drugs} resolved")
    if failed:
        print(f"\n  WARNING: {len(failed)} drugs had no usable SMILES and keep "
              f"zero vectors: {failed}")


if __name__ == "__main__":
    main()
