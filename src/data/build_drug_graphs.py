"""
Build molecular graph data for GMP-HCPNet.

1. Reads drug names from sample_table.csv
2. Fetches canonical SMILES from PubChem REST API (ChEMBL fallback)
3. Converts each molecule to atom features + row-normalised adjacency using RDKit
4. Pads to MAX_ATOMS and saves:
     data/matrices/drug_atom_feats.npy  (D, MAX_ATOMS, 45)
     data/matrices/drug_adj_norm.npy    (D, MAX_ATOMS, MAX_ATOMS)
     data/matrices/drug_mask.npy        (D, MAX_ATOMS)  bool
     data/matrices/drug_smiles.csv      SMILES lookup table

Usage:
    python src/data/build_drug_graphs.py --mat_dir data/matrices [--max_atoms 100]
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

# ── Atom feature constants ─────────────────────────────────────────────────────

_ATOM_TYPES  = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]   # H B C N O F Si P S Cl Se Br I
_DEGREES     = list(range(11))                                      # 0–10
_FORMAL_CHG  = [-3, -2, -1, 0, 1, 2, 3]
_NUM_HS      = [0, 1, 2, 3, 4]
_HYBRID      = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]

# Every _one_hot() appends a trailing "unknown" bucket (+1); +1 more for the
# aromatic flag. Must match _atom_features() exactly or feats broadcast fails.
ATOM_FEAT_DIM = ((len(_ATOM_TYPES) + 1) + (len(_DEGREES) + 1)
                 + (len(_FORMAL_CHG) + 1) + (len(_NUM_HS) + 1)
                 + (len(_HYBRID) + 1) + 1)   # 14+12+8+6+8+1 = 49


def _one_hot(val, allowed):
    vec = [0] * (len(allowed) + 1)          # last index = "unknown"
    if val in allowed:
        vec[allowed.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def _atom_features(atom) -> list:
    return (
        _one_hot(atom.GetAtomicNum(), _ATOM_TYPES)                  # 14
        + _one_hot(atom.GetDegree(), _DEGREES)                      # 11
        + _one_hot(int(atom.GetFormalCharge()), _FORMAL_CHG)        # 7
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)                   # 5
        + _one_hot(atom.GetHybridization(), _HYBRID)                # 7
        + [int(atom.GetIsAromatic())]                               # 1
    )                                                               # = 45


# ── Molecule → graph arrays ────────────────────────────────────────────────────

def mol_to_arrays(mol, max_atoms: int):
    """Returns (atom_feats, adj_norm, mask) as float32 / bool numpy arrays."""
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
    for i in range(n):                        # self-loops
        adj[i, i] = 1.0

    row_sum = adj.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = (adj / row_sum).astype(np.float32)

    mask = np.zeros(max_atoms, dtype=bool)
    mask[:n] = True

    return feats, adj_norm, mask


# ── SMILES fetching ───────────────────────────────────────────────────────────

# PubChem periodically renames SMILES properties (Isomeric/Canonical -> SMILES/
# ConnectivitySMILES). Try them in order so the fetch survives such renames.
_SMILES_PROPS = ["IsomericSMILES", "CanonicalSMILES", "SMILES", "ConnectivitySMILES"]
_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def _fetch_json(url: str, retries: int = 4):
    """GET JSON with backoff. Returns None on genuine 404/400; retries on
    throttling (429/503/500), timeouts and transient network errors."""
    delay = 0.5
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None                 # genuine "not found" — don't retry
        except Exception:
            pass                            # timeout / URLError / server busy
        time.sleep(delay)
        delay *= 2
    return None


def _pubchem_smiles(kind: str, ident: str) -> str | None:
    """kind = 'name' or 'cid'. Tries each SMILES property name in turn."""
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


def _pubchem_cid_by_name(name: str) -> str | None:
    data = _fetch_json(f"{_PUBCHEM}/name/{urllib.parse.quote(name)}/cids/JSON")
    try:
        return str(data["IdentifierList"]["CID"][0])
    except Exception:
        return None


def _get_from_chembl(name: str) -> str | None:
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


# Known typos / alias fixes for GDSC drug names.
_ALIASES = {
    "venotoclax": "venetoclax",
    "picolinici-acid": "picolinic acid",
}


def _name_candidates(name: str):
    """Ordered, de-duplicated name variants to try (handles concentration/stereo
    suffixes in parens, comma-separated synonyms, and known typos)."""
    name = name.strip().lower()
    cands = []

    def add(x):
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in cands:
            cands.append(x)

    add(name)
    if name in _ALIASES:
        add(_ALIASES[name])
    add(re.sub(r"\s*\([^)]*\)\s*", " ", name))          # drop "(10 um)", "(-)" ...
    for part in name.split(","):                          # "brivanib, bms-540215"
        add(part)
        add(re.sub(r"\s*\([^)]*\)\s*", " ", part))
    for inner in re.findall(r"\(([^)]*)\)", name):       # "ascorbate (vitamin c)"
        add(inner)
    return cands


def _resolve_one(name: str) -> str | None:
    name = name.strip()
    # Numeric drug names are often PubChem CIDs (or NSC ids) — try CID lookup.
    if name.isdigit():
        s = _pubchem_smiles("cid", name)
        if s:
            return s
    # name -> SMILES directly
    s = _pubchem_smiles("name", name)
    if s:
        return s
    # name -> CID -> SMILES (catches synonyms the property endpoint misses)
    cid = _pubchem_cid_by_name(name)
    if cid:
        s = _pubchem_smiles("cid", cid)
        if s:
            return s
    # last resort: ChEMBL by preferred name
    return _get_from_chembl(name)


def fetch_smiles(name: str) -> str | None:
    for cand in _name_candidates(name):
        s = _resolve_one(cand)
        if s:
            return s
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_dir",   default="data/matrices")
    parser.add_argument("--max_atoms", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="keep drugs already resolved in drug_smiles.csv and "
                             "only re-fetch the failed/empty ones")
    parser.add_argument("--sleep", type=float, default=0.34,
                        help="polite delay between API calls (s)")
    args = parser.parse_args()

    sample_table = pd.read_csv(os.path.join(args.mat_dir, "sample_table.csv"))

    # Build drug index mapping
    drug_map = (
        sample_table[["drug_idx", "drug_name_lower"]]
        .drop_duplicates()
        .sort_values("drug_idx")
        .reset_index(drop=True)
    )
    n_drugs   = len(drug_map)
    max_atoms = args.max_atoms

    ff = os.path.join(args.mat_dir, "drug_atom_feats.npy")
    fa = os.path.join(args.mat_dir, "drug_adj_norm.npy")
    fm = os.path.join(args.mat_dir, "drug_mask.npy")
    fs = os.path.join(args.mat_dir, "drug_smiles.csv")

    prev_smiles = {}
    if args.resume and all(os.path.exists(p) for p in (ff, fa, fm, fs)):
        atom_feats_all = np.load(ff)
        adj_norm_all   = np.load(fa)
        mask_all       = np.load(fm)
        prev = pd.read_csv(fs).fillna({"smiles": ""})
        prev_smiles = dict(zip(prev["drug_idx"].astype(int), prev["smiles"].astype(str)))
        n_have = sum(1 for v in prev_smiles.values() if v)
        print(f"[resume] loaded {ff} + {fs}: {n_have}/{n_drugs} already resolved; "
              f"re-fetching the rest.")
    else:
        atom_feats_all = np.zeros((n_drugs, max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
        adj_norm_all   = np.zeros((n_drugs, max_atoms, max_atoms),     dtype=np.float32)
        mask_all       = np.zeros((n_drugs, max_atoms),                dtype=bool)

    records        = []
    failed         = []

    print(f"Fetching SMILES for {n_drugs} drugs (PubChem → ChEMBL)...")

    for _, row in drug_map.iterrows():
        idx  = int(row["drug_idx"])
        name = row["drug_name_lower"]

        # resume: skip drugs already resolved to a valid SMILES
        prev = prev_smiles.get(idx, "")
        if prev and Chem.MolFromSmiles(prev) is not None:
            records.append({"drug_idx": idx, "drug_name": name, "smiles": prev})
            continue

        smiles = fetch_smiles(name)
        time.sleep(args.sleep)                # be polite to APIs

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
        adj_norm_all[idx]   = adj_norm
        mask_all[idx]       = mask

        n_atoms = mol.GetNumAtoms()
        print(f"  [OK]  {name:30s}  atoms={n_atoms}")
        records.append({"drug_idx": idx, "drug_name": name, "smiles": smiles})

    # Save
    np.save(os.path.join(args.mat_dir, "drug_atom_feats.npy"), atom_feats_all)
    np.save(os.path.join(args.mat_dir, "drug_adj_norm.npy"),   adj_norm_all)
    np.save(os.path.join(args.mat_dir, "drug_mask.npy"),       mask_all)
    pd.DataFrame(records).to_csv(
        os.path.join(args.mat_dir, "drug_smiles.csv"), index=False
    )

    print(f"\nSaved to {args.mat_dir}/")
    print(f"  drug_atom_feats.npy : {atom_feats_all.shape}")
    print(f"  drug_adj_norm.npy   : {adj_norm_all.shape}")
    print(f"  drug_mask.npy       : {mask_all.shape}")
    if failed:
        print(f"\n  WARNING: {len(failed)} drugs had no SMILES — "
              f"will use zero-vectors: {failed}")


if __name__ == "__main__":
    main()
