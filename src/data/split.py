"""
Split strategies for PG-HCPNet.
All return (train_idx, val_idx, test_idx) as numpy arrays of row indices
into sample_table.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def random_split(sample_table: pd.DataFrame, seed: int = 42,
                 val_ratio: float = 0.1, test_ratio: float = 0.1):
    n = len(sample_table)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx

def drug_kfold(sample_table: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Yields (train_idx, val_idx, test_idx) for each fold.
    test drugs are the held-out fold; val drugs are a random 10% of train drugs.
    """
    drugs = np.array(sorted(sample_table["drug_name_lower"].unique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_d_idx, te_d_idx) in enumerate(kf.split(drugs)):
        test_drugs  = set(drugs[te_d_idx])
        train_drugs = set(drugs[tr_d_idx])

        rng = np.random.default_rng(seed + fold)
        tr_drug_arr = np.array(sorted(train_drugs))
        n_val_d = max(1, int(len(tr_drug_arr) * 0.1))
        val_drug_set = set(tr_drug_arr[rng.choice(len(tr_drug_arr), n_val_d, replace=False)])
        train_drugs = train_drugs - val_drug_set

        train_idx = sample_table.index[sample_table["drug_name_lower"].isin(train_drugs)].values
        val_idx   = sample_table.index[sample_table["drug_name_lower"].isin(val_drug_set)].values
        test_idx  = sample_table.index[sample_table["drug_name_lower"].isin(test_drugs)].values
        yield fold, train_idx, val_idx, test_idx

def cell_split(sample_table: pd.DataFrame, seed: int = 42,
               val_ratio: float = 0.1, test_ratio: float = 0.2):
    cells = np.array(sorted(sample_table["cell_line_id"].unique()))
    rng = np.random.default_rng(seed)
    cells = rng.permutation(cells)
    n_test = int(len(cells) * test_ratio)
    n_val  = int(len(cells) * val_ratio)
    test_cells  = set(cells[:n_test])
    val_cells   = set(cells[n_test:n_test + n_val])
    train_cells = set(cells[n_test + n_val:])

    train_idx = sample_table.index[sample_table["cell_line_id"].isin(train_cells)].values
    val_idx   = sample_table.index[sample_table["cell_line_id"].isin(val_cells)].values
    test_idx  = sample_table.index[sample_table["cell_line_id"].isin(test_cells)].values
    return train_idx, val_idx, test_idx

def loto_split(sample_table: pd.DataFrame, target_drug: str):
    """Leave-one-target-out: one drug is held out as test."""
    test_idx  = sample_table.index[sample_table["drug_name_lower"] == target_drug.lower()].values
    train_idx = sample_table.index[sample_table["drug_name_lower"] != target_drug.lower()].values

    rng = np.random.default_rng(42)
    n_val = max(1, int(len(train_idx) * 0.05))
    val_mask = rng.choice(len(train_idx), n_val, replace=False)
    val_idx   = train_idx[val_mask]
    train_idx = np.delete(train_idx, val_mask)
    return train_idx, val_idx, test_idx
