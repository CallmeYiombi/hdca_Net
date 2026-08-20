import numpy as np
import torch
from torch.utils.data import Dataset

class PancdrDataset(Dataset):
    """Yields (drug_feat, drug_adj, gexpr, y) for PANCDR.

    drug_atom_feats : (D, N, AF)   atom features per drug (indexed by drug_idx)
    drug_adj_norm   : (D, N, N)    normalised adjacency per drug
    cell_expr       : (C, G)       (already gene-standardised upstream)
    """

    def __init__(self, sample_indices, sample_table,
                 drug_atom_feats, drug_adj_norm, cell_expr,
                 y_mean=0.0, y_std=1.0):
        tbl = sample_table.iloc[sample_indices].reset_index(drop=True)
        self.drug_idx = tbl["drug_idx"].values.astype(np.int64)
        self.cell_idx = tbl["cell_idx"].values.astype(np.int64)
        raw_y = np.clip(tbl["ln_ic50"].values.astype(np.float64), -1e6, 1e6).astype(np.float32)
        self.y = (raw_y - y_mean) / y_std

        self.drug_feat = drug_atom_feats
        self.drug_adj = drug_adj_norm
        self.cell_expr = cell_expr

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        di, ci = self.drug_idx[idx], self.cell_idx[idx]
        return (
            torch.from_numpy(self.drug_feat[di]),
            torch.from_numpy(self.drug_adj[di]),
            torch.from_numpy(self.cell_expr[ci]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

class ExprOnlyDataset(Dataset):
    """Unlabeled target-domain expression (for the adversarial discriminator)."""

    def __init__(self, cell_expr):
        self.cell_expr = cell_expr

    def __len__(self):
        return len(self.cell_expr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.cell_expr[idx])
