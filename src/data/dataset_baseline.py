import numpy as np
import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """Minimal dataset for baseline models: drug_fp, cell_expr, cell_mut, drug_target, y.

    ``drug_target`` is the drug--gene mask row used by the baseline-fairness control
    (src/models/baselines/aux_prior.py). When no matrix is supplied a length-1 zero
    tensor is returned so the batch shape stays fixed and the models ignore it.
    """

    def __init__(self,
                 sample_indices: np.ndarray,
                 sample_table,
                 drug_fp: np.ndarray,
                 cell_expr: np.ndarray,
                 cell_mut: np.ndarray,
                 y_mean: float = 0.0,
                 y_std: float = 1.0,
                 drug_target: np.ndarray = None):
        tbl = sample_table.iloc[sample_indices].reset_index(drop=True)
        self.drug_idx = tbl["drug_idx"].values.astype(np.int64)
        self.cell_idx = tbl["cell_idx"].values.astype(np.int64)
        raw_y = np.clip(tbl["ln_ic50"].values.astype(np.float64), -1e6, 1e6).astype(np.float32)
        self.y = (raw_y - y_mean) / y_std

        self.drug_fp     = drug_fp
        self.cell_expr   = cell_expr
        self.cell_mut    = cell_mut
        self.drug_target = drug_target
        self._empty = np.zeros(1, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        di = self.drug_idx[idx]
        ci = self.cell_idx[idx]
        tgt = self._empty if self.drug_target is None \
            else self.drug_target[di].astype(np.float32)
        return (
            torch.from_numpy(self.drug_fp[di]),
            torch.from_numpy(self.cell_expr[ci]),
            torch.from_numpy(self.cell_mut[ci]),
            torch.from_numpy(tgt),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )
