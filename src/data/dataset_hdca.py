import numpy as np
import torch
from torch.utils.data import Dataset

class HDCADataset(Dataset):
    def __init__(self,
                 sample_indices: np.ndarray,
                 sample_table: "pd.DataFrame",
                 drug_fp: np.ndarray,
                 cell_expr: np.ndarray,
                 hcdt_drug_gene: np.ndarray,
                 hcdt_drug_path: np.ndarray,
                 hcdt_neg_drug_gene: np.ndarray,
                 cell_mut: np.ndarray = None,
                 y_mean: float = 0.0,
                 y_std: float = 1.0):
        tbl = sample_table.iloc[sample_indices].reset_index(drop=True)
        self.drug_idx = tbl["drug_idx"].values.astype(np.int64)
        self.cell_idx = tbl["cell_idx"].values.astype(np.int64)
        raw_y = tbl["ln_ic50"].values.astype(np.float64)

        raw_y = np.clip(raw_y, -1e6, 1e6).astype(np.float32)
        self.y = (raw_y - y_mean) / y_std

        self.drug_fp             = drug_fp
        self.cell_expr           = cell_expr
        self.hcdt_drug_gene      = hcdt_drug_gene
        self.hcdt_drug_path      = hcdt_drug_path
        self.hcdt_neg_drug_gene  = hcdt_neg_drug_gene
        self.cell_mut            = cell_mut

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        di = self.drug_idx[idx]
        ci = self.cell_idx[idx]

        drug_fp    = torch.from_numpy(self.drug_fp[di])
        cell_expr  = torch.from_numpy(self.cell_expr[ci])
        dg_mask    = torch.from_numpy(self.hcdt_drug_gene[di])
        dp_mask    = torch.from_numpy(self.hcdt_drug_path[di])
        neg_mask   = torch.from_numpy(self.hcdt_neg_drug_gene[di])
        y          = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.cell_mut is not None:
            cell_mut = torch.from_numpy(self.cell_mut[ci])
            return drug_fp, cell_expr, dg_mask, dp_mask, neg_mask, cell_mut, y

        return drug_fp, cell_expr, dg_mask, dp_mask, neg_mask, y
