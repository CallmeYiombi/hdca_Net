import torch
import torch.nn as nn

class DrugEncoder(nn.Module):
    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 256,
                 dropout: float = 0.3, fp_input_dropout: float = 0.0):
        super().__init__()
        self.fp_dropout = nn.Dropout(fp_input_dropout) if fp_input_dropout > 0 else None
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, drug_fp: torch.Tensor) -> torch.Tensor:
        if self.fp_dropout is not None and self.training:
            drug_fp = self.fp_dropout(drug_fp)
        return self.net(drug_fp)
