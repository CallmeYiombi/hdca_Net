"""
GraphDRP baseline (Nguyen et al., Bioinformatics 2021).

Original uses GCN on molecular graph + cell expression FC.
Here both drug and cell use FP/expression inputs with FC encoders of
comparable capacity — drug representation is equalized across all baselines.

Drug  : FP(2048) → FC(1024) → BN → ReLU → Dropout → FC(256) → BN → ReLU
Cell  : expr(G)  → FC(1024) → BN → ReLU → Dropout → FC(256) → BN → ReLU
Fusion: cat(512) → FC(256) → ReLU → Dropout → FC(128) → ReLU → FC(1)
"""
import torch
import torch.nn as nn

from .aux_prior import AuxPriorEncoder


def _fc_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class GraphDRP(nn.Module):
    def __init__(self,
                 fp_dim: int = 2048,
                 num_genes: int = 19215,
                 hidden: int = 1024,
                 latent: int = 256,
                 dropout: float = 0.2,
                 aux_mode: str = "none",
                 mask_dropout: float = 0.1):
        super().__init__()
        self.aux = AuxPriorEncoder(num_genes, latent, aux_mode, mask_dropout, dropout)
        self.drug_enc = nn.Sequential(
            _fc_block(fp_dim, hidden, dropout),
            _fc_block(hidden, latent, dropout),
        )
        self.cell_enc = nn.Sequential(
            _fc_block(num_genes, hidden, dropout),
            _fc_block(hidden, latent, dropout),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent * 2 + self.aux.out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, drug_fp: torch.Tensor, cell_expr: torch.Tensor,
                cell_mut: torch.Tensor = None,
                drug_target: torch.Tensor = None) -> torch.Tensor:
        z_d = self.drug_enc(drug_fp)
        z_c = self.cell_enc(cell_expr)
        parts = [z_d, z_c]
        z_a = self.aux(cell_expr, drug_target)
        if z_a is not None:
            parts.append(z_a)
        return self.predictor(torch.cat(parts, dim=-1))
