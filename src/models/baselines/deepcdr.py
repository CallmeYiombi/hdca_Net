"""
DeepCDR baseline (Liu et al., Nucleic Acids Research 2020).

Original uses GCN drug encoder + three omics sub-networks (expression,
methylation, mutation) with an attention-weighted fusion.
Here we use FP drug encoder + two omics (expression + mutation), keeping
the multi-stream attention fusion which is the core architectural contribution.

Drug  : FP(2048)     -> FC(512) -> ReLU -> z_d (512)
Cell  : expr(G)      -> FC(1024) -> ReLU -> FC(256) -> v_expr
        mut(G)       -> FC(256)  -> ReLU -> FC(256) -> v_mut
Attn  : scores = softmax(W_a . tanh([v_expr; v_mut]))  (2-way softmax)
        z_c = attn[0]*v_expr + attn[1]*v_mut
Fusion: cat(z_d, z_c)(768) -> FC(512) -> ReLU -> Dropout -> FC(256) -> ReLU -> FC(1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCDR(nn.Module):
    def __init__(self,
                 fp_dim: int = 2048,
                 num_genes: int = 19215,
                 drug_latent: int = 512,
                 omics_latent: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.drug_enc = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, drug_latent),
            nn.ReLU(),
        )
        self.expr_enc = nn.Sequential(
            nn.Linear(num_genes, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, omics_latent),
            nn.ReLU(),
        )
        self.mut_enc = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, omics_latent),
            nn.ReLU(),
        )

        self.attn = nn.Linear(omics_latent, 1)

        self.predictor = nn.Sequential(
            nn.Linear(drug_latent + omics_latent, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, drug_fp: torch.Tensor,
                cell_expr: torch.Tensor,
                cell_mut: torch.Tensor) -> torch.Tensor:
        z_d    = self.drug_enc(drug_fp)
        v_expr = self.expr_enc(cell_expr)
        v_mut  = self.mut_enc(cell_mut)

        v_stack  = torch.stack([v_expr, v_mut], dim=1)
        scores   = self.attn(v_stack).squeeze(-1)
        weights  = F.softmax(scores, dim=-1).unsqueeze(-1)
        z_c      = (weights * v_stack).sum(dim=1)

        return self.predictor(torch.cat([z_d, z_c], dim=-1))
