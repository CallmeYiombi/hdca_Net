"""
TGSA baseline (Wang et al., Bioinformatics 2021).

Original uses drug SMILES + cell gene expression with Tumor-Specific Gene
Similarity (TSGS) and Association (TSGA) attention over gene pairs.

Here we capture the core idea of biology-guided gene grouping:
  - Gene expression is aggregated into pathway-level vectors via a
    fixed gene_pathway membership matrix (same matrix HCP-Net uses),
    then refined with a learned self-attention over pathways.
  - Drug encoder uses FP for fair comparison.

Drug   : FP(2048) → FC(512) → ReLU → Dropout → FC(256) → z_d
Cell   : expr(G) × A_norm(G×P) → p_raw(P)
         p_raw → pathway_self_attn → p_attn(P)  [TSGS-style grouping]
         p_attn → FC(P→512) → ReLU → Dropout → FC(512→256) → z_c
Fusion : cat(z_d, z_c)(512) → FC(256) → ReLU → Dropout → FC(128) → FC(1)

gene_pathway_matrix : (G, P)  float binary, row=gene col=pathway
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_prior import AuxPriorEncoder


class TGSA(nn.Module):
    def __init__(self,
                 fp_dim: int,
                 num_genes: int,
                 gene_pathway_matrix: torch.Tensor,
                 latent: int = 256,
                 attn_heads: int = 4,
                 dropout: float = 0.3,
                 aux_mode: str = "none",
                 mask_dropout: float = 0.1):
        super().__init__()
        self.aux = AuxPriorEncoder(num_genes, latent, aux_mode, mask_dropout, dropout)
        A = gene_pathway_matrix.float()                     # (G, P)
        col_sum = A.sum(dim=0).clamp(min=1.0)
        A_norm  = A / torch.sqrt(col_sum)                   # normalise by pathway size
        self.register_buffer("A_norm", A_norm)
        num_pathways = A.shape[1]

        self.drug_enc = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent),
            nn.ReLU(),
        )

        # Project pathways to attention-friendly dim, then self-attend (TSGS-style)
        attn_dim = (latent // attn_heads) * attn_heads   # e.g. 256, always divisible
        self.pathway_proj = nn.Linear(num_pathways, attn_dim)
        self.pathway_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cell_mlp = nn.Sequential(
            nn.Linear(attn_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent * 2 + self.aux.out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, drug_fp: torch.Tensor,
                cell_expr: torch.Tensor,
                cell_mut: torch.Tensor = None,
                drug_target: torch.Tensor = None) -> torch.Tensor:
        z_d    = self.drug_enc(drug_fp)                         # (B, L)
        p_raw  = cell_expr @ self.A_norm                        # (B, P)
        p_proj = self.pathway_proj(p_raw)                       # (B, attn_dim)

        # self-attention treats each sample's pathway vector as a single token
        p_in      = p_proj.unsqueeze(1)                         # (B, 1, attn_dim)
        p_attn, _ = self.pathway_attn(p_in, p_in, p_in)        # (B, 1, attn_dim)
        p_attn    = p_attn.squeeze(1)                           # (B, attn_dim)

        z_c    = self.cell_mlp(p_attn)                          # (B, L)
        parts = [z_d, z_c]
        z_a = self.aux(cell_expr, drug_target)
        if z_a is not None:
            parts.append(z_a)
        return self.predictor(torch.cat(parts, dim=-1))
