"""Auxiliary drug--target input, for the baseline-fairness control.

HDCA-Net receives a curated drug--gene mask and the three deep-learning baselines receive
a fingerprint only. To ask whether the accuracy gap is that difference rather than the
architecture, each baseline can be given the same annotation in one of two forms:

  gated   ``x * M_dg[i]`` -- the expression gated by the drug's targets, which is the
          signal HDCA-Net's gene branch actually forms (Eq. gene_attn). Under a hard mask
          with a single annotated target the two are identical, and 195 of the 421
          annotated compounds have exactly one, so this is the strong form of the control.
  concat  ``M_dg[i]`` alone -- target identity without the expression it gates. The weak
          form, kept only as a contrast.

The mask is dropped entry-wise during training with the same probability HDCA-Net uses,
so the control is not handed a cleaner mask than the model it is compared against.
"""
import torch
import torch.nn as nn

MODES = ("none", "gated", "concat")


class AuxPriorEncoder(nn.Module):
    def __init__(self, num_genes: int, latent: int, mode: str = "none",
                 mask_dropout: float = 0.1, dropout: float = 0.2):
        super().__init__()
        assert mode in MODES, f"aux mode must be one of {MODES}"
        self.mode = mode
        self.mask_dropout = mask_dropout
        self.latent = latent
        self.enc = None if mode == "none" else nn.Sequential(
            nn.Linear(num_genes, latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def out_dim(self) -> int:
        return 0 if self.mode == "none" else self.latent

    def forward(self, cell_expr: torch.Tensor, drug_target: torch.Tensor):
        if self.mode == "none" or drug_target is None:
            return None
        m = drug_target
        if self.training and self.mask_dropout > 0:
            m = m * torch.bernoulli(torch.full_like(m, 1.0 - self.mask_dropout))
        return self.enc(cell_expr * m if self.mode == "gated" else m)
