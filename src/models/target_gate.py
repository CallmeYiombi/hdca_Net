import torch
import torch.nn as nn

class HCDTTargetGate(nn.Module):
    """
    Projects drug embedding onto HCDT target gene space.
    Returns per-gene attention scores (soft probability over genes).

    mode='soft': score += gamma * hcdt_mask  (non-HCDT genes still attend but weakly)
    mode='hard': HCDT=0 genes masked to -inf
    """
    def __init__(self, drug_dim: int = 256, gene_dim: int = 256,
                 mode: str = "soft", gamma: float = 2.0):
        super().__init__()
        assert mode in ("soft", "hard", "none")
        self.mode = mode
        self.gamma = gamma
        self.query = nn.Linear(drug_dim, gene_dim, bias=False)
        self.scale = gene_dim ** 0.5

    def forward(self, drug_embedding: torch.Tensor,
                gene_embedding: torch.Tensor,
                hcdt_mask: torch.Tensor) -> torch.Tensor:
        """
        drug_embedding : (B, drug_dim)
        gene_embedding : (G, gene_dim)
        hcdt_mask      : (B, G) float, 1 if HCDT relation exists else 0
        returns        : (B, G) attention weights summing to 1
        """
        q = self.query(drug_embedding)
        score = q @ gene_embedding.T / self.scale

        if self.mode == "hard":
            score = score.masked_fill(hcdt_mask <= 0, -1e9)
        elif self.mode == "soft":
            score = score + self.gamma * hcdt_mask

        return torch.softmax(score, dim=-1)
