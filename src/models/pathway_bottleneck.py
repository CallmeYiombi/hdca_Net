import torch
import torch.nn as nn

class PathwayBottleneck(nn.Module):
    """
    Aggregates gene-level scores into pathway-level scores using a fixed
    gene-pathway membership matrix (non-trainable, structural constraint).
    Prevents individual gene/drug identity memorization.
    """
    def __init__(self, gene_pathway_matrix: torch.Tensor, normalize: bool = True):
        super().__init__()
        A = gene_pathway_matrix.float()
        if normalize:
            pathway_size = A.sum(dim=0).clamp(min=1.0)
            A = A / torch.sqrt(pathway_size)
        self.register_buffer("A", A)

    def forward(self, gene_score: torch.Tensor) -> torch.Tensor:
        """
        gene_score : (B, G)
        returns    : (B, P)
        """
        return gene_score @ self.A
