import torch
import torch.nn as nn

class CellPathwayEncoder(nn.Module):
    """
    Aggregates cell-line omics (expression, optionally mutation) into
    pathway-level vulnerability vectors.
    """
    def __init__(self, gene_pathway_matrix: torch.Tensor,
                 use_mutation: bool = False, dropout: float = 0.2):
        super().__init__()
        A = gene_pathway_matrix.float()
        pathway_size = A.sum(dim=0).clamp(min=1.0)
        A_norm = A / torch.sqrt(pathway_size)
        self.register_buffer("A", A_norm)

        self.use_mutation = use_mutation
        num_pathways = A.shape[1]
        in_dim = num_pathways * 2 if use_mutation else num_pathways

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, num_pathways),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_pathways, num_pathways),
        )

    def forward(self, cell_expr: torch.Tensor,
                cell_mut: torch.Tensor = None) -> torch.Tensor:
        """
        cell_expr : (B, G)
        cell_mut  : (B, G) optional
        returns   : (B, P)
        """
        expr_path = cell_expr @ self.A

        if self.use_mutation and cell_mut is not None:
            mut_path = cell_mut @ self.A
            combined = torch.cat([expr_path, mut_path], dim=-1)
        else:
            combined = expr_path

        return self.mlp(combined)
