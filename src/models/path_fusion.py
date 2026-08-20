import torch
import torch.nn as nn

class MultiPathFusion(nn.Module):
    """
    Drug-conditional weighted fusion of 3 HCDT biological paths.

    The fusion weights are conditioned on z_d so each drug can rely on
    different paths depending on which HCDT evidence is available.

    drug_p1 : (B, P) via protein target gate
    drug_p2 : (B, P) via RNA-mediated gate
    drug_p3 : (B, P) via direct pathway gate

    weights = softmax(MLP(z_d))  -> (B, 3)
    drug_pathway = sum_i weights[:,i] * drug_pi
    """
    def __init__(self, drug_dim: int, num_pathways: int, dropout: float = 0.1):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(drug_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, z_d: torch.Tensor,
                drug_p1: torch.Tensor,
                drug_p2: torch.Tensor,
                drug_p3: torch.Tensor) -> tuple:
        """
        Returns (drug_pathway, path_weights) for interpretability.
        path_weights : (B, 3) -- how much each path contributes per sample
        """
        logits = self.weight_net(z_d)
        weights = torch.softmax(logits, dim=-1)

        drug_pathway = (
            weights[:, 0:1] * drug_p1 +
            weights[:, 1:2] * drug_p2 +
            weights[:, 2:3] * drug_p3
        )
        return drug_pathway, weights
