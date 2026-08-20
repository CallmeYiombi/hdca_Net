import torch
import torch.nn as nn

class DirectPathwayGate(nn.Module):
    """
    Path 3: Drug -> Pathway attention using HCDT drug-pathway direct annotations.

    Biologically: captures database-curated drug-pathway associations (Reactome,
    KEGG, SMPDB) without going through gene-level intermediaries.

      z_d -> query over pathway embeddings + HCDT drug-pathway bias
          -> path_attn (B, P) = drug_p3
    """
    def __init__(self, drug_dim: int, num_pathways: int, gamma: float = 2.0,
                 mode: str = "soft"):
        super().__init__()
        assert mode in ("soft", "hard", "none")
        self.mode = mode
        self.gamma = gamma
        self.pathway_embedding = nn.Parameter(
            torch.randn(num_pathways, drug_dim) * 0.01
        )
        self.query = nn.Linear(drug_dim, drug_dim, bias=False)
        self.scale = drug_dim ** 0.5

    def forward(self, z_d: torch.Tensor,
                hcdt_drug_path: torch.Tensor) -> torch.Tensor:
        """
        z_d            : (B, drug_dim)
        hcdt_drug_path : (B, P) float binary -- 1 if drug-pathway in HCDT
        returns        : (B, P) drug pathway representation via direct path
        """
        q = self.query(z_d)
        score = q @ self.pathway_embedding.T / self.scale

        if self.mode == "hard":

            score = score.masked_fill(hcdt_drug_path <= 0, -1e9)
        elif self.mode == "soft":
            score = score + self.gamma * hcdt_drug_path

        drug_p3 = torch.softmax(score, dim=-1)
        return drug_p3
