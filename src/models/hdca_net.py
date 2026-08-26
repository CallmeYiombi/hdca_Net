"""
HDCA-Net: Hierarchical Drug-Cell Cross-Alignment Network.

Most drug-response models encode the drug and the cell line separately and leave
their interaction to the predictor MLP. HDCA-Net instead aligns drug-target signal
with cell state explicitly, at two biological scales:

  (1) Gene-level alignment
        gene_attn    = TargetGate(z_d, hcdt_drug_gene)         (B, G)
        gene_aligned = gene_attn * cell_expr                   (B, G)
        p_gene_align = PathwayBottleneck(gene_aligned)         (B, P)

      How strongly are the drug's target genes expressed in this cell line,
      aggregated per pathway?

  (2) Pathway-level alignment
        path_attn    = DirectPathwayGate(z_d, hcdt_drug_path)  (B, P)
        cell_path    = CellEncoder(cell_expr)                  (B, P)
        p_path_align = path_attn * cell_path                   (B, P)

      How active are the drug's annotated pathways in this cell line?

A drug-conditional 2-way attention fuses the branches into p_align (B, P), and the
prediction head consumes cat([p_align, cell_path, p_align * cell_path,
|p_align - cell_path|]).

Cell expression enters both branches, so the representation is anchored to the cell
line rather than to drug identity, and gene_attn, path_attn and the fusion weights
are each inspectable for the interpretability analyses.
"""
import torch
import torch.nn as nn

from .drug_encoder import DrugEncoder
from .target_gate import HCDTTargetGate
from .direct_path_gate import DirectPathwayGate
from .pathway_bottleneck import PathwayBottleneck
from .cell_encoder import CellPathwayEncoder

class HDCANet(nn.Module):
    def __init__(self,
                 fp_dim: int,
                 num_genes: int,
                 num_pathways: int,
                 gene_pathway_matrix: torch.Tensor,
                 drug_enc_hidden: int = 512,
                 drug_enc_out: int = 256,
                 gate_mode: str = "hard",
                 gate_gamma: float = 2.0,
                 path_gate_mode: str = "soft",
                 fp_input_dropout: float = 0.0,
                 use_mutation: bool = False,
                 dropout: float = 0.3,
                 lambda_neg: float = 0.1,
                 mask_dropout: float = 0.0,
                 align_branches: tuple = ("gene", "pathway"),
                 norm_type: str = "layer",
                 gene_norm_type: str = None,
                 path_norm_type: str = None,
                 cell_norm_type: str = None):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.mask_dropout = mask_dropout
        self.align_branches = tuple(align_branches)
        assert len(self.align_branches) >= 1
        for b in self.align_branches:
            assert b in ("gene", "pathway")

        self.gene_norm_type = gene_norm_type or norm_type
        self.path_norm_type = path_norm_type or norm_type
        self.cell_norm_type = cell_norm_type or norm_type
        for nt in (norm_type, self.gene_norm_type,
                   self.path_norm_type, self.cell_norm_type):
            assert nt in ("layer", "batch"), f"norm_type={nt}"
        self.norm_type = norm_type

        def _make_norm(dim, kind=None):
            kind = kind or norm_type
            return nn.BatchNorm1d(dim) if kind == "batch" else nn.LayerNorm(dim)
        self._make_norm = _make_norm

        self.drug_encoder = DrugEncoder(
            fp_dim=fp_dim,
            hidden_dim=drug_enc_hidden,
            out_dim=drug_enc_out,
            dropout=dropout,
            fp_input_dropout=fp_input_dropout,
        )

        if "gene" in self.align_branches:
            self.gene_embedding = nn.Parameter(
                torch.randn(num_genes, drug_enc_out) * 0.01
            )
            self.gene_gate = HCDTTargetGate(
                drug_dim=drug_enc_out,
                gene_dim=drug_enc_out,
                mode=gate_mode,
                gamma=gate_gamma,
            )
            self.pathway_bottleneck = PathwayBottleneck(
                gene_pathway_matrix=gene_pathway_matrix,
                normalize=True,
            )
            self.norm_p_gene_align = self._make_norm(num_pathways, self.gene_norm_type)

        if "pathway" in self.align_branches:
            self.path_gate = DirectPathwayGate(
                drug_dim=drug_enc_out,
                num_pathways=num_pathways,
                gamma=gate_gamma,
                mode=path_gate_mode,
            )
            self.norm_p_path_align = self._make_norm(num_pathways, self.path_norm_type)

        self.cell_encoder = CellPathwayEncoder(
            gene_pathway_matrix=gene_pathway_matrix,
            use_mutation=use_mutation,
            dropout=dropout,
        )
        self.norm_cell = self._make_norm(num_pathways, self.cell_norm_type)

        if len(self.align_branches) > 1:
            self.align_weight_net = nn.Sequential(
                nn.Linear(drug_enc_out, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(self.align_branches)),
            )

        fusion_dim = num_pathways * 4
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _mask_drop(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_dropout <= 0:
            return mask
        keep = torch.bernoulli(torch.full_like(mask, 1.0 - self.mask_dropout))
        return mask * keep

    def _compute_alignments(self, z_d, cell_expr, cell_mut,
                            hcdt_drug_gene, hcdt_drug_path):
        """Returns dict with branch vectors + intermediate attentions."""
        out = {}
        cell_path_raw = self.cell_encoder(cell_expr, cell_mut)
        out["cell_path_raw"] = cell_path_raw

        if "gene" in self.align_branches:
            gene_attn    = self.gene_gate(z_d, self.gene_embedding, hcdt_drug_gene)
            gene_aligned = gene_attn * cell_expr
            p_gene_align = self.norm_p_gene_align(
                self.pathway_bottleneck(gene_aligned)
            )
            has_target = (hcdt_drug_gene.sum(dim=1) > 0).float().unsqueeze(1)
            p_gene_align = p_gene_align * has_target
            out["gene_attn"]    = gene_attn
            out["gene_aligned"] = gene_aligned
            out["p_gene_align"] = p_gene_align

        if "pathway" in self.align_branches:
            path_attn    = self.path_gate(z_d, hcdt_drug_path)
            path_aligned = path_attn * cell_path_raw
            p_path_align = self.norm_p_path_align(path_aligned)
            has_path = (hcdt_drug_path.sum(dim=1) > 0).float().unsqueeze(1)
            p_path_align = p_path_align * has_path
            out["path_attn"]    = path_attn
            out["path_aligned"] = path_aligned
            out["p_path_align"] = p_path_align

        return out

    def forward(self,
                drug_fp: torch.Tensor,
                cell_expr: torch.Tensor,
                hcdt_drug_gene: torch.Tensor,
                hcdt_drug_path: torch.Tensor,
                hcdt_neg_gene: torch.Tensor = None,
                cell_mut: torch.Tensor = None):

        if self.mask_dropout > 0 and self.training:
            hcdt_drug_gene = self._mask_drop(hcdt_drug_gene)
            hcdt_drug_path = self._mask_drop(hcdt_drug_path)

        z_d = self.drug_encoder(drug_fp)

        align = self._compute_alignments(
            z_d, cell_expr, cell_mut, hcdt_drug_gene, hcdt_drug_path
        )

        branch_vecs = []
        if "gene" in self.align_branches:
            branch_vecs.append(align["p_gene_align"])
        if "pathway" in self.align_branches:
            branch_vecs.append(align["p_path_align"])

        if len(branch_vecs) == 1:
            p_align = branch_vecs[0]
            align_weights = None
        else:
            align_logits = self.align_weight_net(z_d)
            align_weights = torch.softmax(align_logits, dim=-1)
            p_align = sum(
                align_weights[:, i:i+1] * v for i, v in enumerate(branch_vecs)
            )

        p_cell = self.norm_cell(align["cell_path_raw"])
        prod   = p_align * p_cell
        diff   = torch.abs(p_align - p_cell)
        fusion = torch.cat([p_align, p_cell, prod, diff], dim=-1)
        y_pred = self.predictor(fusion)

        neg_loss = torch.tensor(0.0, device=drug_fp.device)
        if ("gene" in self.align_branches
                and hcdt_neg_gene is not None and self.training):
            gene_attn = align["gene_attn"]
            neg_sum = hcdt_neg_gene.sum(dim=1)
            has_pos = hcdt_drug_gene.sum(dim=1) > 0
            valid = (neg_sum > 0) & has_pos
            if valid.any():
                attn_on_neg = (gene_attn[valid] * hcdt_neg_gene[valid]).sum(dim=1)                              / neg_sum[valid]
                neg_loss = attn_on_neg.mean()

        gene_entropy = torch.tensor(0.0, device=drug_fp.device)
        if "gene" in self.align_branches and self.training:
            ga = align["gene_attn"]
            n_tgt = (hcdt_drug_gene > 0).sum(dim=1)
            valid = n_tgt > 1
            if valid.any():
                p = ga[valid].clamp_min(1e-12)
                ent = -(p * p.log()).sum(dim=1)
                denom = n_tgt[valid].float().log().clamp_min(1e-6)
                gene_entropy = (ent / denom).mean()

        return y_pred, neg_loss, align_weights, gene_entropy

    @torch.no_grad()
    def get_alignment_scores(self, drug_fp, cell_expr,
                             hcdt_drug_gene, hcdt_drug_path, cell_mut=None):
        """Interpretability hook. No mask dropout, no negative loss."""
        z_d = self.drug_encoder(drug_fp)
        align = self._compute_alignments(
            z_d, cell_expr, cell_mut, hcdt_drug_gene, hcdt_drug_path
        )

        branch_vecs = []
        names = []
        if "gene" in self.align_branches:
            branch_vecs.append(align["p_gene_align"])
            names.append("gene")
        if "pathway" in self.align_branches:
            branch_vecs.append(align["p_path_align"])
            names.append("pathway")

        if len(branch_vecs) == 1:
            p_align = branch_vecs[0]
            align_weights = None
        else:
            align_weights = torch.softmax(self.align_weight_net(z_d), dim=-1)
            p_align = sum(
                align_weights[:, i:i+1] * v for i, v in enumerate(branch_vecs)
            )

        result = dict(align)
        result["p_align"] = p_align
        result["p_cell"]  = self.norm_cell(align["cell_path_raw"])
        result["align_weights"] = align_weights
        result["branch_names"]  = names
        return result
