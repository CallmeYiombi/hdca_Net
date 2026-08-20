"""
PANCDR baseline (Briefings in Bioinformatics 2024, DMCB-GIST).

Faithful port of the original architecture (src/ModelTraining/model.py of the
official repo) adapted to our HDCA-Net harness:

  * Drug encoder : molecular-graph GraphConv stack (DeepCDR-style), fed with our
                   own rdkit graphs (drug_atom_feats.npy / drug_adj_norm.npy).
  * Expr encoder : VAE-style MLP encoder producing a latent z (KL is *not* used
                   in the loss, matching the original -- reparametrise acts as a
                   stochastic bottleneck).
  * Predictor    : concat(drug, z) -> Linear(300) -> Conv2d stack -> scalar.
  * Discriminator: aligns source (GDSC) vs target (CCLE/gCSI) expression latents
                   via the adversarial scheme in train_pancdr.py.

Regression (ln IC50) is the default here (is_regr=True in the original).
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
from typing import Optional

class GraphConv(nn.Module):
    """Original PANCDR GraphConv: X' = A^T (X W + b)."""

    def __init__(self, in_channels, units, step_num=1):
        super().__init__()
        self.weight = Parameter(torch.empty((in_channels, units)))
        self.bias = Parameter(torch.empty(units))
        self.step_num = step_num
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(torch.matmul(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return torch.gt(deeper, 0.0)

    def forward(self, features: Tensor, edges: Tensor) -> Tensor:
        outputs = torch.matmul(features, self.weight) + self.bias
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = torch.matmul(edges.permute(0, 2, 1), outputs)
        return outputs.permute(0, 2, 1)

class ExprEncoder(nn.Module):
    """VAE-style expression encoder. forward returns (z, mu, logvar)."""

    def __init__(self, n_input, nz, n_hidden=256, n_layers=3):
        super().__init__()
        self.nz = nz
        enc = [nn.Linear(n_input, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU()]
        for _ in range(n_layers):
            enc += [nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU()]
        enc += [nn.Linear(n_hidden, n_hidden)]
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(n_hidden, nz)
        self.fc_logvar = nn.Linear(n_hidden, nz)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar) if sample else mu
        return z, mu, logvar

class DrugCDRNet(nn.Module):
    """Drug-graph GCN + CNN predictor. Takes an already-encoded expr latent."""

    def __init__(self, in_channels, units_list=(256, 256, 256),
                 d_dim=100, nz=256, use_dropout=False, is_regr=True):
        super().__init__()
        self.GC1 = GraphConv(in_channels, units_list[0])
        self.GC2 = GraphConv(units_list[0], units_list[1])
        self.GC3 = GraphConv(units_list[1], units_list[2])
        self.GC4 = GraphConv(units_list[2], d_dim)

        def brd(dim):
            layers = [nn.BatchNorm1d(dim), nn.ReLU()]
            if use_dropout:
                layers += [nn.Dropout(0.1)]
            return nn.Sequential(*layers)

        self.BRD1, self.BRD2 = brd(units_list[0]), brd(units_list[1])
        self.BRD3, self.BRD4 = brd(units_list[2]), brd(d_dim)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(nz + d_dim, 300), nn.Dropout(0.1))
        self.CONV = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(150, 1), stride=(1, 1)), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(30, 10, kernel_size=(5, 1), stride=(1, 1)), nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 5, kernel_size=(5, 1), stride=(1, 1)), nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(30, 1) if is_regr else nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, drug_feat, drug_adj, z_gexpr):
        h = self.BRD1(self.GC1(drug_feat, drug_adj)).permute(0, 2, 1)
        h = self.BRD2(self.GC2(h, drug_adj)).permute(0, 2, 1)
        h = self.BRD3(self.GC3(h, drug_adj)).permute(0, 2, 1)
        h = self.BRD4(self.GC4(h, drug_adj))
        x_drug = self.Pool(h).view(h.shape[0], -1)
        x = torch.cat((x_drug, z_gexpr), dim=1)
        x = self.Linear(x).view(-1, 1, 300, 1)
        x = self.CONV(x).view(x.shape[0], -1)
        return self.fc(x)

class Discriminator(nn.Module):
    """Domain discriminator on the expression latent (source vs target)."""

    def __init__(self, nz):
        super().__init__()
        self.adv = nn.Sequential(
            nn.Linear(nz, nz // 2), nn.ReLU(),
            nn.Linear(nz // 2, nz // 4), nn.ReLU(),
            nn.Linear(nz // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.adv(x)

class PANCDR(nn.Module):
    """Convenience wrapper bundling the three sub-models."""

    def __init__(self, n_genes, atom_feat_dim, nz=256, d_dim=100,
                 units_list=(256, 256, 256), is_regr=True):
        super().__init__()
        self.encoder = ExprEncoder(n_genes, nz)
        self.cdr = DrugCDRNet(atom_feat_dim, units_list=units_list,
                              d_dim=d_dim, nz=nz, is_regr=is_regr)
        self.disc = Discriminator(nz)

    def forward(self, drug_feat, drug_adj, gexpr, sample=True):
        z, mu, logvar = self.encoder(gexpr, sample=sample)
        y = self.cdr(drug_feat, drug_adj, z)
        return y, z
