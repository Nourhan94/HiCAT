import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayerDense(nn.Module):
    """
    Dense GAT over all node pairs.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,F]
        B, T, N, F_in = x.shape
        h = self.W(x)  # [B,T,N,Fout]

        # e_ij = LeakyReLU(a(h_i) + a(h_j))
        e_i = self.a_src(h)               # [B,T,N,1]
        e_j = self.a_dst(h)               # [B,T,N,1]
        e = e_i + e_j.transpose(2, 3)     # [B,T,N,N] via broadcast
        e = self.leaky(e)

        # attention over neighbors j (here: all nodes)
        alpha = F.softmax(e, dim=-1)      # [B,T,N,N]
        alpha = self.dropout(alpha)

        # aggregate: sum_j alpha_ij * h_j
        out = torch.einsum("btij,btjf->btif", alpha, h)  # [B,T,N,Fout]
        return out


class StructuralEncoderGAT(nn.Module):
    """
    Learns adjacency via attention.
    """
    def __init__(self, n_feats: int, hidden: int, latent: int, gcn_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = n_feats
        for k in range(gcn_layers - 1):
            layers.append(GATLayerDense(in_dim, hidden, dropout))
            in_dim = hidden
        layers.append(GATLayerDense(in_dim, latent, dropout))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
