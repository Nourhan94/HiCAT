from __future__ import annotations
import torch
import torch.nn as nn

class SharedDecoder(nn.Module):
    """
    Shared decoder used for both temporal and structural latents.
    Input:  z (N,T,latent)
    Output: X_hat (N,T,D)
    """
    def __init__(self, latent: int, out_dim: int, hidden: int, mlp_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = latent
        for _ in range(max(0, mlp_layers - 1)):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        N, T, F = z.shape
        y = self.mlp(z.reshape(N*T, F)).reshape(N, T, -1)
        return y
