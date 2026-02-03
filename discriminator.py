from __future__ import annotations
import torch
import torch.nn as nn

class LatentDiscriminator(nn.Module):
    """
    Discriminator over latent vectors z (flattened over N,T).
    Outputs probability of 'real prior sample' vs 'encoder sample'.
    """
    def __init__(self, latent: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
