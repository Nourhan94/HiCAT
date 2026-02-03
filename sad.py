from __future__ import annotations
import torch
import torch.nn as nn
from .temporal import TemporalEncoder
from .structural import StructuralEncoder
from .decoder import SharedDecoder
from .discriminator import LatentDiscriminator

class SaDTS(nn.Module):
    """
    SaD-style TS imputation:
      - Temporal encoder -> shared decoder -> Xhat_T
      - Structural encoder -> shared decoder -> Xhat_S
      - AAE discriminator regularization on latent space
    """
    def __init__(
        self,
        D: int,
        hidden: int,
        latent: int,
        temporal_cfg: dict,
        structural_cfg: dict,
        decoder_cfg: dict,
        adv_cfg: dict,
    ):
        super().__init__()
        self.temporal = TemporalEncoder(
            in_dim=D,
            cnn_channels=temporal_cfg["cnn_channels"],
            cnn_kernel=temporal_cfg["cnn_kernel"],
            lstm_hidden=hidden,
            latent=latent,
            lstm_layers=temporal_cfg.get("lstm_layers", 1),
        )
        self.structural = StructuralEncoder(
            in_dim=D,
            hidden=hidden,
            latent=latent,
            gcn_layers=structural_cfg.get("gcn_layers", 2),
        )
        self.decoder = SharedDecoder(
            latent=latent,
            out_dim=D,
            hidden=hidden,
            mlp_layers=decoder_cfg.get("mlp_layers", 2),
        )

        self.use_adv = adv_cfg.get("use_adv", True)
        self.disc = LatentDiscriminator(latent=latent, hidden=adv_cfg.get("disc_hidden", 64)) if self.use_adv else None
        self.latent = latent

    def forward(self, X_in: torch.Tensor, A: torch.Tensor):
        """
        X_in: (N,T,D)
        A:    (N,N)
        """
        zT = self.temporal(X_in)            # (N,T,latent)
        zS = self.structural(X_in, A)       # (N,T,latent)
        Xhat_T = self.decoder(zT)           # (N,T,D)
        Xhat_S = self.decoder(zS)           # (N,T,D)
        return {"zT": zT, "zS": zS, "Xhat_T": Xhat_T, "Xhat_S": Xhat_S}
