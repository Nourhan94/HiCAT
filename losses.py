from __future__ import annotations
import torch
import torch.nn.functional as F

def recon_loss_l1(X: torch.Tensor, Xhat: torch.Tensor, M_obs: torch.Tensor) -> torch.Tensor:
    """
    L1 reconstruction on OBSERVED entries (mask=1 observed).
    """
    # avoid divide-by-zero
    denom = M_obs.sum().clamp_min(1.0)
    return (torch.abs(X - Xhat) * M_obs).sum() / denom

def sample_prior(latent: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.randn(n, latent, device=device)

def adv_discriminator_loss(D, z_real: torch.Tensor, z_fake: torch.Tensor) -> torch.Tensor:
    """
    Standard GAN BCE loss for discriminator.
    """
    pr = D(z_real)
    pf = D(z_fake.detach())
    loss = F.binary_cross_entropy(pr, torch.ones_like(pr)) + F.binary_cross_entropy(pf, torch.zeros_like(pf))
    return loss

def adv_encoder_loss(D, z_fake: torch.Tensor) -> torch.Tensor:
    """
    Encoder tries to fool discriminator.
    """
    pf = D(z_fake)
    loss = F.binary_cross_entropy(pf, torch.ones_like(pf))
    return loss
