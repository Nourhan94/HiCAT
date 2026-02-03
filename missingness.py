from __future__ import annotations
import torch

def make_general_mask(X: torch.Tensor, p: float, seed: int = 0) -> torch.Tensor:
    """
    General missing: "missing points distributed at random way".
    Returns M in {0,1} with same shape as X (N,T,D).
    """
    assert 0.0 <= p < 1.0
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    # 1 means observed, 0 means missing
    M = (torch.rand_like(X, generator=g) > p).to(X.dtype)
    return M

def make_block_mask(
    X: torch.Tensor,
    random_drop_rate: float = 0.05,
    failure_prob: float = 0.0015,
    block_len_min: int = 12,
    block_len_max: int = 36,
    seed: int = 0,
) -> torch.Tensor:
    """
    Block missing (GRIN-style as described in SaD experiments):
      1) 5% random removal per sensor (configurable random_drop_rate)
      2) rare failure event with probability failure_prob
      3) if failure happens, remove a contiguous interval of length L in [min,max]

    Output mask M: 1 observed, 0 missing, shape (N,T,D)
    """
    assert X.dim() == 3
    N, T, D = X.shape
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)

    M = torch.ones_like(X)

    # Step 1: sparse random removal per sensor
    # remove K ≈ random_drop_rate * T positions (for all channels)
    K = max(0, int(random_drop_rate * T))
    if K > 0:
        for i in range(N):
            idx = torch.randperm(T, generator=g, device=X.device)[:K]
            M[i, idx, :] = 0.0

    # Step 2: failure event -> contiguous block removal
    for i in range(N):
        u = torch.rand((), generator=g, device=X.device).item()
        if u < failure_prob:
            L = int(torch.randint(low=block_len_min, high=block_len_max+1, size=(), generator=g, device=X.device).item())
            L = max(1, min(L, T))
            s = int(torch.randint(low=0, high=max(1, T - L + 1), size=(), generator=g, device=X.device).item())
            M[i, s:s+L, :] = 0.0

    return M
