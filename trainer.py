from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict
import torch
from torch.optim import Adam
from tqdm import tqdm

from .losses import recon_loss_l1, sample_prior, adv_discriminator_loss, adv_encoder_loss
from .metrics import mae_on_missing, mre_on_missing

@dataclass
class TrainState:
    best_val: float = math.inf
    bad_epochs: int = 0

class Trainer:
    def __init__(self, model, cfg: dict, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = Adam(self.model.parameters(), lr=cfg["train"]["lr"])
        self.optD = Adam(self.model.disc.parameters(), lr=cfg["train"]["lr"]) if model.use_adv else None
        self.lambda_adv = cfg["train"].get("lambda_adv", 0.1)
        self.grad_clip = cfg["train"].get("grad_clip", 1.0)
        self.patience = cfg["train"].get("early_stop_patience", 10)

    def step_batch(self, X: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> Dict[str, float]:
        """
        X: (N,T,D) full
        M: (N,T,D) mask (1 observed, 0 missing)
        X_in = M*X (missing filled with 0)
        """
        X = X.to(self.device)
        M = M.to(self.device)
        A = A.to(self.device)

        X_in = M * X

        out = self.model(X_in, A)
        Xhat_T = out["Xhat_T"]
        Xhat_S = out["Xhat_S"]

        # reconstruction losses on OBSERVED entries
        Lrec_T = recon_loss_l1(X, Xhat_T, M)
        Lrec_S = recon_loss_l1(X, Xhat_S, M)
        Lrec = Lrec_T + Lrec_S

        # adversarial (AAE-style) on latent space (flatten N,T)
        Ld = torch.tensor(0.0, device=self.device)
        Ladv = torch.tensor(0.0, device=self.device)

        if self.model.use_adv:
            zT = out["zT"].reshape(-1, self.model.latent)
            zS = out["zS"].reshape(-1, self.model.latent)
            z_fake = torch.cat([zT, zS], dim=0)
            z_real = sample_prior(self.model.latent, z_fake.shape[0], self.device)

            # 1) update discriminator
            self.optD.zero_grad(set_to_none=True)
            Ld = adv_discriminator_loss(self.model.disc, z_real, z_fake)
            Ld.backward()
            torch.nn.utils.clip_grad_norm_(self.model.disc.parameters(), self.grad_clip)
            self.optD.step()

            # 2) update encoders+decoder to fool discriminator
            Ladv = adv_encoder_loss(self.model.disc, z_fake)

        self.opt.zero_grad(set_to_none=True)
        L = Lrec + self.lambda_adv * Ladv
        L.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()

        # Evaluate (on missing) using the average prediction
        Xhat = 0.5 * (Xhat_T + Xhat_S)
        return {
            "loss": float(L.detach().cpu()),
            "lrec": float(Lrec.detach().cpu()),
            "ld": float(Ld.detach().cpu()),
            "ladv": float(Ladv.detach().cpu()),
            "mae_miss": mae_on_missing(X, Xhat, M),
            "mre_miss": mre_on_missing(X, Xhat, M),
        }

    @torch.no_grad()
    def eval_batch(self, X: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> Dict[str, float]:
        X = X.to(self.device)
        M = M.to(self.device)
        A = A.to(self.device)
        X_in = M * X
        out = self.model(X_in, A)
        Xhat = 0.5 * (out["Xhat_T"] + out["Xhat_S"])
        return {
            "mae_miss": mae_on_missing(X, Xhat, M),
            "mre_miss": mre_on_missing(X, Xhat, M),
        }

    def fit(self, train_loader, val_loader, A: torch.Tensor):
        state = TrainState()
        for epoch in range(1, self.cfg["train"]["epochs"] + 1):
            self.model.train()
            tr = {"loss": 0.0, "lrec": 0.0, "ld": 0.0, "ladv": 0.0, "mae_miss": 0.0, "mre_miss": 0.0}
            n = 0
            for X, M in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                # X: (B,N,T,D) -> we process per batch by merging B into time windows
                # Here we just loop over batch items to keep code simple and readable.
                for b in range(X.shape[0]):
                    stats = self.step_batch(X[b], A, M[b])
                    for k in tr:
                        tr[k] += stats[k]
                    n += 1
            for k in tr:
                tr[k] /= max(1, n)

            self.model.eval()
            vr = {"mae_miss": 0.0, "mre_miss": 0.0}
            vn = 0
            for X, M in val_loader:
                for b in range(X.shape[0]):
                    stats = self.eval_batch(X[b], A, M[b])
                    vr["mae_miss"] += stats["mae_miss"]
                    vr["mre_miss"] += stats["mre_miss"]
                    vn += 1
            vr["mae_miss"] /= max(1, vn)
            vr["mre_miss"] /= max(1, vn)

            # early stopping on val MAE missing
            val_key = vr["mae_miss"]
            if val_key < state.best_val - 1e-6:
                state.best_val = val_key
                state.bad_epochs = 0
                best = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                state.bad_epochs += 1

            print(f"[epoch {epoch}] train loss={tr['loss']:.4f} "
                  f"train MAE(miss)={tr['mae_miss']:.4f} val MAE(miss)={vr['mae_miss']:.4f} val MRE(miss)={vr['mre_miss']:.4f}")

            if state.bad_epochs >= self.patience:
                print(f"Early stop (patience={self.patience}). Restoring best model.")
                self.model.load_state_dict(best)
                break

    @torch.no_grad()
    def test(self, test_loader, A: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        tr = {"mae_miss": 0.0, "mre_miss": 0.0}
        n = 0
        for X, M in test_loader:
            for b in range(X.shape[0]):
                stats = self.eval_batch(X[b], A, M[b])
                tr["mae_miss"] += stats["mae_miss"]
                tr["mre_miss"] += stats["mre_miss"]
                n += 1
        tr["mae_miss"] /= max(1, n)
        tr["mre_miss"] /= max(1, n)
        return tr
