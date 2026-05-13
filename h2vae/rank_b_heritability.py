"""Rank-B updated heritability quadratic form for h2vae training.

Maintains ``u = X^T Z`` as state across minibatches within an epoch.
At the start of each epoch, ``rebuild(Z)`` recomputes ``u`` from
scratch (gradient-free, will eventually call a mailman streaming
kernel). Within the epoch, each minibatch calls ``update_and_loss``,
which applies a rank-``B`` update to ``u`` using only the rows of
``X`` at the current minibatch indices.

Loss form: ``L(Z) = ||X^T Z||_F^2 / m_scale``. This is the quadratic
form ``y^T K y`` (summed over latent dims) with ``K = X X^T / m``,
ignoring the MoM denominator (which is constant in ``Z``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RankBHeritability(nn.Module):
    """Stateful quadratic-form heritability with rank-B minibatch updates.

    Args:
        X: ``(n, m)`` tensor of (standardized) genotypes. Held as a
            non-trainable buffer. In production this can be backed by a
            row-reader / BED-decoder; for the prototype we just hold
            the full matrix.
        m_scale: Divisor for the loss. Defaults to ``m`` (so
            ``L = y^T (X X^T / m) y`` summed over latent dims).
    """

    def __init__(self, X: Tensor, m_scale: float | None = None):
        super().__init__()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")
        n, m = X.shape
        self.n = n
        self.m = m
        self.m_scale = float(m if m_scale is None else m_scale)
        self.register_buffer("X", X.detach())
        # State buffers, sized lazily on first rebuild.
        self.register_buffer("u", torch.empty(0))
        self.register_buffer("Z_prev", torch.empty(0))

    @torch.no_grad()
    def rebuild(self, Z: Tensor) -> None:
        """Recompute ``u = X^T Z`` from scratch (start-of-epoch refresh).

        Gradient-free — this is the call site that, in the streaming
        production version, would invoke the mailman kernel.

        Args:
            Z: ``(n, zdim)`` latent matrix.
        """
        if Z.shape[0] != self.n:
            raise ValueError(
                f"Z has {Z.shape[0]} rows; X has {self.n}"
            )
        Z = Z.detach().to(self.X.dtype)
        self.u = self.X.T @ Z          # (m, zdim)
        self.Z_prev = Z.clone()        # (n, zdim)

    def update_and_loss(self, Z_batch: Tensor, idxs: Tensor) -> Tensor:
        """Apply rank-B update and return the heritability loss scalar.

        Args:
            Z_batch: ``(B, zdim)`` latent rows for the current minibatch,
                live in the autograd graph (gradient flows back to the
                encoder).
            idxs: ``(B,)`` long tensor of indices into ``X`` / ``Z_prev``.

        Returns:
            Scalar tensor ``||u_new||_F^2 / m_scale``.
        """
        if self.u.numel() == 0:
            raise RuntimeError("Call rebuild(Z) before update_and_loss().")
        if Z_batch.shape[0] != idxs.shape[0]:
            raise ValueError(
                f"Z_batch rows ({Z_batch.shape[0]}) != idxs len ({idxs.shape[0]})"
            )

        # Detach u from any previous graph history. Gradient only flows
        # through delta_Z this step.
        u_prev = self.u.detach()

        # Difference between fresh and stale minibatch rows. Only
        # Z_batch is live; Z_prev[idxs] is a frozen snapshot.
        delta_Z = Z_batch - self.Z_prev[idxs]              # (B, zdim)

        # Rank-B update via a plain torch.matmul. Autograd-friendly.
        X_batch = self.X[idxs]                              # (B, m)
        delta_u = X_batch.T @ delta_Z                       # (m, zdim)
        u_new = u_prev + delta_u                            # (m, zdim)

        loss = (u_new ** 2).sum() / self.m_scale

        # Persist for the next step. Detach to prevent graph growth.
        self.u = u_new.detach()
        with torch.no_grad():
            self.Z_prev[idxs] = Z_batch.detach()

        return loss

    @torch.no_grad()
    def full_loss(self, Z: Tensor) -> Tensor:
        """Ground-truth ``||X^T Z||_F^2 / m_scale`` from the full ``Z``.

        For testing / sanity-checking only.
        """
        Z = Z.detach().to(self.X.dtype)
        u = self.X.T @ Z
        return (u ** 2).sum() / self.m_scale
