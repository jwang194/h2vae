"""Per-SNP GWAS-style summary statistics from the rank-B state.

Reuses the cached state of a ``RankBHeritability`` instance to expose
differentiable per-SNP ``(β̂, SE, Z, χ²)`` for every latent dim.  Pure
read-only consumer of the module's existing buffers — does **not**
register any new buffers or alter any existing behaviour.

Math (per latent dim ``d`` and SNP ``j``, with ``W`` the residualisation
basis and ``V = I - W(W'W)^{-1}W'``):

    β̂_{j,d}  = (X_j^T V y_d) / d_j        where d_j = X_j^T V X_j
    SE²(β̂)  = σ̂²_d / d_j
    Z_{j,d}  = β̂_{j,d} / SE(β̂_{j,d})
    χ²_{j,d} = Z_{j,d}²

The numerator ``X_j^T V y_d`` is exactly the ``u_resid`` row already
formed by ``RankBHeritability._mom_h2`` — its rank-B update is what
makes the whole pipeline cheap.  ``d_j`` is a fixed setup constant
(computed lazily on first call and cached on the instance as
``_sumstats_d_j``).  ``σ̂²_d`` is the per-dim residual variance estimate
``||V y_d||² / (n - c - 1)``; both terms come from rank-B state with no
extra cache walks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def _get_or_compute_d_j(rankb) -> Tensor:
    """Compute and cache ``d_j = X_j^T V X_j`` on the instance.

    Does one cohort-cache walk (no BED I/O) on first call.  Subsequent
    calls return the cached tensor.  The cache attribute uses a private
    name to avoid collision with ``register_buffer``-managed state.
    """
    cached = getattr(rankb, "_sumstats_d_j", None)
    if cached is not None:
        return cached

    xtx = torch.zeros(rankb.m, device=rankb.device, dtype=rankb.dtype)
    for j_lo, j_hi in rankb._chunks():
        Xc = rankb._decode_chunk_std(j_lo, j_hi)
        xtx[j_lo:j_hi] = (Xc * Xc).sum(dim=0)

    if rankb.has_W:
        cov_term = (rankb._M * rankb._XtW).sum(dim=1)
        d_j = xtx - cov_term
    else:
        d_j = xtx

    d_j = d_j.clamp_min(1e-12)
    rankb._sumstats_d_j = d_j.detach()
    return rankb._sumstats_d_j


def compute_sumstats(rankb,
                     sigma_sq_d: Optional[Tensor] = None,
                     u_override: Optional[Tensor] = None,
                     Z_override: Optional[Tensor] = None) -> dict:
    """Differentiable per-SNP GWAS sumstats for the current ``Z_prev``.

    Args:
        rankb: a ``RankBHeritability`` instance whose ``rebuild`` /
            ``update_and_loss`` has populated ``u_raw`` / ``w_raw`` /
            ``Z_prev``.
        sigma_sq_d: optional ``(zdim,)`` override for the per-dim
            residual variance.  Default: ``||V Zs||² / (n - c - 1)``,
            computed from existing rank-B state (no cache walk).
        u_override: optional ``(m, zdim)`` tensor to use in place of
            ``rankb.u_raw``.  ``update_and_loss`` detaches ``u_raw`` at
            the end of each step; pass the un-detached ``u_new`` here
            to keep gradients flowing into ``Z_batch``.
        Z_override: optional ``(n, zdim)`` tensor to use in place of
            ``rankb.Z_prev``.  Pair with ``u_override`` for grad-aware
            calls from a wrapping module.

    Returns:
        Dict with keys ``beta``, ``se``, ``z``, ``chisq`` (all shape
        ``(m, zdim)``), ``d_j`` (shape ``(m,)``) and ``sigma_sq`` (shape
        ``(zdim,)``).
    """
    Z = rankb.Z_prev if Z_override is None else Z_override
    u = rankb.u_raw if u_override is None else u_override
    if u.numel() == 0:
        raise RuntimeError("Call rankb.rebuild(Z) before compute_sumstats().")

    mu = Z.mean(dim=0, keepdim=True)
    sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
    Zs = (Z - mu) / sd

    u_std = u / sd

    if rankb.has_W:
        CtZs = rankb._W.T @ Zs
        u_resid = u_std - rankb._M @ CtZs
    else:
        CtZs = None
        u_resid = u_std

    d_j = _get_or_compute_d_j(rankb)

    if sigma_sq_d is None:
        zz = (Zs * Zs).sum(dim=0)
        if rankb.has_W:
            wwz = (CtZs * (rankb._WtW_inv @ CtZs)).sum(dim=0)
            q_p = zz - wwz
        else:
            q_p = zz
        denom = max(rankb.n - rankb.c - 1, 1)
        sigma_sq_d = (q_p / denom).clamp_min(1e-12)
    else:
        sigma_sq_d = sigma_sq_d.to(device=rankb.device, dtype=rankb.dtype)

    d_j_col = d_j[:, None]
    beta = u_resid / d_j_col
    se = torch.sqrt(sigma_sq_d[None, :] / d_j_col)
    z = beta / se
    chisq = z * z

    return {
        "beta": beta,
        "se": se,
        "z": z,
        "chisq": chisq,
        "d_j": d_j,
        "sigma_sq": sigma_sq_d,
    }


def update_and_sumstats(rankb, Z_batch: Tensor, cohort_idx: Tensor,
                        sigma_sq_d: Optional[Tensor] = None) -> dict:
    """Rank-B minibatch step that returns grad-attached sumstats.

    Mirrors ``RankBHeritability.update_and_loss``'s rank-B update body
    but, instead of building the MoM/gc loss expression on the
    grad-attached ``u_new`` and then detaching, builds a
    ``compute_sumstats`` dict on ``u_new`` and returns it.  State is
    detached at the end of the step exactly as ``update_and_loss``
    leaves it, so subsequent ``update_and_sumstats`` (or
    ``update_and_loss``) calls see a clean fresh-step graph.

    This is the entry point for the LDSC genetic-correlation pipeline:
    the returned ``z``/``chisq`` carry grad to ``Z_batch`` via the
    per-step rank-B delta, exactly as ``update_and_loss``'s scalar
    loss does for the MoM path.

    .. note::
        This duplicates ~10 lines of the rank-B update from
        ``rank_b_heritability.update_and_loss``.  A follow-up cleanup
        should factor those into a shared ``_apply_update`` primitive
        on ``RankBHeritability`` and have both ``update_and_loss`` and
        ``update_and_sumstats`` consume it.
    """
    if rankb.u_raw.numel() == 0:
        raise RuntimeError("Call rankb.rebuild(Z) before update_and_sumstats().")
    if Z_batch.shape[0] != cohort_idx.shape[0]:
        raise ValueError(
            f"Z_batch rows ({Z_batch.shape[0]}) != "
            f"cohort_idx len ({cohort_idx.shape[0]})"
        )

    cohort_idx_np = cohort_idx.detach().cpu().numpy().astype(np.int64)

    u_prev = rankb.u_raw.detach()
    Z_prev_batch = rankb.Z_prev[cohort_idx]
    delta_Z = Z_batch - Z_prev_batch

    X_batch = rankb._decode_rows_std(cohort_idx_np)
    delta_u = X_batch.T @ delta_Z
    u_new = u_prev + delta_u

    if rankb.has_W:
        w_prev = rankb.w_raw.detach()
        W_batch = rankb._W[cohort_idx]
        delta_w = W_batch.T @ delta_Z
        w_new = w_prev + delta_w
    else:
        w_new = None

    # Full-cohort Z snapshot with Z_batch grad-attached in the cohort rows.
    # cohort_idx may arrive on CPU from the DataLoader; index_copy is strict
    # about device-matching (unlike fancy indexing above), so move it here.
    Z_full = rankb.Z_prev.clone()
    Z_full = Z_full.index_copy(0, cohort_idx.to(Z_full.device), Z_batch)

    out = compute_sumstats(rankb, sigma_sq_d=sigma_sq_d,
                            u_override=u_new, Z_override=Z_full)

    # Commit state for the next step (mirror update_and_loss).
    rankb.u_raw = u_new.detach()
    if rankb.has_W:
        rankb.w_raw = w_new.detach()
    with torch.no_grad():
        rankb.Z_prev[cohort_idx] = Z_batch.detach()

    return out
