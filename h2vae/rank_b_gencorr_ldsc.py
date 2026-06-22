"""Differentiable LDSC genetic-correlation loss against an external trait.

Composes:

* a ``RankBHeritability`` instance (provides per-step GWAS-style
  Z-scores via ``update_and_sumstats``),
* a precomputed ``LDSCContext`` (external sumstats Z, ref-LD,
  regression-weight LD, M, aligned to the BED variant axis),
* the vendored ``h2vae.ldsc_torch.RG`` per latent dim,

into a single per-step loss ``-sum_d hweights_d · rg_d``.  Free
intercepts on both heritabilities and the gencov are the default,
absorbing residual population structure (hsq intercept) and
unknown sample overlap (gencov intercept).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from h2vae.ldsc_io import LDSCContext
from h2vae.ldsc_torch import RG
from h2vae.rank_b_heritability import RankBHeritability
from h2vae.rank_b_sumstats import compute_sumstats, update_and_sumstats


class RankBGenCorrLDSC(nn.Module):
    """Per-step LDSC rg loss using rank-B-streamed cohort sumstats.

    Args:
        rankb: a ``RankBHeritability`` instance.
        ctx: precomputed ``LDSCContext`` (output of
            ``h2vae.ldsc_io.build_ldsc_context``).
        n_cohort: scalar cohort sample size to use as the per-SNP N for
            the cohort sumstats.  Defaults to ``rankb.n - rankb.c``
            (residualisation-adjusted effective N).
        intercept_hsq: scalar or ``None``.  ``None`` (default) leaves
            both Hsq intercepts free, absorbing residual stratification
            in the cohort and in the external study.  Float forces
            both intercepts to that value.
        intercept_gencov: scalar or ``None``.  ``None`` (default)
            leaves the gencov intercept free to absorb sample overlap.
        hweights: optional ``(zdim,)`` weights for the loss reduction;
            ``loss = -sum_d hweights[d] · rg[d]``.
    """

    def __init__(self,
                 rankb: RankBHeritability,
                 ctx: LDSCContext,
                 n_cohort: Optional[float] = None,
                 intercept_hsq: Optional[float] = None,
                 intercept_gencov: Optional[float] = None,
                 hweights: Optional[Tensor] = None):
        super().__init__()
        self.rankb = rankb
        self.ctx = ctx.to(device=rankb.device, dtype=rankb.dtype)
        self.intercept_hsq = intercept_hsq
        self.intercept_gencov = intercept_gencov

        self.n_cohort = (float(n_cohort) if n_cohort is not None
                         else float(rankb.n - rankb.c))
        self.register_buffer(
            "N1",
            torch.full((self.ctx.m_use, 1), self.n_cohort,
                       device=rankb.device, dtype=rankb.dtype),
        )
        self.register_buffer(
            "bed_to_ldsc_idx_t",
            torch.from_numpy(self.ctx.bed_to_ldsc_idx).to(rankb.device),
        )

        if hweights is not None:
            self.register_buffer("hweights",
                                  hweights.to(device=rankb.device,
                                              dtype=rankb.dtype))
        else:
            self.hweights = None

        # Diagnostics populated by the last call to update_and_loss / display.
        self.last_rg: Optional[Tensor] = None
        self.last_intercepts: dict[str, Tensor] = {}
        self.last_skipped: list[int] = []

    # ------------------------------------------------------------------
    # Epoch boundary
    # ------------------------------------------------------------------

    def rebuild(self, Z: Tensor) -> None:
        self.rankb.rebuild(Z)

    # ------------------------------------------------------------------
    # Per-step loss + display
    # ------------------------------------------------------------------

    def update_and_loss(self, Z_batch: Tensor, idxs: Tensor) -> Tensor:
        out = update_and_sumstats(self.rankb, Z_batch, idxs)
        # Per-step loss only needs gencov for dims with non-zero hweight —
        # unpressured dims are masked out by `(gencov_vec * hweights).sum()`.
        # Skipping their RG construction here is a ~zdim/k_pressured speedup
        # for sparse hweights (e.g. `first8` on zdim=128 → 16× fewer RG
        # calls per minibatch step).
        active_dims = None
        if self.hweights is not None:
            active_dims = (self.hweights != 0).nonzero(as_tuple=True)[0].tolist()
        return self._loss_from_sumstats(out, active_dims=active_dims)

    @torch.no_grad()
    def display(self, Z: Tensor) -> Tensor:
        """Per-dim rg vector at the supplied Z snapshot (validation use)."""
        self.rankb.rebuild(Z)
        out = compute_sumstats(self.rankb)
        loss = self._loss_from_sumstats(out)  # all dims for display
        return self.last_rg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loss_from_sumstats(self, sumstats: dict, active_dims=None) -> Tensor:
        z_all = sumstats["z"]                                  # (m, zdim)
        z_aligned = z_all[self.bed_to_ldsc_idx_t]              # (m_use, zdim)
        zdim = z_aligned.shape[1]
        # `active_dims=None` means run RG for every dim (display path).
        # A non-None list restricts the per-dim RG loop to those indices
        # (loss path with sparse hweights).  Skipped dims get gencov=0 and
        # rg=0 entries in the output vectors so the loss sum and downstream
        # diagnostics still see length-zdim tensors.
        dims_to_run = set(range(zdim)) if active_dims is None else set(active_dims)

        # We optimise the genetic COVARIANCE (``r.gencov.tot``), not the
        # genetic correlation ``r.rg_ratio = gencov / sqrt(hsq1·hsq2)``.
        # Maximising the ratio creates an incentive to *shrink* the cohort
        # latent's hsq1 (small denominator → ratio explodes), which produces
        # apparently-large ρ̂ on near-noise dims without aligning any actual
        # heritable variance with the external trait.  ``gencov`` is unbounded
        # above and only grows by genuinely aligning the latent's heritable
        # component with the external study's, removing the small-denominator
        # gaming.  ``last_rg`` still stores the rg_ratio for diagnostic display
        # in the per-epoch h_val log lines.
        gencov_list: list[Tensor] = []
        rg_list: list[Tensor] = []
        gencov_tot_diag: list[float] = []
        int_hsq1: list[float] = []
        int_hsq2: list[float] = []
        int_gc: list[float] = []
        skipped: list[int] = []

        zero = torch.zeros((), device=self.rankb.device, dtype=self.rankb.dtype)

        for d in range(zdim):
            if d not in dims_to_run:
                gencov_list.append(zero)
                rg_list.append(zero)
                gencov_tot_diag.append(0.0)
                int_hsq1.append(float("nan"))
                int_hsq2.append(float("nan"))
                int_gc.append(float("nan"))
                continue
            # IRWLS occasionally hits a singular design matrix when the
            # cohort latent's Z²-scores are degenerate (constant, all-zero,
            # collinear with the LD-score regressor).  This fails the whole
            # per-step loss for *all* dims unless we catch it per-dim and
            # treat the singular dim as skipped (zero gencov contribution,
            # no graph dependency).  Mirrors the `_negative_hsq` skip path.
            try:
                r = RG(
                    z_aligned[:, d:d + 1], self.ctx.z_external,
                    self.ctx.ref_ld, self.ctx.w_ld,
                    self.N1, self.ctx.n_external, self.ctx.M,
                    intercept_hsq1=self.intercept_hsq,
                    intercept_hsq2=self.intercept_hsq,
                    intercept_gencov=self.intercept_gencov,
                )
            except torch._C._LinAlgError:
                gencov_list.append(zero)
                rg_list.append(zero)
                gencov_tot_diag.append(0.0)
                int_hsq1.append(float("nan"))
                int_hsq2.append(float("nan"))
                int_gc.append(float("nan"))
                skipped.append(d)
                continue

            gencov_t = (r.gencov.tot if isinstance(r.gencov.tot, Tensor)
                        else torch.as_tensor(r.gencov.tot,
                                             device=self.rankb.device,
                                             dtype=self.rankb.dtype))
            gencov_list.append(gencov_t)
            gencov_tot_diag.append(_as_float(r.gencov.tot))

            if r._negative_hsq:
                rg_list.append(zero)
                skipped.append(d)
            else:
                rg_t = (r.rg_ratio if isinstance(r.rg_ratio, Tensor)
                        else torch.as_tensor(r.rg_ratio,
                                             device=self.rankb.device,
                                             dtype=self.rankb.dtype))
                rg_list.append(rg_t)
            int_hsq1.append(_as_float(r.hsq1.intercept))
            int_hsq2.append(_as_float(r.hsq2.intercept))
            int_gc.append(_as_float(r.gencov.intercept))

        gencov_vec = torch.stack(gencov_list)                  # (zdim,)
        rg_vec = torch.stack(rg_list)                          # (zdim,)
        self.last_rg = rg_vec.detach()
        self.last_gencov = gencov_vec.detach()
        self.last_intercepts = {
            "hsq1": torch.tensor(int_hsq1),
            "hsq2": torch.tensor(int_hsq2),
            "gencov": torch.tensor(int_gc),
            "gencov_tot": torch.tensor(gencov_tot_diag),
        }
        self.last_skipped = skipped

        if self.hweights is None:
            return -gencov_vec.sum()
        return -(gencov_vec * self.hweights).sum()


def _as_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)
