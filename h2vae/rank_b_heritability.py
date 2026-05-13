"""Rank-B updated method-of-moments heritability and genetic correlation.

Maintains factored state ``u_raw = X^T Z`` and ``w_raw = W^T Z`` across
minibatches.  At the start of each epoch ``rebuild(Z)`` streams the
genome once to populate the state; within the epoch each minibatch
applies a rank-B update using ``BedFile.decode_rows`` on B sample
indices plus a small matmul for ``w_raw``.  The residualised projection
of ``X^T Z`` against the covariate space ``W`` is then exact:

    (PX)^T Z = X^T P Z = X^T Z − X^T W (W^T W)^{-1} W^T Z
            = u_raw − M w_raw

with ``M = X^T W (W^T W)^{-1}`` constant and precomputed at setup.
``tr((PKP)²)`` is estimated once via Hutchinson with ``b_hutch``
Gaussian probes (two BED passes); ``tr(PKP)`` is computed exactly from
per-variant statistics.

Two modes:

* **mom** (default): replicates ``h2vae.heritability.mom()`` exactly.
  ``W = C`` (no intercept prepended, matching mom's convention);
  ``W = None`` when ``C`` is ``None``, in which case no projection is
  applied and the no-C formula is used.
* **gc**: replicates ``h2vae.heritability.gc()``.  Requires a target
  phenotype ``y_target``.  ``W = [1 | C]`` (intercept-augmented,
  matching gc's convention); ``C`` may be ``None`` (W is then just the
  intercept).  Precomputes ``V y_target``, ``VKV y_target`` and the
  fixed scalars ``tr_Ktil``, ``tr_Ktil2``, ``gc_det``, ``d2``.

Both modes share the rank-B update pathway through ``u_raw`` and
``w_raw``; the per-dim loss / display formulas differ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from h2vae.plink import BedFile


def _stream_variant_chunks(bed: BedFile, row_idx: np.ndarray,
                           chunk: int = 4096):
    """Yield ``(j_lo, j_hi, X_int8)`` for variant chunks."""
    m = bed.m
    j = 0
    while j < m:
        j_hi = min(j + chunk, m)
        yield j, j_hi, bed.decode_variants(j, j_hi, row_idx=row_idx)
        j = j_hi


class RankBHeritability(nn.Module):
    """Streaming-style MoM heritability / gc with rank-B minibatch updates.

    Args:
        bed: PLINK ``BedFile`` (mmap'd; consumed lazily).
        row_idx: ``(n,)`` int array of BED sample-row indices in the
            cohort; same row order as the ``Z`` passed to ``rebuild``.
        C: ``(n, c)`` covariate matrix to residualise against, or
            ``None``.  Meaning is mode-specific:
              * ``mom`` mode: ``W = C``; ``None`` ⇒ no projection.
              * ``gc`` mode:  ``W = [1 | C]``; ``None`` ⇒ ``W = [1]``.
        y_target: ``(n, 1)`` reference phenotype.  ``None`` selects mom
            mode; non-``None`` selects gc mode.
        hweights: optional ``(zdim,)`` per-latent-dim weighting for the
            scalar loss reduction.
        device: torch device for state tensors.
        chunk_variants: BED stream chunk size in variants.
        b_hutch: Hutchinson probe count for ``tr((PKP)²)``.
        seed_hutch: RNG seed for the Hutchinson probes.
        dtype: state dtype (default fp32).
    """

    def __init__(
        self,
        bed: BedFile,
        row_idx: np.ndarray,
        C: Tensor | None = None,
        y_target: Tensor | None = None,
        hweights: Tensor | None = None,
        device: torch.device | str = "cpu",
        chunk_variants: int = 4096,
        b_hutch: int = 10,
        seed_hutch: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bed = bed
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.n = int(len(self.row_idx))
        self.m = int(bed.m)
        self.device = torch.device(device)
        self.chunk = int(chunk_variants)
        self.dtype = dtype
        self.mode = "gc" if y_target is not None else "mom"

        # --- Variant standardisation + tr(K) (one BED stream) ----------
        mean, sd, n_obs = self._compute_variant_stats()
        self.register_buffer("var_mean", torch.from_numpy(mean).to(self.device, dtype))
        self.register_buffer("var_sd",   torch.from_numpy(sd).to(self.device, dtype))
        self.tr_K = float(n_obs.sum() / self.m)

        # --- Build W per mode ------------------------------------------
        if self.mode == "mom":
            if C is None:
                self.has_W = False
                self.c = 0
                self.nc = float(self.n)
                self._W: Tensor | None = None
            else:
                C = C.to(self.device, dtype)
                if C.shape[0] != self.n:
                    raise ValueError(
                        f"C has {C.shape[0]} rows; cohort has {self.n}"
                    )
                self.register_buffer("_W", C)
                self.has_W = True
                self.c = int(C.shape[1])
                self.nc = float(self.n - self.c)
        else:  # gc
            ones = torch.ones((self.n, 1), device=self.device, dtype=dtype)
            if C is None:
                W = ones
            else:
                C = C.to(self.device, dtype)
                if C.shape[0] != self.n:
                    raise ValueError(
                        f"C has {C.shape[0]} rows; cohort has {self.n}"
                    )
                W = torch.hstack((ones, C))
            self.register_buffer("_W", W)
            self.has_W = True
            self.c = int(W.shape[1])
            self.nc = float(self.n - self.c)

        # --- WtW_inv, X^T W, M, tr(PKP) --------------------------------
        if self.has_W:
            WtW = self._W.T @ self._W                          # (c, c)
            self.register_buffer("_WtW_inv", torch.linalg.inv(WtW))
            XtW = self._compute_XtW(self._W)                    # (m, c)
            self.register_buffer("_XtW", XtW)
            self.register_buffer("_M", XtW @ self._WtW_inv)     # (m, c)
            tr_K_W = float((self._M * XtW).sum() / self.m)
            self.tr_pkp = self.tr_K - tr_K_W
            # Precompute W^T 1 (constant) for the centering correction.
            self.register_buffer("_W_col_sum",
                                  self._W.sum(dim=0))           # (c,)
        else:
            self.tr_pkp = self.tr_K

        # --- tr((PKP)²) via Hutchinson ---------------------------------
        self.b_hutch = int(b_hutch)
        self.register_buffer("tr_pkp2", self._hutchinson_trace_K2(seed_hutch))

        # --- gc-mode precomputation -----------------------------------
        if self.mode == "gc":
            y2 = y_target.to(self.device, dtype)
            if y2.ndim == 1:
                y2 = y2[:, None]
            if y2.shape[0] != self.n:
                raise ValueError(
                    f"y_target has {y2.shape[0]} rows; cohort has {self.n}"
                )
            # Match gc()'s _standardize: (y - mean) / std (unbiased).
            mu_y2 = y2.mean(dim=0, keepdim=True)
            sd_y2 = y2.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
            y2 = (y2 - mu_y2) / sd_y2                          # (n, 1)
            self.register_buffer("y_target", y2)
            self._precompute_gc()                              # fills V_y2 etc.

        # --- Empty state buffers (sized on first rebuild) -------------
        self.register_buffer("u_raw", torch.empty(0, dtype=dtype, device=self.device))
        if self.has_W:
            self.register_buffer("w_raw", torch.empty(0, dtype=dtype, device=self.device))
        self.register_buffer("Z_prev", torch.empty(0, dtype=dtype, device=self.device))

        if hweights is not None:
            self.register_buffer("hweights", hweights.to(self.device, dtype))
        else:
            self.hweights = None

    # ==================================================================
    # Setup-time streaming primitives
    # ==================================================================

    def _compute_variant_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.empty(self.m, dtype=np.float64)
        var = np.empty(self.m, dtype=np.float64)
        n_obs = np.empty(self.m, dtype=np.float64)
        for j_lo, j_hi, X_int8 in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            X = X_int8.astype(np.float64)
            mask = X != -1
            ct = mask.sum(axis=0)
            ct_safe = np.where(ct == 0, 1, ct)
            X_for_mean = np.where(mask, X, 0.0)
            mu = X_for_mean.sum(axis=0) / ct_safe
            sq = (np.where(mask, X - mu, 0.0) ** 2).sum(axis=0) / ct_safe
            mean[j_lo:j_hi] = mu
            var[j_lo:j_hi] = sq
            n_obs[j_lo:j_hi] = ct
        sd = np.sqrt(var)
        sd[sd < 1e-8] = 1e-8
        return mean.astype(np.float32), sd.astype(np.float32), n_obs

    def _decode_chunk_std(self, j_lo: int, j_hi: int,
                          row_idx: np.ndarray | None = None) -> Tensor:
        rows = self.row_idx if row_idx is None else row_idx
        X_int8 = self.bed.decode_variants(j_lo, j_hi, row_idx=rows)
        X = torch.from_numpy(X_int8.astype(np.float32)).to(self.device)
        mu = self.var_mean[j_lo:j_hi]
        sd = self.var_sd[j_lo:j_hi]
        missing = X.eq(-1)
        if missing.any():
            X = torch.where(missing, mu.expand_as(X), X)
        return (X - mu) / sd

    def _decode_rows_std(self, rows: np.ndarray) -> Tensor:
        X_int8 = self.bed.decode_rows(rows)
        X = torch.from_numpy(X_int8.astype(np.float32)).to(self.device)
        mu = self.var_mean
        sd = self.var_sd
        missing = X.eq(-1)
        if missing.any():
            X = torch.where(missing, mu.expand_as(X), X)
        return (X - mu) / sd

    def _compute_XtW(self, W: Tensor) -> Tensor:
        XtW = torch.zeros((self.m, W.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            XtW[j_lo:j_hi] = Xc.T @ W
        return XtW

    @torch.no_grad()
    def _hutchinson_trace_K2(self, seed: int) -> Tensor:
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        Z = torch.randn(self.n, self.b_hutch, generator=rng).to(self.device, self.dtype)

        if self.has_W:
            PZ = Z - self._W @ (self._WtW_inv @ (self._W.T @ Z))
        else:
            PZ = Z

        # Stream 1: v = X^T (P Z)
        v = torch.zeros((self.m, self.b_hutch), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            v[j_lo:j_hi] = Xc.T @ PZ

        # Stream 2: y = X v
        y = torch.zeros((self.n, self.b_hutch), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            y += Xc @ v[j_lo:j_hi]

        if self.has_W:
            Py = y - self._W @ (self._WtW_inv @ (self._W.T @ y))
        else:
            Py = y

        norm_sq = (Py * Py).sum(dim=0)
        return (norm_sq.mean() / (self.m ** 2)).detach()

    @torch.no_grad()
    def _precompute_gc(self) -> None:
        """gc-mode precomputation: V y2, KV y2, VKV y2, gc_det, d2."""
        y2 = self.y_target                                     # (n, 1)
        # V y2 (already standardised internally).
        V_y2 = y2 - self._W @ (self._WtW_inv @ (self._W.T @ y2))
        self.register_buffer("_V_y2", V_y2)

        # K V_y2  via streaming: X^T V_y2 (m-vec), then X (X^T V_y2) (n-vec).
        XtV_y2 = torch.zeros((self.m, 1), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            XtV_y2[j_lo:j_hi] = Xc.T @ V_y2
        # X (X^T V_y2)
        KV_y2 = torch.zeros((self.n, 1), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            KV_y2 += Xc @ XtV_y2[j_lo:j_hi]
        KV_y2 = KV_y2 / self.m
        self.register_buffer("_KV_y2", KV_y2)

        # VKV y2 = V (KV y2)
        VKV_y2 = KV_y2 - self._W @ (self._WtW_inv @ (self._W.T @ KV_y2))
        self.register_buffer("_VKV_y2", VKV_y2)

        # gc_det = tr_Ktil2 · nc − tr_Ktil²
        tr_pkp2_scalar = self.tr_pkp2
        gc_det = tr_pkp2_scalar * self.nc - self.tr_pkp ** 2
        self.register_buffer("_gc_det", gc_det.detach())

        # d2 = (y2 · VKV y2) · nc − (y2 · V y2) · tr_Ktil, floored.
        d2_raw = ((y2 * VKV_y2).sum() * self.nc
                  - (y2 * V_y2).sum() * self.tr_pkp)
        self.register_buffer("_d2", torch.clamp(d2_raw, min=1e-8))

    # ==================================================================
    # Epoch-start rebuild
    # ==================================================================

    @torch.no_grad()
    def rebuild(self, Z: Tensor) -> None:
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")
        Z = Z.detach().to(self.device, self.dtype)
        zdim = Z.shape[1]
        u_raw = torch.zeros((self.m, zdim), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            u_raw[j_lo:j_hi] = Xc.T @ Z
        self.u_raw = u_raw
        if self.has_W:
            self.w_raw = self._W.T @ Z
        self.Z_prev = Z.clone()

    # ==================================================================
    # Per-minibatch update + loss
    # ==================================================================

    def update_and_loss(self, Z_batch: Tensor, cohort_idx: Tensor) -> Tensor:
        if self.u_raw.numel() == 0:
            raise RuntimeError("Call rebuild(Z) before update_and_loss().")
        if Z_batch.shape[0] != cohort_idx.shape[0]:
            raise ValueError(
                f"Z_batch rows ({Z_batch.shape[0]}) != "
                f"cohort_idx len ({cohort_idx.shape[0]})"
            )

        cohort_idx_np = cohort_idx.detach().cpu().numpy().astype(np.int64)
        bed_rows = self.row_idx[cohort_idx_np]

        u_prev = self.u_raw.detach()
        Z_prev_batch = self.Z_prev[cohort_idx]
        delta_Z = Z_batch - Z_prev_batch

        X_batch = self._decode_rows_std(bed_rows)              # (B, m)
        delta_u = X_batch.T @ delta_Z
        u_new = u_prev + delta_u

        if self.has_W:
            w_prev = self.w_raw.detach()
            W_batch = self._W[cohort_idx]                      # (B, c)
            delta_w = W_batch.T @ delta_Z
            w_new = w_prev + delta_w
        else:
            w_new = None

        per_dim = self._per_dim_signal(u_new, w_new, Z_batch, cohort_idx)

        self.u_raw = u_new.detach()
        if self.has_W:
            self.w_raw = w_new.detach()
        with torch.no_grad():
            self.Z_prev[cohort_idx] = Z_batch.detach()

        # Both mom and gc loss formulas already return per-dim values in
        # the "higher = better" direction.  Negate to make this a loss.
        if self.hweights is None:
            return -per_dim.sum()
        return -(per_dim * self.hweights).sum()

    # ==================================================================
    # Per-dim signal (mom: h²; gc: γ̂)
    # ==================================================================

    def _per_dim_signal(self, u: Tensor, w: Tensor | None,
                        Z_batch: Tensor, cohort_idx: Tensor) -> Tensor:
        """Per-dim h² (mom) or γ̂ (gc), matching the upstream estimators."""
        # Reconstruct live Z so that per-column mean / std are current.
        Z = self.Z_prev.clone()
        Z[cohort_idx] = Z_batch

        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)

        if self.mode == "mom":
            return self._mom_h2(u, w, Z, mu, sd)
        return self._gc_gamma(Z, mu, sd)

    def _mom_h2(self, u: Tensor, w: Tensor | None,
                Z: Tensor, mu: Tensor, sd: Tensor) -> Tensor:
        u_std = u / sd                                          # (m, zdim); X^T 1 = 0

        if not self.has_W:
            q_pkp = (u_std * u_std).sum(dim=0) / self.m         # (zdim,)
            num = q_pkp - (self.n - 1)
            denom = self.tr_pkp2 - self.n                       # = tr(K²) − n
            return num / denom

        # With covariates.  Center Z first (W has no intercept under mom).
        Z_centered = Z - mu
        Zs = Z_centered / sd

        CtZs = self._W.T @ Zs                                   # (c, zdim)
        u_resid = u_std - self._M @ CtZs                        # (m, zdim)
        q_pkp = (u_resid * u_resid).sum(dim=0) / self.m

        zz = (Zs * Zs).sum(dim=0)                               # ≈ n-1
        wwz = (CtZs * (self._WtW_inv @ CtZs)).sum(dim=0)
        q_p = zz - wwz

        # 2x2 system per dim:  A V = B
        device = self.device
        dtype = self.dtype
        tr_pkp_t = torch.as_tensor(self.tr_pkp, device=device, dtype=dtype)
        nc_t = torch.as_tensor(self.nc, device=device, dtype=dtype)
        A = torch.stack([
            torch.stack([self.tr_pkp2, tr_pkp_t]),
            torch.stack([tr_pkp_t, nc_t]),
        ])
        B = torch.stack([q_pkp, q_p], dim=0)
        V = torch.linalg.solve(A, B)
        V_sum = V.sum(dim=0)
        sign = torch.where(V_sum >= 0, 1.0, -1.0)
        return V[0] / (V_sum + 1e-8 * sign)

    def _gc_gamma(self, Z: Tensor, mu: Tensor, sd: Tensor) -> Tensor:
        """gc-mode genetic covariance γ̂ per dim (matches gc()'s loss).

        Uses the live ``Z`` (with the minibatch already patched in by
        the caller, preserving the autograd graph through ``Z_batch``).
        The dot products are O(n) per latent dim and avoid any per-step
        BED stream.
        """
        Zs = (Z - mu) / sd                                      # (n, zdim)
        num = ((Zs * self._VKV_y2).sum(dim=0) * self.nc
               - (Zs * self._V_y2).sum(dim=0) * self.tr_pkp)
        return num / self._gc_det

    # ==================================================================
    # Display
    # ==================================================================

    @torch.no_grad()
    def display(self, Z: Tensor) -> Tensor:
        """Per-dim h² (mom) or ρ̂ (gc), bounded for gc."""
        Z = Z.detach().to(self.device, self.dtype)
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")

        # Fresh rebuild on the supplied Z (gradient-free; only used for
        # display so we don't disturb the running state).
        u = torch.zeros((self.m, Z.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi, _ in _stream_variant_chunks(
            self.bed, self.row_idx, self.chunk
        ):
            Xc = self._decode_chunk_std(j_lo, j_hi)
            u[j_lo:j_hi] = Xc.T @ Z
        w = self._W.T @ Z if self.has_W else None

        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)

        if self.mode == "mom":
            return self._mom_h2(u, w, Z, mu, sd)

        # gc: ρ̂ = num / sqrt(d1 · d2), clamped to [-1, 1].
        Zs = (Z - mu) / sd
        num = (Zs * self._VKV_y2).sum(dim=0) * self.nc - (Zs * self._V_y2).sum(dim=0) * self.tr_pkp
        # d1 = (y1s · VKV y1s) · nc − (y1s · V y1s) · tr_Ktil
        # y1s · V y1s = ‖V y1s‖² = Zs^T V Zs = Zs^T Zs − Zs^T W (W'W)^{-1} W^T Zs
        WtZs = self._W.T @ Zs
        zz = (Zs * Zs).sum(dim=0)
        wwz = (WtZs * (self._WtW_inv @ WtZs)).sum(dim=0)
        q_p = zz - wwz
        # y1s · VKV y1s = ‖X^T V y1s‖² / m
        # X^T V y1s = u_std − M (W^T y1s),  u_std = u / sd (X^T 1 = 0).
        u_std = u / sd
        u_resid = u_std - self._M @ WtZs
        q_pkp = (u_resid * u_resid).sum(dim=0) / self.m
        d1_raw = q_pkp * self.nc - q_p * self.tr_pkp
        d1 = torch.clamp(d1_raw, min=1e-8)
        rho = num / torch.sqrt(d1 * self._d2)
        return torch.clamp(rho, min=-1.0, max=1.0)
