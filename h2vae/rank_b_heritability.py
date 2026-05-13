"""Rank-B updated method-of-moments heritability and genetic correlation.

Maintains factored state ``u_raw = X^T Z`` and ``w_raw = W^T Z`` across
minibatches.  At the start of each epoch ``rebuild(Z)`` recomputes
``u_raw`` from scratch; within the epoch each minibatch applies a
rank-B update using a sample-row gather on the cohort cache, plus a
small matmul for ``w_raw``.  Residualised projection of ``X^T Z`` is
exact via

    (PX)^T Z = X^T Z − X^T W (W^T W)^{-1} W^T Z = u_raw − M w_raw

with ``M = X^T W (W^T W)^{-1}`` constant.  ``tr((PKP)²)`` is estimated
once via Hutchinson with ``b_hutch`` Gaussian probes (in-memory cache
walks); ``tr(PKP)`` is exact from per-variant statistics.

**Performance architecture** (see ``notes/rank_b_heritability_perf.md``,
combined options 5+6+8): on construction, the BED is streamed **once**
to build a bit-packed sample-major cohort cache (``CohortCache``,
``~19.5 GB`` per cohort at UKB scale) and accumulate variant stats in
the same pass.  Every subsequent operation —
``_compute_XtW``, ``_hutchinson_trace_K2``, ``_precompute_gc``,
``rebuild``, ``update_and_loss`` — reads from the in-memory cache
rather than the BED.  This matches SCORE's Round 1 per-fit BED traffic.

Two modes:

* **mom** (default): replicates ``h2vae.heritability.mom()`` exactly.
* **gc**: replicates ``h2vae.heritability.gc()`` (loss = γ̂,
  ``.display`` = ρ̂).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from h2vae.cohort_cache import CohortCache
from h2vae.plink import BedFile


class RankBHeritability(nn.Module):
    """Streaming-style MoM heritability / gc with rank-B minibatch updates.

    Args:
        bed: PLINK ``BedFile``; consumed exactly once at construction.
        row_idx: ``(n,)`` int array of BED sample-row indices in the
            cohort; same row order as the ``Z`` passed to ``rebuild``.
        C: ``(n, c)`` covariate matrix to residualise against, or
            ``None``.  Mode-specific:
              * mom mode: ``W = C``; ``None`` ⇒ no projection.
              * gc mode:  ``W = [1 | C]``; ``None`` ⇒ ``W = [1]``.
        y_target: ``(n, 1)`` reference phenotype.  ``None`` selects mom
            mode; non-``None`` selects gc mode.
        hweights: optional ``(zdim,)`` per-latent-dim weighting for the
            scalar loss reduction.
        device: torch device for state tensors.
        chunk_variants: variant chunk size used during build / walks.
            Must be a multiple of 4 (CohortCache alignment).
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

        # --- The ONE BED pass: build cache + variant stats together ---
        self.cache = CohortCache(self.n, self.m, chunk_variants=self.chunk)
        mean, sd, n_obs = self._build_cache_and_compute_variant_stats()
        # All subsequent setup work reads from `self.cache`, not `self.bed`.
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

        # --- WtW_inv, X^T W, M, tr(PKP) (cache walks) -------------------
        if self.has_W:
            WtW = self._W.T @ self._W
            self.register_buffer("_WtW_inv", torch.linalg.inv(WtW))
            XtW = self._compute_XtW(self._W)
            self.register_buffer("_XtW", XtW)
            self.register_buffer("_M", XtW @ self._WtW_inv)
            tr_K_W = float((self._M * XtW).sum() / self.m)
            self.tr_pkp = self.tr_K - tr_K_W
            self.register_buffer("_W_col_sum", self._W.sum(dim=0))
        else:
            self.tr_pkp = self.tr_K

        # --- tr((PKP)²) via Hutchinson (cache walks) --------------------
        self.b_hutch = int(b_hutch)
        self.register_buffer("tr_pkp2", self._hutchinson_trace_K2(seed_hutch))

        # --- gc-mode precomputation (cache walks) ----------------------
        if self.mode == "gc":
            y2 = y_target.to(self.device, dtype)
            if y2.ndim == 1:
                y2 = y2[:, None]
            if y2.shape[0] != self.n:
                raise ValueError(
                    f"y_target has {y2.shape[0]} rows; cohort has {self.n}"
                )
            mu_y2 = y2.mean(dim=0, keepdim=True)
            sd_y2 = y2.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
            y2 = (y2 - mu_y2) / sd_y2
            self.register_buffer("y_target", y2)
            self._precompute_gc()

        # --- Empty per-step state buffers (sized on first rebuild) -----
        self.register_buffer("u_raw", torch.empty(0, dtype=dtype, device=self.device))
        if self.has_W:
            self.register_buffer("w_raw", torch.empty(0, dtype=dtype, device=self.device))
        self.register_buffer("Z_prev", torch.empty(0, dtype=dtype, device=self.device))

        if hweights is not None:
            self.register_buffer("hweights", hweights.to(self.device, dtype))
        else:
            self.hweights = None

    # ==================================================================
    # Chunk iteration over the cache
    # ==================================================================

    def _chunks(self):
        """Yield ``(j_lo, j_hi)`` over the variant axis at the build chunk size."""
        j = 0
        while j < self.m:
            yield j, min(j + self.chunk, self.m)
            j += self.chunk

    # ==================================================================
    # ONE-PASS build: BED → cache + variant stats together
    # ==================================================================

    def _build_cache_and_compute_variant_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single BED pass.  Populates ``self.cache`` and returns
        ``(mean, sd, n_obs)`` for every variant.

        Variance is computed by the two-pass identity
        ``var = Σ(X − μ)² / n`` on an fp32 deviation working copy
        (option 5 in the perf notes) — half the memory footprint of the
        original fp64 path, and slightly more numerically stable than
        the one-pass ``E[X²] − (E[X])²`` formula.
        """
        mean = np.empty(self.m, dtype=np.float64)
        var = np.empty(self.m, dtype=np.float64)
        n_obs = np.empty(self.m, dtype=np.int64)
        for j_lo, j_hi in self._chunks():
            X_int8 = self.bed.decode_variants(j_lo, j_hi, row_idx=self.row_idx)
            # --- Cache build for this chunk ---
            self.cache.build_chunk(j_lo, j_hi, X_int8)
            # --- Variant stats (option 5: int + fp32 working tensors) ---
            mask = X_int8 != -1
            n_observed = mask.sum(axis=0, dtype=np.int64)
            ct_safe = np.maximum(n_observed, 1).astype(np.float64)
            X_int = X_int8.copy()
            np.putmask(X_int, ~mask, 0)
            sum_x = X_int.sum(axis=0, dtype=np.int64)
            mu = sum_x / ct_safe
            dev = X_int.astype(np.float32) - mu.astype(np.float32)
            np.putmask(dev, ~mask, 0.0)
            sq = np.einsum("ij,ij->j", dev, dev, dtype=np.float64)
            v = sq / ct_safe
            mean[j_lo:j_hi] = mu
            var[j_lo:j_hi] = v
            n_obs[j_lo:j_hi] = n_observed
        self.cache.finalise()
        sd = np.sqrt(var)
        sd[sd < 1e-8] = 1e-8
        return mean, sd, n_obs.astype(np.float64)

    # ==================================================================
    # Cache-backed standardised decoders (no BED I/O)
    # ==================================================================

    def _decode_chunk_std(self, j_lo: int, j_hi: int) -> Tensor:
        """Standardised fp32 ``(n_cohort, j_hi - j_lo)`` chunk from cache."""
        X_int8 = self.cache.decode_variant_chunk(j_lo, j_hi)
        X = torch.from_numpy(X_int8.astype(np.float32)).to(self.device)
        mu = self.var_mean[j_lo:j_hi]
        sd = self.var_sd[j_lo:j_hi]
        missing = X.eq(-1)
        if missing.any():
            X = torch.where(missing, mu.expand_as(X), X)
        return (X - mu) / sd

    def _decode_rows_std(self, cohort_idx: np.ndarray) -> Tensor:
        """Standardised fp32 ``(B, m)`` row gather from cache.

        ``cohort_idx`` indexes into the cohort (not the BED).
        """
        X_int8 = self.cache.decode_rows(cohort_idx)
        X = torch.from_numpy(X_int8.astype(np.float32)).to(self.device)
        mu = self.var_mean
        sd = self.var_sd
        missing = X.eq(-1)
        if missing.any():
            X = torch.where(missing, mu.expand_as(X), X)
        return (X - mu) / sd

    # ==================================================================
    # Setup-time accumulators (all reading from the cache)
    # ==================================================================

    def _compute_XtW(self, W: Tensor) -> Tensor:
        XtW = torch.zeros((self.m, W.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
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

        v = torch.zeros((self.m, self.b_hutch), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            Xc = self._decode_chunk_std(j_lo, j_hi)
            v[j_lo:j_hi] = Xc.T @ PZ

        y = torch.zeros((self.n, self.b_hutch), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
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
        y2 = self.y_target
        V_y2 = y2 - self._W @ (self._WtW_inv @ (self._W.T @ y2))
        self.register_buffer("_V_y2", V_y2)

        XtV_y2 = torch.zeros((self.m, 1), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            Xc = self._decode_chunk_std(j_lo, j_hi)
            XtV_y2[j_lo:j_hi] = Xc.T @ V_y2
        KV_y2 = torch.zeros((self.n, 1), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            Xc = self._decode_chunk_std(j_lo, j_hi)
            KV_y2 += Xc @ XtV_y2[j_lo:j_hi]
        KV_y2 = KV_y2 / self.m
        self.register_buffer("_KV_y2", KV_y2)

        VKV_y2 = KV_y2 - self._W @ (self._WtW_inv @ (self._W.T @ KV_y2))
        self.register_buffer("_VKV_y2", VKV_y2)

        gc_det = self.tr_pkp2 * self.nc - self.tr_pkp ** 2
        self.register_buffer("_gc_det", gc_det.detach())

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
        for j_lo, j_hi in self._chunks():
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

        u_prev = self.u_raw.detach()
        Z_prev_batch = self.Z_prev[cohort_idx]
        delta_Z = Z_batch - Z_prev_batch

        # Rank-B BED-row gather → straight from the cache, no disk.
        X_batch = self._decode_rows_std(cohort_idx_np)
        delta_u = X_batch.T @ delta_Z
        u_new = u_prev + delta_u

        if self.has_W:
            w_prev = self.w_raw.detach()
            W_batch = self._W[cohort_idx]
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

        if self.hweights is None:
            return -per_dim.sum()
        return -(per_dim * self.hweights).sum()

    # ==================================================================
    # Per-dim signal (mom: h²; gc: γ̂)
    # ==================================================================

    def _per_dim_signal(self, u: Tensor, w: Tensor | None,
                        Z_batch: Tensor, cohort_idx: Tensor) -> Tensor:
        Z = self.Z_prev.clone()
        Z[cohort_idx] = Z_batch

        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)

        if self.mode == "mom":
            return self._mom_h2(u, w, Z, mu, sd)
        return self._gc_gamma(Z, mu, sd)

    def _mom_h2(self, u: Tensor, w: Tensor | None,
                Z: Tensor, mu: Tensor, sd: Tensor) -> Tensor:
        u_std = u / sd

        if not self.has_W:
            q_pkp = (u_std * u_std).sum(dim=0) / self.m
            num = q_pkp - (self.n - 1)
            denom = self.tr_pkp2 - self.n
            return num / denom

        Z_centered = Z - mu
        Zs = Z_centered / sd

        CtZs = self._W.T @ Zs
        u_resid = u_std - self._M @ CtZs
        q_pkp = (u_resid * u_resid).sum(dim=0) / self.m

        zz = (Zs * Zs).sum(dim=0)
        wwz = (CtZs * (self._WtW_inv @ CtZs)).sum(dim=0)
        q_p = zz - wwz

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
        Zs = (Z - mu) / sd
        num = ((Zs * self._VKV_y2).sum(dim=0) * self.nc
               - (Zs * self._V_y2).sum(dim=0) * self.tr_pkp)
        return num / self._gc_det

    # ==================================================================
    # Display
    # ==================================================================

    @torch.no_grad()
    def display(self, Z: Tensor) -> Tensor:
        Z = Z.detach().to(self.device, self.dtype)
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")

        u = torch.zeros((self.m, Z.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            Xc = self._decode_chunk_std(j_lo, j_hi)
            u[j_lo:j_hi] = Xc.T @ Z
        w = self._W.T @ Z if self.has_W else None

        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)

        if self.mode == "mom":
            return self._mom_h2(u, w, Z, mu, sd)

        Zs = (Z - mu) / sd
        num = (Zs * self._VKV_y2).sum(dim=0) * self.nc - (Zs * self._V_y2).sum(dim=0) * self.tr_pkp
        WtZs = self._W.T @ Zs
        zz = (Zs * Zs).sum(dim=0)
        wwz = (WtZs * (self._WtW_inv @ WtZs)).sum(dim=0)
        q_p = zz - wwz
        u_std = u / sd
        u_resid = u_std - self._M @ WtZs
        q_pkp = (u_resid * u_resid).sum(dim=0) / self.m
        d1_raw = q_pkp * self.nc - q_p * self.tr_pkp
        d1 = torch.clamp(d1_raw, min=1e-8)
        rho = num / torch.sqrt(d1 * self._d2)
        return torch.clamp(rho, min=-1.0, max=1.0)
