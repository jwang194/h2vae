"""Rank-B heritability **spectrum** objective (linearly-accessible heritability).

The legacy :class:`RankBHeritability` loss maximizes a weighted sum of
*per-dimension* heritabilities.  A CCA analysis showed that objective only
applies a **linear rotation** of the latent basis (``Σ_d h²_d`` is not
rotation-invariant), so it cannot increase the heritability genuinely
accessible in *linear combinations* of the latents.

This module maximizes the **heritability spectrum** instead: the generalized
eigenvalues ``λ`` of ``G v = λ P v``, where ``G`` is the full pairwise
genetic-covariance matrix across latent dims and ``P`` is their phenotypic
correlation matrix.  ``Σλ = tr(P⁻¹G)`` is invariant under any invertible linear
reparameterization of the latents, so maximizing it forces the **nonlinear**
encoder to make more heritable variation linearly extractable.

The full ``G`` is the SCORE-OVERLAP estimator (:func:`h2vae.heritability.gc`)
applied to every latent pair — see ``notes/SPECTRUM-MATH.md`` (M1):

    u_resid = XᵀV Zs                      (rank-B state; gradients to the batch)
    Q_PKP   = u_residᵀ u_resid / m        (= z_iᵀ K̃ z_j)
    Q_P     = Zsᵀ Zs − CtZsᵀ WtW⁻¹ CtZs   (= z_iᵀ V z_j)
    G       = A⁻¹[0,0]·Q_PKP + A⁻¹[0,1]·Q_P   (A the shared 2×2 MoM system)
    P_corr  = D⁻¹ Q_P D⁻¹,  D = diag(√diag Q_P)
    λ       = eigvalsh(L⁻¹ G L⁻ᵀ),  L = chol(P_corr + ridge·I)   (M4)
    loss    = −Σ_k w_k λ_k                  (w = per-rank weights; uniform default)

All heavy setup (cohort cache, ``M``, ``WtW⁻¹``, traces) is inherited unchanged
from :class:`RankBHeritability`; only the per-minibatch reduction differs.  To
match ``gc()`` exactly we **prepend an intercept** to the residualization
covariates (``W=[1|C]``), unlike the legacy mom-path which uses ``W=C``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from h2vae.rank_b_heritability import RankBHeritability


class RankBHeritabilitySpectrum(RankBHeritability):
    """Rank-B maximizer of the heritability spectrum (see module docstring).

    Args:
        bed, row_idx, device, chunk_variants, b_hutch, seed_hutch, dtype:
            forwarded to :class:`RankBHeritability` (mom mode, ``y_target=None``).
        C: ``(n, c_user)`` residualization covariates *without* an intercept; an
            intercept column is prepended automatically so ``W=[1|C]`` (matching
            ``gc()``).  ``None`` ⇒ intercept-only projection ``W=[1]``.
        ridge: ridge added to the phenotypic correlation before whitening,
            ``P_reg = P_corr + ridge·I`` (keeps ``P_reg`` PD and tempers the
            eigendecomposition gradient).
        spectrum_clamp: if True, clamp the spectrum at 0 (``relu(λ)``) before the
            weighted sum — the differentiable nearest-PSD projection of the
            whitened matrix (see ``notes/SPECTRUM-MATH.md`` M3).
        rank_weights: optional ``(d,)`` per-**rank** weights applied to the
            descending-sorted spectrum (reuses the ``--hweights`` convention).
            ``None`` ⇒ uniform weights ⇒ objective = ``tr(P⁻¹G)``.  A one-hot
            vector with ones in the first ``k`` entries ⇒ top-``k`` objective.
        spectrum_dims: optional ``K`` — restrict the spectrum to the **first K
            latent dimensions** by slicing ``G[:K,:K]`` and ``P_corr[:K,:K]``
            before the generalized eigendecomposition.  The loss then pressures
            only that sub-block of the latent space (dims K..d-1 get only
            reconstruction + KL), an anti-overfitting knob distinct from the
            per-rank ``rank_weights`` truncation.  ``None``/``0`` ⇒ full d dims.
            When set, ``rank_weights`` (if given) is sliced to ``[:K]`` and must be
            at least length ``K``.
        spectrum_weight: coefficient λ_s on the spectrum loss.
        marginal_weight: coefficient λ_m on the **marginal** per-dim heritability
            loss (``−Σ_d h²_d``, the legacy objective, uniform over dims).  Default
            0 ⇒ pure spectrum.  When >0 the total loss is
            ``λ_s·spectrum_loss + λ_m·marginal_loss`` so the marginal (single-latent)
            heritabilities are pushed up alongside the linearly-accessible spectrum.
    """

    def __init__(
        self,
        bed,
        row_idx,
        C: Tensor | None = None,
        ridge: float = 1e-4,
        spectrum_clamp: bool = False,
        rank_weights: Tensor | None = None,
        spectrum_dims: int | None = None,
        spectrum_weight: float = 1.0,
        marginal_weight: float = 0.0,
        device: torch.device | str = "cpu",
        chunk_variants: int = 4096,
        b_hutch: int = 10,
        seed_hutch: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        # Prepend an intercept so W = [1 | C] (matches gc()'s projection).  When
        # C is None this gives W = [1].  The genotype side is unaffected
        # (standardized X ⇒ Xᵀ1 = 0); the intercept only completes the Zs
        # residualization.  Passed as the mom-mode covariate matrix, so the
        # parent builds W = C_aug with has_W=True.
        dev = torch.device(device)
        ones = torch.ones((len(row_idx), 1), device=dev, dtype=dtype)
        if C is None:
            C_aug = ones
        else:
            C_aug = torch.hstack((ones, C.to(dev, dtype)))

        super().__init__(
            bed, row_idx,
            C=C_aug,
            y_target=None,          # mom mode
            hweights=None,          # the spectrum loss applies per-rank weights itself
            device=device,
            chunk_variants=chunk_variants,
            b_hutch=b_hutch,
            seed_hutch=seed_hutch,
            dtype=dtype,
        )
        assert self.mode == "mom" and self.has_W, "spectrum estimator requires W=[1|C]"
        self.ridge = float(ridge)
        self.spectrum_clamp = bool(spectrum_clamp)
        self.spectrum_weight = float(spectrum_weight)
        self.marginal_weight = float(marginal_weight)
        # 0/None ⇒ full latent dimensionality (no sub-block restriction).
        self.spectrum_dims = int(spectrum_dims) if spectrum_dims else None
        if rank_weights is not None:
            if self.spectrum_dims is not None and rank_weights.shape[0] < self.spectrum_dims:
                raise ValueError(
                    f"rank_weights has length {rank_weights.shape[0]} but "
                    f"spectrum_dims={self.spectrum_dims} requires at least that many "
                    f"per-rank weights (the spectrum is restricted to the first "
                    f"{self.spectrum_dims} dims). Pass a matched K-length --hweights "
                    f"file (e.g. aux/uniform.{self.spectrum_dims}.weights)."
                )
            self.register_buffer("rank_weights", rank_weights.to(self.device, self.dtype))
        else:
            self.rank_weights = None

    # ==================================================================
    # Full genetic-covariance + phenotypic-correlation matrices
    # ==================================================================

    def _genetic_cov_matrix(self, u: Tensor, w: Tensor | None,
                            Z: Tensor, mu: Tensor, sd: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(G, P_corr)`` — the full d×d matrices (see M1).

        Mirrors the has_W branch of :meth:`RankBHeritability._mom_h2` but keeps
        the off-diagonals (Gram matrices, not just their diagonals).
        """
        u_std = u / sd                                    # (m, d)
        Zs = (Z - mu) / sd                                # (n, d)
        CtZs = self._W.T @ Zs                             # (c, d)
        u_resid = u_std - self._M @ CtZs                  # (m, d)

        Q_PKP = (u_resid.T @ u_resid) / self.m            # z_iᵀ K̃ z_j
        Q_P = Zs.T @ Zs - CtZs.T @ (self._WtW_inv @ CtZs) # z_iᵀ V z_j

        # Shared 2×2 MoM system A (identical for all pairs); build per-call so a
        # test-patched self.tr_pkp2 is respected, exactly like _mom_h2.
        tr_pkp_t = torch.as_tensor(self.tr_pkp, device=self.device, dtype=self.dtype)
        nc_t = torch.as_tensor(self.nc, device=self.device, dtype=self.dtype)
        A = torch.stack([
            torch.stack([self.tr_pkp2, tr_pkp_t]),
            torch.stack([tr_pkp_t, nc_t]),
        ])
        Ainv = torch.linalg.inv(A)
        G = Ainv[0, 0] * Q_PKP + Ainv[0, 1] * Q_P
        G = 0.5 * (G + G.T)

        d = torch.sqrt(torch.diagonal(Q_P).clamp_min(1e-12))
        P_corr = Q_P / (d[:, None] * d[None, :])
        P_corr = 0.5 * (P_corr + P_corr.T)
        return G, P_corr

    def _take(self, M: Tensor) -> Tensor:
        """Restrict a d×d matrix to its first-K sub-block when ``spectrum_dims`` set.

        Used to confine the heritability spectrum (and the held-out eval that
        scores train-optimal directions on val) to a chosen sub-block of latent
        dimensions, leaving the remaining dims unpressured by the genetic loss.
        """
        if self.spectrum_dims is None:
            return M
        k = min(self.spectrum_dims, M.shape[0])
        return M[:k, :k]

    def _spectrum(self, G: Tensor, P_corr: Tensor) -> Tensor:
        """Descending-sorted generalized eigenvalues of ``G v = λ P_corr v``.

        Uses the symmetric Cholesky whitening ``M = L⁻¹ G L⁻ᵀ`` (M4): real
        eigenvalues, same spectrum as ``P⁻¹G``.  When ``spectrum_dims`` is set the
        d×d matrices are first restricted to their first-K sub-block.
        """
        G = self._take(G)
        P_corr = self._take(P_corr)
        zdim = G.shape[0]
        I = torch.eye(zdim, device=self.device, dtype=self.dtype)
        P_reg = P_corr + self.ridge * I                   # trace(P_corr)=zdim ⇒ +ridge·I
        L = torch.linalg.cholesky(P_reg)
        Linv = torch.linalg.inv(L)
        M = Linv @ G @ Linv.T
        M = 0.5 * (M + M.T)
        lam = torch.linalg.eigvalsh(M)                    # ascending
        return lam.flip(0)                                # descending

    def _spectrum_loss(self, G: Tensor, P_corr: Tensor) -> Tensor:
        """Negative weighted sum of the spectrum (the spectrum training loss)."""
        lam = self._spectrum(G, P_corr)
        if self.spectrum_clamp:
            lam = lam.clamp_min(0)
        if self.rank_weights is None:
            return -lam.sum()
        k = min(lam.shape[0], self.rank_weights.shape[0])
        return -(lam[:k] * self.rank_weights[:k]).sum()

    def _marginal_loss(self, u: Tensor, w: Tensor | None,
                       Z: Tensor, mu: Tensor, sd: Tensor) -> Tensor:
        """Marginal (per-dim) heritability loss ``−Σ_d h²_d`` (uniform over dims).

        Reuses the inherited :meth:`RankBHeritability._mom_h2` (per-dim h² with the
        same W=[1|C] residualization), so the marginal objective is the legacy
        single-latent heritability summed over dimensions.
        """
        per_dim = self._mom_h2(u, w, Z, mu, sd)
        return -per_dim.sum()

    # ==================================================================
    # Per-minibatch update + loss (override; rank-B bookkeeping matches parent)
    # ==================================================================

    def update_and_loss(self, Z_batch: Tensor, cohort_idx: Tensor) -> Tensor:
        if self.u_raw.numel() == 0:
            raise RuntimeError("Call rebuild(Z) before update_and_loss().")
        if Z_batch.shape[0] != cohort_idx.shape[0]:
            raise ValueError(
                f"Z_batch rows ({Z_batch.shape[0]}) != "
                f"cohort_idx len ({cohort_idx.shape[0]})"
            )

        cohort_idx_np = cohort_idx.detach().cpu().numpy().astype("int64")

        # --- rank-B state update (mirrors RankBHeritability.update_and_loss) ---
        u_prev = self.u_raw.detach()
        Z_prev_batch = self.Z_prev[cohort_idx]
        delta_Z = Z_batch - Z_prev_batch
        X_batch = self._decode_rows_std(cohort_idx_np)
        delta_u = X_batch.T @ delta_Z
        u_new = u_prev + delta_u

        w_prev = self.w_raw.detach()
        W_batch = self._W[cohort_idx]
        delta_w = W_batch.T @ delta_Z
        w_new = w_prev + delta_w

        # --- full-Z splice (batch rows live), then the combined loss ---
        Z = self.Z_prev.clone()
        Z[cohort_idx] = Z_batch
        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
        G, P_corr = self._genetic_cov_matrix(u_new, w_new, Z, mu, sd)
        loss = self.spectrum_weight * self._spectrum_loss(G, P_corr)
        if self.marginal_weight != 0.0:
            loss = loss + self.marginal_weight * self._marginal_loss(
                u_new, w_new, Z, mu, sd)

        # --- detach state for the next step (mirrors parent) ---
        self.u_raw = u_new.detach()
        self.w_raw = w_new.detach()
        with torch.no_grad():
            self.Z_prev[cohort_idx] = Z_batch.detach()

        return loss

    # ==================================================================
    # Display: inherited display() (per-dim h²) stays valid; spectrum here
    # ==================================================================

    @torch.no_grad()
    def spectrum_display(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        """Full-cohort spectrum snapshot. Returns ``(spectrum_desc, total)``.

        ``spectrum_desc`` are the descending generalized eigenvalues (the per-
        combination heritabilities); ``total = Σλ = tr(P⁻¹G)``.
        """
        Z = Z.detach().to(self.device, self.dtype)
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")

        u = torch.zeros((self.m, Z.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            Xc = self._decode_chunk_std(j_lo, j_hi)
            u[j_lo:j_hi] = Xc.T @ Z
        w = self._W.T @ Z

        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
        G, P_corr = self._genetic_cov_matrix(u, w, Z, mu, sd)
        lam = self._spectrum(G, P_corr)
        return lam, lam.sum()

    @torch.no_grad()
    def eig_decompose(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        """Full-cohort generalized eig: returns ``(spectrum_desc, W)``.

        ``W`` are the generalized eigenvectors (columns), ``P_reg``-orthonormal
        (``Wᵀ P_reg W = I``), ordered to match the descending spectrum.  Use ``W``
        from one cohort (e.g. train) with :meth:`heritability_of_directions` on
        another (e.g. val) to get a **held-out** spectrum.
        """
        Z = Z.detach().to(self.device, self.dtype)
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")
        u = torch.zeros((self.m, Z.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            u[j_lo:j_hi] = self._decode_chunk_std(j_lo, j_hi).T @ Z
        w = self._W.T @ Z
        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
        G, P_corr = self._genetic_cov_matrix(u, w, Z, mu, sd)
        G = self._take(G)            # restrict to first-K sub-block when set
        P_corr = self._take(P_corr)

        zdim = G.shape[0]
        I = torch.eye(zdim, device=self.device, dtype=self.dtype)
        L = torch.linalg.cholesky(P_corr + self.ridge * I)
        Linv = torch.linalg.inv(L)
        M = 0.5 * (Linv @ G @ Linv.T + (Linv @ G @ Linv.T).T)
        lam, U = torch.linalg.eigh(M)              # ascending; M U = U diag(lam)
        V = Linv.T @ U                             # generalized eigvecs, Wᵀ P_reg W = I
        return lam.flip(0), V.flip(1)

    @torch.no_grad()
    def heritability_of_directions(self, Z: Tensor, W: Tensor) -> Tensor:
        """Heritability of FIXED directions ``W`` under THIS cohort's (G, P).

        ``h²_k = (w_kᵀ G w_k) / (w_kᵀ P_reg w_k)`` for each column of ``W``.  With
        ``W`` taken from the train cohort (:meth:`eig_decompose`) and ``Z`` the val
        latents, this is the **held-out spectrum**: the train-optimal combinations
        scored on independent val data (the overfit leading direction deflates to
        its honest value).
        """
        Z = Z.detach().to(self.device, self.dtype)
        if Z.shape[0] != self.n:
            raise ValueError(f"Z has {Z.shape[0]} rows; cohort has {self.n}")
        W = W.to(self.device, self.dtype)
        u = torch.zeros((self.m, Z.shape[1]), device=self.device, dtype=self.dtype)
        for j_lo, j_hi in self._chunks():
            u[j_lo:j_hi] = self._decode_chunk_std(j_lo, j_hi).T @ Z
        w = self._W.T @ Z
        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
        G, P_corr = self._genetic_cov_matrix(u, w, Z, mu, sd)
        G = self._take(G)            # match the K-dim directions W from eig_decompose
        P_corr = self._take(P_corr)
        P_reg = P_corr + self.ridge * torch.eye(
            G.shape[0], device=self.device, dtype=self.dtype)
        gvar = torch.einsum("ik,ij,jk->k", W, G, W)
        pvar = torch.einsum("ik,ij,jk->k", W, P_reg, W)
        return gvar / pvar
