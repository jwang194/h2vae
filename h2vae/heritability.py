"""Method-of-moments heritability and genetic correlation estimators.

All estimator functions return a callable ``loss(y)`` that maps a phenotype
vector to a scalar estimate, enabling use as differentiable loss terms in
a training loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import relu, pad
from torch.nn import L1Loss


# ---------------------------------------------------------------------------
# Randomized trace estimator (optional fast path for large n)
# ---------------------------------------------------------------------------

def _rtrace_K2(X: np.ndarray, l: int) -> float:
    """Randomized trace estimator for tr(XX'XX').

    Args:
        X: Matrix of shape (n, m).
        l: Number of random probe vectors.

    Returns:
        Approximate trace of (XX')^2.
    """
    probes = np.random.randn(X.shape[0], l)
    W = probes.T @ (X @ (X.T @ (X @ (X.T @ probes))))
    return np.trace(W) / l


# ---------------------------------------------------------------------------
# Method-of-moments heritability estimator (unified, with optional covariates)
# ---------------------------------------------------------------------------

def mom(
    X: Tensor,
    kinship: bool = False,
    C: Tensor | None = None,
    device: torch.device = torch.device("cuda:0"),
) -> Callable[[Tensor], Tensor]:
    """Method-of-moments heritability estimator.

    When ``C`` is None, computes the standard Haseman-Elston-style estimator:
        h² = y'(K - I)y / (tr(K²) - n)

    When ``C`` is provided, projects out covariates and solves the resulting
    normal equations on the projected kinship matrix.

    The returned callable accepts y of shape ``(n, 1)`` (single phenotype)
    or ``(n, d)`` (batched). When batched, returns a vector of ``d``
    heritability estimates computed in a single matrix multiply.

    Args:
        X: Genetic data — either a genotype matrix (n, m) or a kinship
            matrix (n, n) if ``kinship=True``.
        kinship: If True, treat X as a precomputed kinship matrix.
        C: Optional covariate matrix of shape (n, c). When provided,
            covariates are projected out before estimation.
        device: Torch device for computation.

    Returns:
        A callable ``loss(y) -> Tensor``.
    """
    n, m = X.shape
    K = X if kinship else (X @ X.T) / m

    if C is None:
        # Standard MoM without covariates
        quad = K.to(device) - torch.diag(torch.ones(n, device=device))
        trace = torch.trace(K @ K)
        denom = trace - n

        def loss(y: Tensor) -> Tensor:
            yp = y - y.mean(dim=0, keepdim=True)
            ypp = yp / (yp.std(dim=0, keepdim=True) + 1e-8)
            # Batched quadratic form: diag(ypp' @ quad @ ypp)
            return (ypp * (quad @ ypp)).sum(dim=0) / denom

    else:
        # MoM with covariate projection
        c = C.shape[1]
        P = torch.eye(n, device=device) - C @ torch.inverse(C.T @ C) @ C.T
        PKP = P @ K.to(device) @ P
        PKP2 = PKP @ PKP
        tr_pkp = torch.trace(PKP)
        tr_pkp2 = torch.trace(PKP2)
        nc = torch.tensor(float(n - c), device=device)
        A = torch.stack([
            torch.stack([tr_pkp2, tr_pkp]),
            torch.stack([tr_pkp, nc]),
        ])

        def loss(y: Tensor) -> Tensor:
            yp = y - y.mean(dim=0, keepdim=True)
            ypp = yp / (yp.std(dim=0, keepdim=True) + 1e-8)
            # Batched quadratic forms
            q_pkp = (ypp * (PKP @ ypp)).sum(dim=0)   # (d,)
            q_p = (ypp * (P @ ypp)).sum(dim=0)        # (d,)
            # Solve 2x2 system per dimension: A @ V = B
            B = torch.stack([q_pkp, q_p], dim=0)      # (2, d)
            V = torch.linalg.solve(A, B)               # (2, d)
            V_sum = V.sum(dim=0)
            # Preserve sign of denominator; add epsilon away from zero
            return V[0] / (V_sum + 1e-8 * torch.where(V_sum >= 0, 1.0, -1.0))

    return loss


# ---------------------------------------------------------------------------
# OLS variance-explained estimator
# ---------------------------------------------------------------------------

def var_exp(
    X: Tensor,
    C: Tensor,
    device: torch.device = torch.device("cuda:0"),
) -> Callable[[Tensor], Tensor]:
    """OLS-based variance-explained estimator, conditional on covariates.

    The returned callable accepts y of shape ``(n, 1)`` (single phenotype)
    or ``(n, d)`` (batched). When batched, returns a vector of ``d``
    variance-explained estimates computed in a single matrix multiply.

    Args:
        X: Explanatory variables, centered and scaled, shape (n, m).
        C: Covariate matrix, shape (n, c).
        device: Torch device.

    Returns:
        A callable ``loss(y) -> Tensor`` estimating variance explained by X.
    """
    C = torch.hstack((torch.ones((C.shape[0], 1), device=device), C))
    CI = torch.inverse(C.T @ C)
    CP = lambda x: x - C @ (CI @ (C.T @ x))
    CPX = CP(X)
    CPXI = torch.inverse(CPX.T @ CPX)
    CPXP = lambda x: x - CPX @ (CPXI @ (CPX.T @ x))

    def loss(y: Tensor) -> Tensor:
        yp = y - y.mean(dim=0, keepdim=True)
        ypp = yp / (yp.std(dim=0, keepdim=True) + 1e-8)
        Py = CP(ypp)
        R = CPXP(Py)
        return Py.var(dim=0) - R.var(dim=0)

    return loss


# ---------------------------------------------------------------------------
# First-order Taylor expansion of variance explained
# ---------------------------------------------------------------------------

def var_exp_taylor(
    X: Tensor,
    C: Tensor,
    Z: Tensor,
    device: torch.device = torch.device("cuda:0"),
) -> Callable[[Tensor, Tensor], Tensor]:
    """First-order Taylor expansion of RSS around a reference Z.

    Args:
        X: Explanatory variables, shape (n, m).
        C: Covariate matrix, shape (n, c).
        Z: Reference phenotype vector for the expansion, shape (n, 1).
        device: Torch device.

    Returns:
        A callable ``loss(y, idxs) -> Tensor``.
    """
    n, c = C.shape
    CI = torch.inverse(C.T @ C)
    CP = lambda x: x - C @ (CI @ (C.T @ x))
    CPX = CP(X)
    CPXI = torch.inverse(CPX.T @ CPX)
    CPXP = lambda x: CPX @ (CPXI @ (CPX.T @ x))

    Zpp = Z / (torch.std(Z) + 1e-8)
    CPZ = CP(Zpp)
    CPXPCPZ = CPXP(CPZ)
    CPXP2CPZ = CPXP(CPXPCPZ)
    product = 2 * (CPZ - CPXPCPZ) + CPXP2CPZ

    Zbeta = CI @ (C.T @ Z)
    _CPZ_batch = lambda x, idxs: x - C[idxs] @ Zbeta

    del CI, CP, CPXI, CPXP

    def loss(y: Tensor, idxs: Tensor) -> Tensor:
        yp = y - torch.mean(y)
        ypp = yp / (torch.std(yp) + 1e-8)
        Py = _CPZ_batch(ypp, idxs)
        return -1 * (torch.dot(product[idxs][:, 0], Py[:, 0])).sum()

    return loss


# ---------------------------------------------------------------------------
# Factory for efficient per-epoch Taylor loss creation
# ---------------------------------------------------------------------------

class VarExpTaylorFactory:
    """Precomputes Z-independent quantities once; creates per-dim losses cheaply.

    ``var_exp_taylor`` recomputes expensive matrix inversions every call.
    When iterating over latent dimensions each epoch, the Z-independent work
    (covariate projection, genotype projection) is identical.  This factory
    does that work once at construction and exposes ``make_loss(Z_col)`` which
    only performs the cheap Z-dependent matmuls.

    Args:
        X: Genotype matrix, shape (n, m).
        C: Covariate matrix, shape (n, c).
        device: Torch device.
    """

    def __init__(self, X: Tensor, C: Tensor, device: torch.device):
        self.C = C.to(device)
        CI = torch.inverse(self.C.T @ self.C)
        self.CI = CI
        CPX = X.to(device) - self.C @ (CI @ (self.C.T @ X.to(device)))
        CPXI = torch.inverse(CPX.T @ CPX)
        self.CPX = CPX
        self.CPXI = CPXI

    def _cp(self, x: Tensor) -> Tensor:
        return x - self.C @ (self.CI @ (self.C.T @ x))

    def _cpxp(self, x: Tensor) -> Tensor:
        return self.CPX @ (self.CPXI @ (self.CPX.T @ x))

    def make_loss(self, Z_col: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a ``loss(y, idxs)`` callable for one latent dimension.

        Args:
            Z_col: Reference phenotype column, shape (n, 1).

        Returns:
            Callable with signature ``(y, idxs) -> Tensor``.
        """
        Zpp = Z_col / (torch.std(Z_col) + 1e-8)
        CPZ = self._cp(Zpp)
        CPXPCPZ = self._cpxp(CPZ)
        CPXP2CPZ = self._cpxp(CPXPCPZ)
        product = 2 * (CPZ - CPXPCPZ) + CPXP2CPZ

        Zbeta = self.CI @ (self.C.T @ Z_col)
        C = self.C

        def loss(y: Tensor, idxs: Tensor) -> Tensor:
            yp = y - torch.mean(y)
            ypp = yp / (torch.std(yp) + 1e-8)
            Py = ypp - C[idxs] @ Zbeta
            return -1 * (torch.dot(product[idxs][:, 0], Py[:, 0])).sum()

        return loss


# ---------------------------------------------------------------------------
# Genetic correlation estimator
# ---------------------------------------------------------------------------

def gc(
    X: Tensor,
    y2: Tensor,
    kinship: bool = False,
    C: Tensor | None = None,
    device: torch.device = torch.device("cuda:0"),
) -> Callable[[Tensor], Tensor]:
    """SCORE-OVERLAP genetic-correlation estimator.

    Implements Wu et al. 2022 (AJHG) Eq. 6 in the no-covariate case and the
    covariate-adjusted analogue from their Eq. 9 (specialised to the
    complete-sample-overlap case, where both traits share covariates ``W``
    and hence the projection ``V = I - W(W'W)^{-1}W'``).  Defining
    ``K̃ = V K V`` and ``c = #columns(W)`` (intercept-augmented, so ``c >= 1``),

        γ̂ , σ̂²_{g1} , σ̂²_{g2}  each solve a 2x2 system with the shared
        left-hand matrix  [[tr(K̃²), tr(K̃)], [tr(K̃), n-c]]  and
        right-hand side coming from  (y1' K̃ y2, y1' V y2) ,
                                     (y1' K̃ y1, y1' V y1) , and
                                     (y2' K̃ y2, y2' V y2) respectively.

    The shared denominator of that 2x2 solve cancels when forming
    ``ρ̂ = γ̂ / sqrt(σ̂²_{g1} * σ̂²_{g2})``, leaving the closed form

        ρ̂ = (y1' K̃ y2 · (n-c) − y1' V y2 · tr(K̃))
             / sqrt((y1' K̃ y1 · (n-c) − y1' V y1 · tr(K̃))
                  · (y2' K̃ y2 · (n-c) − y2' V y2 · tr(K̃))) .

    With ``C=None`` the intercept-only W gives the standard no-covariate
    estimator.

    The returned callable accepts ``y1`` of shape ``(n, 1)`` (single
    phenotype) or ``(n, d)`` (batched).  Both ``y1`` and ``y2`` are
    standardised internally (matching the ``mom``/``var_exp`` convention),
    so the callable is invariant under rescaling of the latents.

    Two outputs are exposed:

    * ``loss(y1) -> Tensor`` (the bare callable) returns the genetic-
      covariance estimate ``γ̂`` itself: the SCORE-OVERLAP numerator divided
      by the fixed positive constant ``det = tr(K̃²)·(n-c) − tr(K̃)²``.  This
      is the *training* signal — finite for any input, no ``sqrt`` of a
      sign-ambiguous quantity, invariant to latent magnitude (since the
      inputs are standardised), and **O(1) in magnitude per dim** so that
      h_weight on the same scale as MoM (~0.01–0.1) gives a comparable
      regularisation strength.

    * ``loss.display(y1) -> Tensor`` returns the full ``ρ̂``, clamped to
      ``[-1, 1]`` (Cauchy-Schwarz).  Use this for logging / human-readable
      heritability values.  The clamp only binds in the degenerate-variance
      regime where the unbounded MoM ratio has no statistical meaning.

    Args:
        X: Genetic data — genotype matrix (n, m) or kinship matrix (n, n).
        y2: Reference phenotype vector, shape (n, 1) or (n,).
        kinship: If True, treat X as a precomputed GRM (n, n).
        C: Optional covariate matrix of shape (n, c_user) *without* an
            intercept; an intercept column is prepended automatically (so
            y1 and y2 are implicitly mean-centred).
        device: Torch device for computation.

    Returns:
        A callable ``loss(y1) -> Tensor`` with a ``.display`` attribute
        for the correlation form.
    """
    n = X.shape[0]
    K = X if kinship else (X @ X.T) / X.shape[1]
    K = K.to(device)
    y2 = y2.to(device, K.dtype)
    if y2.ndim == 1:
        y2 = y2[:, None]

    ones = torch.ones((n, 1), device=device, dtype=K.dtype)
    if C is None:
        W = ones
    else:
        W = torch.hstack((ones, C.to(device, K.dtype)))
    c = W.shape[1]
    nc = n - c

    WtW_I = torch.inverse(W.T @ W)
    V_of = lambda x: x - W @ (WtW_I @ (W.T @ x))

    # tr(V K V) without materialising V as (n, n):
    #   tr(VKV) = tr(KV²) = tr(KV)
    #          = tr(K) - tr(W(W'W)^{-1} W' K)
    #          = tr(K) - tr((W'W)^{-1} W'KW)    (cyclic)
    WtKW = W.T @ (K @ W)                                # (c, c)
    tr_Ktil = torch.trace(K) - torch.trace(WtW_I @ WtKW)

    # tr(K̃²) = tr((VK)²) (uses V² = V, then cyclic).  Materialises one (n, n)
    # intermediate during setup and frees it; this lets us divide the training
    # signal below by the fixed positive constant `gc_det`, so the bare loss is
    # O(1) (proportional to γ̂) rather than O(n²).  Without this normalisation,
    # h_weight is profoundly unintuitive: gc raw loss is ~10⁸ at n≈13k, so
    # h_weight=1e-5 silently delivers effective loss ~10³ — orders of magnitude
    # larger than MoM-style h losses at h_weight=0.05.
    WtK = W.T @ K                                       # (c, n)
    VK_tmp = K - W @ (WtW_I @ WtK)                      # (n, n) — V applied from left
    tr_Ktil2 = (VK_tmp * VK_tmp.T).sum()
    del VK_tmp
    gc_det = tr_Ktil2 * nc - tr_Ktil * tr_Ktil          # fixed, positive

    # Standardise y2 once (mean-zero, unit variance) so the result is
    # invariant to its scale, mirroring mom/var_exp.
    y2 = (y2 - y2.mean(dim=0, keepdim=True)) / (y2.std(dim=0, keepdim=True) + 1e-8)
    V_y2 = V_of(y2)                                     # (n, 1)
    KV_y2 = K @ V_y2                                    # (n, 1)
    VKV_y2 = V_of(KV_y2)                                # (n, 1)
    # Variance estimate for y2 (numerator of σ̂²_{g2}); needed for the
    # display/correlation form only.  Floor matches the d1 floor below.
    d2_raw = (y2 * VKV_y2).sum() * nc - (y2 * V_y2).sum() * tr_Ktil
    d2 = torch.clamp(d2_raw, min=1e-8)

    def _standardize(y: Tensor) -> Tensor:
        return (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-8)

    def loss(y1: Tensor) -> Tensor:
        """Genetic-covariance estimate γ̂ (training signal, O(1) per dim)."""
        y1s = _standardize(y1)
        num = (y1s * VKV_y2).sum(dim=0) * nc - (y1s * V_y2).sum(dim=0) * tr_Ktil
        return num / gc_det

    def display(y1: Tensor) -> Tensor:
        """Full SCORE-OVERLAP ρ̂, clamped to [-1, 1] for stability."""
        y1s = _standardize(y1)
        num = (y1s * VKV_y2).sum(dim=0) * nc - (y1s * V_y2).sum(dim=0) * tr_Ktil
        V_y1 = V_of(y1s)
        VKV_y1 = V_of(K @ V_y1)
        d1_raw = (y1s * VKV_y1).sum(dim=0) * nc - (y1s * V_y1).sum(dim=0) * tr_Ktil
        d1 = torch.clamp(d1_raw, min=1e-8)
        return torch.clamp(num / torch.sqrt(d1 * d2), min=-1.0, max=1.0)

    loss.display = display
    return loss


# ---------------------------------------------------------------------------
# Heritability-spectrum dense reference (test/eval oracle for rank-B spectrum)
# ---------------------------------------------------------------------------

def gcov_spectrum(
    X: Tensor,
    C: Tensor | None = None,
    kinship: bool = False,
    ridge: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    """Dense reference for the heritability spectrum (no rank-B).

    The full pairwise genetic-covariance matrix ``G`` is the SCORE-OVERLAP
    estimator :func:`gc` applied to every latent pair; ``P_corr`` is the
    residualized phenotypic *correlation* matrix; the spectrum is the
    generalized eigenvalues of ``G v = λ P_corr v``.  See
    ``notes/SPECTRUM-MATH.md``.  This is the dense analogue of
    :class:`h2vae.rank_b_spectrum.RankBHeritabilitySpectrum`, used to validate
    the rank-B implementation; it is not used as a training loss.

    To match ``gc()`` / the rank-B estimator, an intercept column is prepended
    to ``C`` (so ``W=[1|C]``; ``W=[1]`` when ``C=None``).

    Args:
        X: genotype matrix ``(n, m)`` (standardized) or GRM ``(n, n)`` if
            ``kinship``.
        C: residualization covariates ``(n, c_user)`` without an intercept, or
            ``None``.
        kinship: treat ``X`` as a precomputed GRM.
        ridge: ridge added to ``P_corr`` before whitening.
        device: torch device.

    Returns:
        ``loss(Z) -> (G, P_corr, spectrum)`` where ``Z`` is ``(n, d)``; ``G`` and
        ``P_corr`` are ``(d, d)`` and ``spectrum`` is the descending ``(d,)``
        generalized eigenvalues.
    """
    X = X.to(device)
    n = X.shape[0]
    K = X if kinship else (X @ X.T) / X.shape[1]

    ones = torch.ones((n, 1), device=device, dtype=X.dtype)
    W = ones if C is None else torch.hstack((ones, C.to(device, X.dtype)))
    c = W.shape[1]
    nc = float(n - c)

    WtW_inv = torch.linalg.inv(W.T @ W)
    V = torch.eye(n, device=device, dtype=X.dtype) - W @ (WtW_inv @ W.T)
    PKP = V @ K @ V                                       # K̃ = V K V
    tr_pkp = torch.trace(PKP)
    tr_pkp2 = torch.trace(PKP @ PKP)
    A = torch.stack([
        torch.stack([tr_pkp2, tr_pkp]),
        torch.stack([tr_pkp, torch.as_tensor(nc, device=device, dtype=X.dtype)]),
    ])
    Ainv = torch.linalg.inv(A)

    def loss(Z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        Z = Z.to(device, X.dtype)
        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
        Zs = (Z - mu) / sd
        Q_PKP = Zs.T @ (PKP @ Zs)                         # z_iᵀ K̃ z_j
        Q_P = Zs.T @ (V @ Zs)                             # z_iᵀ V z_j
        G = Ainv[0, 0] * Q_PKP + Ainv[0, 1] * Q_P
        G = 0.5 * (G + G.T)
        d = torch.sqrt(torch.diagonal(Q_P).clamp_min(1e-12))
        P_corr = Q_P / (d[:, None] * d[None, :])
        P_corr = 0.5 * (P_corr + P_corr.T)

        zdim = G.shape[0]
        P_reg = P_corr + ridge * torch.eye(zdim, device=device, dtype=X.dtype)
        L = torch.linalg.cholesky(P_reg)
        Linv = torch.linalg.inv(L)
        M = Linv @ G @ Linv.T
        M = 0.5 * (M + M.T)
        spectrum = torch.linalg.eigvalsh(M).flip(0)       # descending
        return G, P_corr, spectrum

    return loss


# ---------------------------------------------------------------------------
# Spatial continuity loss
# ---------------------------------------------------------------------------

def spat_cont(
    X: Tensor,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Spatial continuity loss via 1-shift differential operator.

    Computes the L1 norm of horizontal and vertical first differences.

    Args:
        X: Image tensor of shape (n, c, h, w).
        device: Torch device.

    Returns:
        Scalar loss value.
    """
    l1 = L1Loss()
    channels = X.shape[1]

    h_targets = torch.zeros_like(X, device=device)
    v_targets = torch.zeros_like(X, device=device)
    for i in range(channels):
        h_targets[:, i, :, :] = pad(X[:, i, :-1, :], (0, 0, 0, 1), "constant", 0)
        v_targets[:, i, :, :] = pad(X[:, i, :, :-1], (0, 1), "constant", 0)

    return l1(X, h_targets) + l1(X, v_targets)
