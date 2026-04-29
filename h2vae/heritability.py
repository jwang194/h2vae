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
    phenotype) or ``(n, d)`` (batched), returning a vector of ``d``
    per-column genetic correlations in a single matmul.

    Args:
        X: Genetic data — genotype matrix (n, m) or kinship matrix (n, n).
        y2: Reference phenotype vector, shape (n, 1) or (n,).
        kinship: If True, treat X as a precomputed GRM (n, n).
        C: Optional covariate matrix of shape (n, c_user) *without* an
            intercept; an intercept column is prepended automatically (so
            y1 and y2 are implicitly mean-centred).
        device: Torch device for computation.

    Returns:
        A callable ``loss(y1) -> Tensor``.
    """
    n = X.shape[0]
    K = X if kinship else (X @ X.T) / X.shape[1]
    K = K.to(device)
    y2 = y2.to(device)
    if y2.ndim == 1:
        y2 = y2[:, None]

    if C is None:
        W = torch.ones((n, 1), device=device)
    else:
        W = torch.hstack((torch.ones((n, 1), device=device), C.to(device)))
    c = W.shape[1]
    nc = n - c

    WtW_I = torch.inverse(W.T @ W)
    V_of = lambda x: x - W @ (WtW_I @ (W.T @ x))

    # tr(V K V) without materialising V as (n, n):
    #   tr(VKV) = tr(VKV) = tr(KV²) = tr(KV)
    #          = tr(K) - tr(W(W'W)^{-1} W' K)
    #          = tr(K) - tr((W'W)^{-1} W'KW)    (cyclic)
    WtKW = W.T @ (K @ W)                                # (c, c)
    tr_Ktil = torch.trace(K) - torch.trace(WtW_I @ WtKW)

    # y2-dependent quantities (precompute once)
    V_y2 = V_of(y2)                                     # (n, 1)
    KV_y2 = K @ V_y2                                    # (n, 1)
    VKV_y2 = V_of(KV_y2)                                # (n, 1)
    d2_raw = (y2 * VKV_y2).sum() * nc - (y2 * V_y2).sum() * tr_Ktil
    # MoM variance estimates can be negative when the true variance is small
    # relative to noise; floor at 1e-8 (so sqrt(d1*d2) is real & non-NaN, and
    # the clamp blocks the denominator's gradient in that regime).  Sign of
    # the gc estimate is carried entirely by the numerator (genetic covariance).
    d2 = torch.clamp(d2_raw, min=1e-8)

    def loss(y1: Tensor) -> Tensor:
        # y1 shape (n, 1) or (n, d) — treated uniformly.
        num = (y1 * VKV_y2).sum(dim=0) * nc - (y1 * V_y2).sum(dim=0) * tr_Ktil
        V_y1 = V_of(y1)
        KV_y1 = K @ V_y1
        VKV_y1 = V_of(KV_y1)
        d1_raw = (y1 * VKV_y1).sum(dim=0) * nc - (y1 * V_y1).sum(dim=0) * tr_Ktil
        d1 = torch.clamp(d1_raw, min=1e-8)  # see comment on d2 above
        return num / torch.sqrt(d1 * d2)

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
