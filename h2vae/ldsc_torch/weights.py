"""Regression weight functions for Hsq and Gencov.

Ported from ldsc/ldscore/regressions.py (Hsq.weights lines 497-535,
Gencov.weights lines 621-677).

Weight functions take Python float scalars for statistical estimates
(hsq, intercept, etc.) because these are always detached from the
autograd graph. The LD score and sample size tensors pass through
unchanged (they are constants in the loss function use case).
"""

import torch


def hsq_weights(
    ld: torch.Tensor,
    w_ld: torch.Tensor,
    N: torch.Tensor,
    M: float,
    hsq: float,
    intercept: float = 1.0,
) -> torch.Tensor:
    """Compute Hsq regression weights.

    Parameters
    ----------
    ld : (n_snp, 1) LD scores (non-partitioned total).
    w_ld : (n_snp, 1) Regression-weight LD scores.
    N : (n_snp, 1) Per-SNP sample sizes.
    M : Total number of SNPs used for LD score estimation.
    hsq : Current heritability estimate (scalar, detached).
    intercept : Current intercept estimate (scalar, detached).

    Returns
    -------
    w : (n_snp, 1) Regression weights.
    """
    M = float(M)
    hsq = max(min(hsq, 1.0), 0.0)
    ld = torch.clamp(ld, min=1.0)
    w_ld = torch.clamp(w_ld, min=1.0)
    c = hsq * N / M
    het_w = 1.0 / (2.0 * (intercept + c * ld).square())
    oc_w = 1.0 / w_ld
    return het_w * oc_w


def gencov_weights(
    ld: torch.Tensor,
    w_ld: torch.Tensor,
    N1: torch.Tensor,
    N2: torch.Tensor,
    M: float,
    h1: float,
    h2: float,
    rho_g: float,
    intercept_gencov: float = 0.0,
    intercept_hsq1: float = 1.0,
    intercept_hsq2: float = 1.0,
) -> torch.Tensor:
    """Compute Gencov regression weights.

    Parameters
    ----------
    ld : (n_snp, 1) LD scores (non-partitioned total).
    w_ld : (n_snp, 1) Regression-weight LD scores.
    N1, N2 : (n_snp, 1) Per-SNP sample sizes for each study.
    M : Total number of SNPs used for LD score estimation.
    h1, h2 : Heritability estimates for each study (scalars, detached).
    rho_g : Genetic covariance estimate (scalar, detached).
    intercept_gencov : Gencov intercept on z1*z2 scale.
    intercept_hsq1, intercept_hsq2 : Hsq intercepts for each study.

    Returns
    -------
    w : (n_snp, 1) Regression weights.
    """
    M = float(M)
    h1 = max(min(h1, 1.0), 0.0)
    h2 = max(min(h2, 1.0), 0.0)
    rho_g = max(min(rho_g, 1.0), -1.0)
    ld = torch.clamp(ld, min=1.0)
    w_ld = torch.clamp(w_ld, min=1.0)

    a = N1 * (h1 * ld) / M + intercept_hsq1
    b = N2 * (h2 * ld) / M + intercept_hsq2
    sqrt_n1n2 = torch.sqrt(N1 * N2)
    c = sqrt_n1n2 * (rho_g * ld) / M + intercept_gencov

    het_w = 1.0 / (a * b + c.square())
    oc_w = 1.0 / w_ld
    return het_w * oc_w
