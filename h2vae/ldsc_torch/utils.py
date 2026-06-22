"""Utility functions ported from ldsc/ldscore/regressions.py."""

import math

import torch
from scipy.stats import norm


def append_intercept(x: torch.Tensor) -> torch.Tensor:
    """Append a column of ones to the design matrix."""
    n_row = x.shape[0]
    intercept = torch.ones(n_row, 1, dtype=x.dtype, device=x.device)
    return torch.cat((x, intercept), dim=1)


def remove_intercept(x: torch.Tensor) -> torch.Tensor:
    """Remove the last column."""
    return x[:, :-1]


def aggregate(y, x, N, M, intercept=None):
    """Compute initial aggregate h2 or gencov estimate.

    Port of LD_Score_Regression.aggregate (regressions.py:238-244).
    All inputs are torch tensors. Returns a Python float.
    """
    if intercept is None:
        intercept = 1.0
    num = float(M) * (y.mean().item() - intercept)
    denom = (x * N).mean().item()
    return num / denom


def h2_obs_to_liab(h2_obs, P, K):
    """Convert observed-scale h2 to liability-scale h2.

    Port of regressions.py:107-137.
    """
    if P is None or K is None:
        return h2_obs
    if math.isnan(P) and math.isnan(K):
        return h2_obs
    if K <= 0 or K >= 1:
        raise ValueError("K must be in the range (0,1)")
    if P <= 0 or P >= 1:
        raise ValueError("P must be in the range (0,1)")

    thresh = norm.isf(K)
    conversion_factor = K**2 * (1 - K) ** 2 / (P * (1 - P) * norm.pdf(thresh) ** 2)
    return h2_obs * conversion_factor


def gencov_obs_to_liab(gencov_obs, P1, P2, K1, K2):
    """Convert observed-scale genetic covariance to liability-scale.

    Port of regressions.py:75-104.
    """
    c1 = 1.0
    c2 = 1.0
    if P1 is not None and K1 is not None:
        c1 = math.sqrt(h2_obs_to_liab(1, P1, K1))
    if P2 is not None and K2 is not None:
        c2 = math.sqrt(h2_obs_to_liab(1, P2, K2))
    return gencov_obs * c1 * c2
