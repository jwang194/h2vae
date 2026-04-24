"""Latent space utility functions for centering, residualization, and correlation."""

from __future__ import annotations

import torch
from torch import Tensor


def center_and_scale(Z: Tensor, eps: float = 1e-8) -> Tensor:
    """Standardize each column of Z to zero mean and unit variance."""
    return (Z - Z.mean(dim=0, keepdim=True)) / (Z.std(dim=0, keepdim=True) + eps)


def residualize(Z: Tensor, P: Tensor | None) -> Tensor:
    """Project Z through projection matrix P. No-op if P is None."""
    if P is None:
        return Z
    return P @ Z


def corrcoef(X: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute the correlation matrix of the rows of X.

    Args:
        X: Tensor of shape (D, N) where D is the number of variables
           and N is the number of observations.
        eps: Small constant for numerical stability.

    Returns:
        Correlation matrix of shape (D, D).
    """
    D = X.shape[-1]
    mean = X.mean(dim=-1, keepdim=True)
    std = X.std(dim=-1, keepdim=True)
    X = (X - mean) / (std + eps)
    return (1 / (D - 1)) * X @ X.transpose(-1, -2)
