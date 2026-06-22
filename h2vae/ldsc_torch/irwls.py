"""Differentiable iteratively re-weighted least squares (IRWLS).

Ported from ldsc/ldscore/irwls.py (lines 56-195).

The IRWLS weight-update iterations are detached from the autograd graph.
Only the final weighted solve is differentiable.
"""

import torch


def _wls_coef(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted least squares via normal equations.

    Used inside the IRWLS weight-update loop (detached).

    Parameters
    ----------
    x : (n, p) design matrix.
    y : (n, 1) response.
    w : (n, 1) weights on sqrt(1/variance) scale.

    Returns
    -------
    coef : (p, 1) regression coefficients.
    """
    w_norm = w / w.sum()
    x_w = x * w_norm
    y_w = y * w_norm
    xtx = x_w.T @ x_w
    xty = x_w.T @ y_w
    return torch.linalg.solve(xtx, xty)


def irwls_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    update_func,
    initial_w: torch.Tensor,
    n_iter: int = 2,
) -> torch.Tensor:
    """Run IRWLS with detached weight updates, then a differentiable final solve.

    Parameters
    ----------
    x : (n, p) design matrix (may include intercept column).
    y : (n, 1) response vector.
    update_func : callable
        Takes (p, 1) coefficient tensor, returns (n, 1) raw weight tensor.
        Internally should use .item() to break the graph.
    initial_w : (n, 1) initial regression weights (on 1/variance scale).
    n_iter : Number of IRWLS weight-update iterations (default 2).

    Returns
    -------
    est : (1, p) point estimate tensor with grad_fn attached.
    """
    # --- Weight iterations (detached) ---
    w = torch.sqrt(initial_w)
    for _ in range(n_iter):
        coef = _wls_coef(x, y, w)
        new_w = torch.sqrt(update_func(coef))
        w = new_w
    w_final = w.detach()

    # --- Final differentiable solve ---
    w_norm = w_final / w_final.sum()
    x_w = x * w_norm
    y_w = y * w_norm
    xtx = x_w.T @ x_w
    xty = x_w.T @ y_w
    est = torch.linalg.solve(xtx, xty)  # (p, 1)
    return est.T  # (1, p)
