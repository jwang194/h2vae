"""ldsc_torch: Differentiable LD Score Regression in PyTorch."""

from .regressions import Hsq, Gencov, RG


def h2_loss(chisq, ref_ld, w_ld, N, M, intercept=1.0):
    """Compute total h2 as a differentiable scalar.

    Convenience wrapper around Hsq for use as a loss function.
    All inputs should be torch.Tensors with appropriate requires_grad.
    """
    return Hsq(chisq, ref_ld, w_ld, N, M, intercept=intercept).tot


def rg_loss(z1, z2, ref_ld, w_ld, N1, N2, M):
    """Compute genetic correlation as a differentiable scalar.

    Convenience wrapper around RG for use as a loss function.
    """
    return RG(z1, z2, ref_ld, w_ld, N1, N2, M).rg_ratio
