"""Differentiable LD Score Regression estimators.

Ported from ldsc/ldscore/regressions.py.
Classes: Hsq (heritability), Gencov (genetic covariance), RG (genetic correlation).

Point estimates (.tot, .coef, .cat, .intercept) are differentiable torch tensors
when torch.Tensor inputs are provided. When numpy inputs are given, results are
converted back to numpy.
"""

import torch
import numpy as np

from ._convert import InputContext, to_tensor, maybe_to_numpy
from .utils import append_intercept, aggregate
from .weights import hsq_weights, gencov_weights
from .irwls import irwls_solve


class Hsq:
    """Heritability estimation via LD Score Regression.

    Port of regressions.py Hsq class (lines 336-535).

    Parameters
    ----------
    y : (n_snp, 1) chi-squared statistics.
    x : (n_snp, n_annot) LD scores.
    w : (n_snp, 1) regression-weight LD scores.
    N : (n_snp, 1) per-SNP sample sizes.
    M : (1, n_annot) number of SNPs per annotation.
    n_blocks : unused (kept for API compatibility with original).
    intercept : float or None. None = free intercept; float = constrained.

    Attributes
    ----------
    tot : total h2 (differentiable).
    coef : (n_annot,) regression coefficients (differentiable).
    cat : (1, n_annot) per-category h2 (differentiable).
    intercept : intercept estimate (differentiable if free, else the constrained value).
    mean_chisq : mean chi-squared statistic.
    lambda_gc : genomic control lambda.
    """

    __null_intercept__ = 1.0

    def __init__(self, y, x, w, N, M, n_blocks=200, intercept=None):
        ctx = InputContext.from_inputs(y, x, w, N, M)
        y = to_tensor(y, ctx)
        x = to_tensor(x, ctx)
        w = to_tensor(w, ctx)
        N = to_tensor(N, ctx)
        M = to_tensor(M, ctx)

        n_snp, n_annot = x.shape
        self.n_annot = n_annot
        self.constrain_intercept = intercept is not None

        # Summary statistics (non-differentiable)
        self.mean_chisq = float(y.mean().item())
        self.lambda_gc = float(np.median(y.detach().cpu().numpy()) / 0.4549)

        # Preprocessing
        M_tot = float(M.sum().item())
        x_tot = x.sum(dim=1, keepdim=True)  # (n_snp, 1)
        Nbar = N.mean()  # tensor scalar

        # Initial aggregate estimate and weights
        null_int = intercept if intercept is not None else self.__null_intercept__
        tot_agg = aggregate(y, x_tot, N, M_tot, null_int)
        initial_w = hsq_weights(x_tot, w, N, M_tot, tot_agg, null_int)

        # Scale design matrix by N/Nbar (keeps condition number low)
        x_scaled = x * N / Nbar

        if not self.constrain_intercept:
            x_design = append_intercept(x_scaled)
            x_tot_design = append_intercept(x_tot)
            yp = y
        else:
            x_design = x_scaled
            x_tot_design = x_tot
            yp = y - intercept

        Nbar_val = Nbar.item()

        # Define the IRWLS weight-update function.
        # This closure extracts scalar estimates via .item() (breaking the graph).
        def update_func(coef):
            # coef: (p, 1) tensor from the WLS solve
            hsq_est = M_tot * coef[0, 0].item() / Nbar_val
            if not self.constrain_intercept:
                int_est = coef[-1, 0].item()
            else:
                int_est = intercept
            ld = x_tot_design[:, 0:1]  # (n_snp, 1), strip intercept col if present
            return hsq_weights(ld, w, N, M_tot, hsq_est, int_est)

        # Run IRWLS → differentiable point estimate
        est = irwls_solve(x_design, yp, update_func, initial_w)  # (1, p)

        # Extract results
        self.coef = est[0, :n_annot] / Nbar  # (n_annot,) tensor
        self.cat = M[0] * self.coef  # (n_annot,) per-category h2
        self.tot = self.cat.sum()  # scalar tensor

        if not self.constrain_intercept:
            self.intercept = est[0, n_annot]  # scalar tensor
        else:
            self.intercept = torch.tensor(intercept, dtype=y.dtype, device=y.device)

        # Convert back to numpy if inputs were numpy
        self.tot = maybe_to_numpy(self.tot, ctx)
        self.coef = maybe_to_numpy(self.coef, ctx)
        self.cat = maybe_to_numpy(self.cat, ctx)
        self.intercept = maybe_to_numpy(self.intercept, ctx)
        self.M = M


class Gencov:
    """Genetic covariance estimation via LD Score Regression.

    Port of regressions.py Gencov class (lines 538-677).

    Parameters
    ----------
    z1, z2 : (n_snp, 1) Z-scores for each trait.
    x : (n_snp, n_annot) LD scores.
    w : (n_snp, 1) regression-weight LD scores.
    N1, N2 : (n_snp, 1) per-SNP sample sizes for each study.
    M : (1, n_annot) number of SNPs per annotation.
    hsq1, hsq2 : float, heritability estimates for each trait (from prior Hsq runs).
    intercept_hsq1, intercept_hsq2 : float, Hsq intercepts for each trait.
    n_blocks : unused (kept for API compatibility).
    intercept_gencov : float or None. None = free intercept; float = constrained.

    Attributes
    ----------
    tot : total genetic covariance (differentiable).
    coef : (n_annot,) regression coefficients (differentiable).
    cat : (1, n_annot) per-category gencov (differentiable).
    intercept : intercept estimate.
    """

    __null_intercept__ = 0.0

    def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2,
                 intercept_hsq1, intercept_hsq2,
                 n_blocks=200, intercept_gencov=None):
        ctx = InputContext.from_inputs(z1, z2, x, w, N1, N2, M)
        z1 = to_tensor(z1, ctx)
        z2 = to_tensor(z2, ctx)
        x = to_tensor(x, ctx)
        w = to_tensor(w, ctx)
        N1 = to_tensor(N1, ctx)
        N2 = to_tensor(N2, ctx)
        M = to_tensor(M, ctx)

        n_snp, n_annot = x.shape
        self.n_annot = n_annot
        self.constrain_intercept = intercept_gencov is not None

        # Response: product of Z-scores (differentiable)
        y = z1 * z2
        # Effective sample size
        N = torch.sqrt(N1 * N2)

        # Store for weight updates
        self._N1 = N1
        self._N2 = N2
        self._hsq1 = float(hsq1)
        self._hsq2 = float(hsq2)
        self._intercept_hsq1 = float(intercept_hsq1)
        self._intercept_hsq2 = float(intercept_hsq2)

        # Preprocessing (mirrors LD_Score_Regression.__init__)
        M_tot = float(M.sum().item())
        x_tot = x.sum(dim=1, keepdim=True)
        Nbar = N.mean()

        null_int = intercept_gencov if intercept_gencov is not None else self.__null_intercept__
        tot_agg = aggregate(y, x_tot, N, M_tot, null_int)

        # Initial weights use gencov formula
        initial_w = gencov_weights(
            x_tot, w, N1, N2, M_tot,
            self._hsq1, self._hsq2, tot_agg,
            intercept_gencov=null_int,
            intercept_hsq1=self._intercept_hsq1,
            intercept_hsq2=self._intercept_hsq2,
        )

        x_scaled = x * N / Nbar

        if not self.constrain_intercept:
            x_design = append_intercept(x_scaled)
            x_tot_design = append_intercept(x_tot)
            yp = y
        else:
            x_design = x_scaled
            x_tot_design = x_tot
            yp = y - intercept_gencov

        Nbar_val = Nbar.item()

        def update_func(coef):
            rho_g = M_tot * coef[0, 0].item() / Nbar_val
            if not self.constrain_intercept:
                int_est = coef[-1, 0].item()
            else:
                int_est = intercept_gencov
            ld = x_tot_design[:, 0:1]
            return gencov_weights(
                ld, w, N1, N2, M_tot,
                self._hsq1, self._hsq2, rho_g,
                intercept_gencov=int_est,
                intercept_hsq1=self._intercept_hsq1,
                intercept_hsq2=self._intercept_hsq2,
            )

        est = irwls_solve(x_design, yp, update_func, initial_w)

        self.coef = est[0, :n_annot] / Nbar
        self.cat = M[0] * self.coef
        self.tot = self.cat.sum()

        if not self.constrain_intercept:
            self.intercept = est[0, n_annot]
        else:
            self.intercept = torch.tensor(intercept_gencov, dtype=y.dtype, device=y.device)

        self.mean_z1z2 = float((z1 * z2).mean().item())

        self.tot = maybe_to_numpy(self.tot, ctx)
        self.coef = maybe_to_numpy(self.coef, ctx)
        self.cat = maybe_to_numpy(self.cat, ctx)
        self.intercept = maybe_to_numpy(self.intercept, ctx)
        self.M = M


class RG:
    """Genetic correlation estimation.

    Port of regressions.py RG class (lines 680-745).
    Composes two Hsq + one Gencov and computes rg = gencov / sqrt(h2_1 * h2_2).

    Parameters
    ----------
    z1, z2 : (n_snp, 1) Z-scores for each trait.
    x : (n_snp, n_annot) LD scores.
    w : (n_snp, 1) regression-weight LD scores.
    N1, N2 : (n_snp, 1) per-SNP sample sizes for each study.
    M : (1, n_annot) number of SNPs per annotation.
    intercept_hsq1, intercept_hsq2 : float or None, Hsq intercept constraints.
    intercept_gencov : float or None, Gencov intercept constraint.
    n_blocks : unused (kept for API compatibility).

    Attributes
    ----------
    rg_ratio : genetic correlation estimate (differentiable).
    hsq1, hsq2 : Hsq objects for each trait.
    gencov : Gencov object.
    """

    def __init__(self, z1, z2, x, w, N1, N2, M,
                 intercept_hsq1=None, intercept_hsq2=None,
                 intercept_gencov=None, n_blocks=200):
        ctx = InputContext.from_inputs(z1, z2, x, w, N1, N2, M)
        z1_t = to_tensor(z1, ctx)
        z2_t = to_tensor(z2, ctx)
        x_t = to_tensor(x, ctx)
        w_t = to_tensor(w, ctx)
        N1_t = to_tensor(N1, ctx)
        N2_t = to_tensor(N2, ctx)
        M_t = to_tensor(M, ctx)

        # Marginal heritabilities (pass tensors to keep grad for rg_ratio)
        self.hsq1 = Hsq(z1_t.square(), x_t, w_t, N1_t, M_t,
                        n_blocks=n_blocks, intercept=intercept_hsq1)
        self.hsq2 = Hsq(z2_t.square(), x_t, w_t, N2_t, M_t,
                        n_blocks=n_blocks, intercept=intercept_hsq2)

        # Extract h2 and intercept as floats for Gencov weight computation
        h1_val = float(self.hsq1.tot.item() if isinstance(self.hsq1.tot, torch.Tensor)
                       else self.hsq1.tot)
        h2_val = float(self.hsq2.tot.item() if isinstance(self.hsq2.tot, torch.Tensor)
                       else self.hsq2.tot)
        int1 = float(self.hsq1.intercept.item() if isinstance(self.hsq1.intercept, torch.Tensor)
                     else self.hsq1.intercept)
        int2 = float(self.hsq2.intercept.item() if isinstance(self.hsq2.intercept, torch.Tensor)
                     else self.hsq2.intercept)

        self.gencov = Gencov(
            z1_t, z2_t, x_t, w_t, N1_t, N2_t, M_t,
            h1_val, h2_val, int1, int2,
            n_blocks=n_blocks, intercept_gencov=intercept_gencov,
        )

        # Genetic correlation (differentiable ratio)
        self._negative_hsq = False
        h1_tot = self.hsq1.tot if isinstance(self.hsq1.tot, torch.Tensor) else torch.tensor(self.hsq1.tot)
        h2_tot = self.hsq2.tot if isinstance(self.hsq2.tot, torch.Tensor) else torch.tensor(self.hsq2.tot)
        gc_tot = self.gencov.tot if isinstance(self.gencov.tot, torch.Tensor) else torch.tensor(self.gencov.tot)

        if float(h1_tot.item()) <= 0 or float(h2_tot.item()) <= 0:
            self._negative_hsq = True
            self.rg_ratio = float("nan")
        else:
            self.rg_ratio = gc_tot / torch.sqrt(h1_tot * h2_tot)
            self.rg_ratio = maybe_to_numpy(self.rg_ratio, ctx)
