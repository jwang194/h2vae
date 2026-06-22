"""Test Hsq, Gencov, RG against original LDSC implementation."""

import numpy as np
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'translate_pkg', 'ldsc'))
import ldscore.regressions as orig_reg

import h2vae.ldsc_torch as ldsc_torch


class TestHsq1D:
    """Port of Test_Hsq_1D from ldsc/test/test_regressions.py."""

    def test_constrained_intercept(self, hsq_data_1d_constrained):
        chisq, ld, w_ld, N, M = hsq_data_1d_constrained
        result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)
        orig = orig_reg.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)
        np.testing.assert_allclose(result.tot, orig.tot, rtol=1e-6)

    def test_aggregate(self):
        """Port of Test_Hsq_1D.test_aggregate."""
        chisq = np.ones((10, 1)) * 3 / 2
        ld = np.ones((10, 1)) * 100
        N = np.ones((10, 1)) * 100000
        M = np.array([[1e7]])

        result = ldsc_torch.Hsq(chisq, ld, N, N, M, intercept=1)
        # With intercept=1, chisq=1.5: agg = M*(1.5-1)/(100*1e5) = 1e7*0.5/1e7 = 0.5
        # Just test the aggregate function directly
        from h2vae.ldsc_torch.utils import aggregate
        agg = aggregate(
            torch.tensor(chisq), torch.tensor(ld), torch.tensor(N), 1e7, 1.0
        )
        np.testing.assert_allclose(agg, 0.5, rtol=1e-6)


class TestHsqCoef:
    """Port of Test_Coef from ldsc/test/test_regressions.py.

    Noiseless 2-annotation data with known h2 = (0.2, 0.7).
    """

    def test_coef_constrained(self, hsq_data_2annot):
        chisq, ld, w_ld, N, M, hsq1, hsq2 = hsq_data_2annot
        result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)
        orig = orig_reg.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)

        np.testing.assert_allclose(result.coef, orig.coef, rtol=1e-6)
        expected_coef = [hsq1 / M[0, 0], hsq2 / M[0, 1]]
        np.testing.assert_allclose(result.coef, expected_coef, rtol=1e-4)

    def test_cat_constrained(self, hsq_data_2annot):
        chisq, ld, w_ld, N, M, hsq1, hsq2 = hsq_data_2annot
        result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)
        np.testing.assert_allclose(result.cat, [hsq1, hsq2], rtol=1e-4)

    def test_tot_constrained(self, hsq_data_2annot):
        chisq, ld, w_ld, N, M, hsq1, hsq2 = hsq_data_2annot
        result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=1)
        np.testing.assert_allclose(result.tot, hsq1 + hsq2, rtol=1e-4)

    def test_free_intercept(self, hsq_data_2annot):
        chisq, ld, w_ld, N, M, hsq1, hsq2 = hsq_data_2annot
        result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3)
        orig = orig_reg.Hsq(chisq, ld, w_ld, N, M, n_blocks=3)

        np.testing.assert_allclose(result.tot, orig.tot, rtol=1e-6)
        np.testing.assert_allclose(float(result.intercept), float(orig.intercept), rtol=1e-4)

    def test_tot_matches_original(self, hsq_data_2annot):
        """Exact numerical match with original on .tot for both constrained and free."""
        chisq, ld, w_ld, N, M, hsq1, hsq2 = hsq_data_2annot

        for intercept in [1, None]:
            result = ldsc_torch.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=intercept)
            orig = orig_reg.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, intercept=intercept)
            np.testing.assert_allclose(
                float(result.tot), float(orig.tot), rtol=1e-6,
                err_msg=f"Mismatch for intercept={intercept}"
            )


class TestGencov1D:
    """Port of Test_Gencov_1D from ldsc/test/test_regressions.py."""

    def test_constrained(self, gencov_data_1d):
        z1, z2, ld, w_ld, N1, N2, M, hsq1, hsq2 = gencov_data_1d
        result = ldsc_torch.Gencov(
            z1, z2, ld, w_ld, N1, N2, M, hsq1, hsq2, 1.0, 1.0,
            n_blocks=3, intercept_gencov=1,
        )
        orig = orig_reg.Gencov(
            z1, z2, ld, w_ld, N1, N2, M, hsq1, hsq2, 1.0, 1.0,
            n_blocks=3, intercept_gencov=1,
        )
        np.testing.assert_allclose(float(result.tot), float(orig.tot), rtol=1e-6)

    def test_aggregate(self, gencov_data_1d):
        """Port of Test_Gencov_1D.test_aggregate."""
        from h2vae.ldsc_torch.utils import aggregate
        z1z2 = torch.ones(10, 1, dtype=torch.float64) / 2
        ld = torch.ones(10, 1, dtype=torch.float64) * 100
        N = torch.ones(10, 1, dtype=torch.float64) * 100000
        M = 1e7

        agg = aggregate(z1z2, ld, N, M, 0.0)
        np.testing.assert_allclose(agg, 0.5, rtol=1e-6)

        agg = aggregate(z1z2, ld, N, M, 0.5)
        np.testing.assert_allclose(agg, 0.0, atol=1e-10)


class TestRG:
    """Port of Test_RG_2D from ldsc/test/test_regressions.py."""

    def test_rg_negative_correlation(self):
        """z1 and -z1 should give rg ≈ -1."""
        rng = np.random.RandomState(99)
        ld = np.abs(rng.normal(size=100).reshape((50, 2))) + 2
        z1 = (np.sum(ld, axis=1) * 10).reshape((50, 1))
        w_ld = rng.normal(size=50).reshape((50, 1))
        N1 = 9 * np.ones((50, 1))
        M = np.array([[700.0, 222.0]])

        result = ldsc_torch.RG(
            z1, -z1, ld, w_ld, N1, N1, M,
            intercept_hsq1=1.0, intercept_hsq2=1.0,
            intercept_gencov=0, n_blocks=20,
        )
        assert abs(float(result.rg_ratio) + 1) < 0.01

    @pytest.mark.xfail(
        raises=TypeError, strict=False,
        reason="vendored upstream ldscore.regressions.RG crashes under numpy>=2 "
               "(`float(rg.jknife_est)` on a non-0d array); our ldsc_torch.RG is "
               "validated by test_rg_negative_correlation. Recovers if upstream is fixed.",
    )
    def test_matches_original(self):
        """Numerical match with original RG."""
        rng = np.random.RandomState(99)
        ld = np.abs(rng.normal(size=100).reshape((50, 2))) + 2
        z1 = (np.sum(ld, axis=1) * 10).reshape((50, 1))
        w_ld = rng.normal(size=50).reshape((50, 1))
        N1 = 9 * np.ones((50, 1))
        M = np.array([[700.0, 222.0]])

        result = ldsc_torch.RG(
            z1, -z1, ld, w_ld, N1, N1, M,
            1.0, 1.0, 0, n_blocks=20,
        )
        orig = orig_reg.RG(
            z1, -z1, ld, w_ld, N1, N1, M,
            1.0, 1.0, 0, n_blocks=20,
        )
        np.testing.assert_allclose(
            float(result.rg_ratio), float(orig.rg_ratio), rtol=1e-4,
        )
