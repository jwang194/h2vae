"""Test weight functions against original LDSC implementation."""

import numpy as np
import torch
import pytest
import sys, os

# Add original ldsc to path for comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'translate_pkg', 'ldsc'))
import ldscore.regressions as orig_reg

from h2vae.ldsc_torch.weights import hsq_weights, gencov_weights


class TestHsqWeights:

    def test_basic(self):
        """Port of Test_Hsq_1D.test_weights (test_regressions.py:89-99)."""
        ld = torch.ones(4, 1, dtype=torch.float64)
        w_ld = torch.ones(4, 1, dtype=torch.float64)
        N = 9 * torch.ones(4, 1, dtype=torch.float64)
        M = 7.0
        hsq = 0.5

        w = hsq_weights(ld, w_ld, N, M, hsq)
        expected = 0.5 / np.square(1 + hsq * 9 / 7)
        np.testing.assert_allclose(w[0, 0].item(), expected, rtol=1e-10)

    def test_clamp_hsq(self):
        """h2 > 1 and h2 < 0 should give same weights as h2=1 and h2=0."""
        ld = torch.ones(4, 1, dtype=torch.float64)
        w_ld = torch.ones(4, 1, dtype=torch.float64)
        N = 9 * torch.ones(4, 1, dtype=torch.float64)
        M = 7.0

        w_high = hsq_weights(ld, w_ld, N, M, 2.0)
        w_one = hsq_weights(ld, w_ld, N, M, 1.0)
        np.testing.assert_allclose(w_high.numpy(), w_one.numpy(), rtol=1e-10)

        w_neg = hsq_weights(ld, w_ld, N, M, -1.0)
        w_zero = hsq_weights(ld, w_ld, N, M, 0.0)
        np.testing.assert_allclose(w_neg.numpy(), w_zero.numpy(), rtol=1e-10)

    def test_matches_original(self):
        """Numerical match against original numpy implementation."""
        rng = np.random.RandomState(123)
        ld_np = np.abs(rng.normal(size=100)).reshape((100, 1)) + 1
        w_ld_np = np.abs(rng.normal(size=100)).reshape((100, 1)) + 1
        N_np = rng.uniform(1e4, 1e5, size=(100, 1))
        M = 1e7
        hsq = 0.3

        orig = orig_reg.Hsq.weights(ld_np, w_ld_np, N_np, M, hsq)
        ours = hsq_weights(
            torch.from_numpy(ld_np), torch.from_numpy(w_ld_np),
            torch.from_numpy(N_np), M, hsq,
        )
        np.testing.assert_allclose(ours.numpy(), orig, rtol=1e-10)


class TestGencovWeights:

    def test_equals_hsq_when_symmetric(self):
        """Port of Test_Gencov_1D.test_weights (test_regressions.py:223-234).

        When N1=N2, h1=h2, rho_g=h, intercept_gencov=1, the Gencov weights
        should equal the Hsq weights.
        """
        rng = np.random.RandomState(456)
        ld = torch.from_numpy(np.abs(rng.normal(size=100)).reshape((100, 1)))
        w_ld = torch.from_numpy(np.abs(rng.normal(size=100)).reshape((100, 1)))
        N1 = torch.from_numpy(np.abs(rng.normal(size=100)).reshape((100, 1)))
        N2 = N1.clone()
        M = 10.0
        h = 0.5

        wg = gencov_weights(ld, w_ld, N1, N2, M, h, h, h, intercept_gencov=1.0)
        wh = hsq_weights(ld, w_ld, N1, M, h, intercept=1.0)
        np.testing.assert_allclose(wg.numpy(), wh.numpy(), rtol=1e-10)

    def test_matches_original(self):
        """Numerical match against original numpy implementation."""
        rng = np.random.RandomState(789)
        ld_np = np.abs(rng.normal(size=50)).reshape((50, 1)) + 1
        w_ld_np = np.abs(rng.normal(size=50)).reshape((50, 1)) + 1
        N1_np = rng.uniform(1e4, 1e5, size=(50, 1))
        N2_np = rng.uniform(1e4, 1e5, size=(50, 1))
        M = 1e6
        h1, h2, rho_g = 0.3, 0.5, 0.2

        orig = orig_reg.Gencov.weights(
            ld_np, w_ld_np, N1_np, N2_np, M, h1, h2, rho_g,
            intercept_gencov=0.0, intercept_hsq1=1.0, intercept_hsq2=1.0,
        )
        ours = gencov_weights(
            torch.from_numpy(ld_np), torch.from_numpy(w_ld_np),
            torch.from_numpy(N1_np), torch.from_numpy(N2_np),
            M, h1, h2, rho_g,
        )
        np.testing.assert_allclose(ours.numpy(), orig, rtol=1e-10)
