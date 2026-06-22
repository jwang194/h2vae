"""Shared fixtures for ldsc_torch tests."""

import numpy as np
import pytest


@pytest.fixture
def hsq_data_1d_constrained():
    """Simple 1-annotation data (4 SNPs) with constrained intercept=1.

    Matches Test_Hsq_1D from ldsc/test/test_regressions.py.
    """
    chisq = np.ones((4, 1)) * 4
    ld = np.ones((4, 1))
    w_ld = np.ones((4, 1))
    N = 9 * np.ones((4, 1))
    M = np.array([[7.0]])
    return chisq, ld, w_ld, N, M


@pytest.fixture
def hsq_data_2annot(rng):
    """2-annotation noiseless data with known h2 values.

    Matches Test_Coef from ldsc/test/test_regressions.py.
    """
    hsq1, hsq2 = 0.2, 0.7
    ld = (np.abs(rng.normal(size=800)) + 1).reshape((400, 2))
    N = np.ones((400, 1)) * 1e5
    M = np.ones((1, 2)) * 1e7 / 2.0
    chisq = 1 + 1e5 * (
        ld[:, 0] * hsq1 / M[0, 0] + ld[:, 1] * hsq2 / M[0, 1]
    ).reshape((400, 1))
    w_ld = np.ones_like(chisq)
    return chisq, ld, w_ld, N, M, hsq1, hsq2


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def gencov_data_1d():
    """Simple 1-annotation Gencov data with constrained intercept.

    Matches Test_Gencov_1D from ldsc/test/test_regressions.py.
    """
    z1 = np.ones((4, 1)) * 4
    z2 = np.ones((4, 1))
    ld = np.ones((4, 1))
    w_ld = np.ones((4, 1))
    N1 = 9 * np.ones((4, 1))
    N2 = 7 * np.ones((4, 1))
    M = np.array([[7.0]])
    hsq1, hsq2 = 0.5, 0.6
    return z1, z2, ld, w_ld, N1, N2, M, hsq1, hsq2
