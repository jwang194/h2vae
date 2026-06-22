"""Test that the forward pass is differentiable via PyTorch autograd."""

import numpy as np
import torch
import pytest

from h2vae.ldsc_torch import h2_loss, rg_loss, Hsq


class TestGradientFlow:

    def test_h2_loss_backward(self):
        """h2_loss should produce a tensor with grad_fn that supports .backward()."""
        n_snp = 100
        ld = torch.ones(n_snp, 1, dtype=torch.float64)
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.ones(n_snp, 1, dtype=torch.float64) * 1e5
        M = torch.tensor([[1e7]], dtype=torch.float64)
        chisq = torch.ones(n_snp, 1, dtype=torch.float64) * 2.0
        chisq.requires_grad_(True)

        loss = h2_loss(chisq, ld, w_ld, N, M, intercept=1.0)
        loss.backward()

        assert chisq.grad is not None
        assert not torch.all(chisq.grad == 0)

    def test_h2_gradient_sign(self):
        """Increasing chi-sq should increase h2, so d(h2)/d(chisq) > 0."""
        n_snp = 100
        ld = torch.ones(n_snp, 1, dtype=torch.float64) * 50
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.ones(n_snp, 1, dtype=torch.float64) * 1e5
        M = torch.tensor([[1e7]], dtype=torch.float64)
        chisq = torch.ones(n_snp, 1, dtype=torch.float64) * 1.5
        chisq.requires_grad_(True)

        loss = h2_loss(chisq, ld, w_ld, N, M, intercept=1.0)
        loss.backward()

        # All gradients should be non-negative (more chi-sq → more h2)
        assert (chisq.grad >= 0).all()

    def test_h2_free_intercept_backward(self):
        """Free-intercept Hsq should also be differentiable."""
        n_snp = 100
        ld = torch.ones(n_snp, 1, dtype=torch.float64) * 50
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.ones(n_snp, 1, dtype=torch.float64) * 1e5
        M = torch.tensor([[1e7]], dtype=torch.float64)
        chisq = torch.ones(n_snp, 1, dtype=torch.float64) * 2.0
        chisq.requires_grad_(True)

        result = Hsq(chisq, ld, w_ld, N, M, intercept=None)
        result.tot.backward()

        assert chisq.grad is not None

    def test_h2_2annot_backward(self):
        """Multi-annotation Hsq should be differentiable."""
        rng = np.random.RandomState(42)
        n_snp = 200
        ld_np = (np.abs(rng.normal(size=n_snp * 2)) + 1).reshape((n_snp, 2))
        N_np = np.ones((n_snp, 1)) * 1e5
        M_np = np.ones((1, 2)) * 5e6
        chisq_np = 1 + 1e5 * (
            ld_np[:, 0:1] * 0.3 / M_np[0, 0] + ld_np[:, 1:2] * 0.5 / M_np[0, 1]
        )

        ld = torch.from_numpy(ld_np)
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.from_numpy(N_np)
        M = torch.from_numpy(M_np)
        chisq = torch.from_numpy(chisq_np).requires_grad_(True)

        result = Hsq(chisq, ld, w_ld, N, M, intercept=1.0)
        result.tot.backward()

        assert chisq.grad is not None
        assert not torch.all(chisq.grad == 0)

    def test_rg_loss_backward(self):
        """rg_loss should support .backward() through both Hsq and Gencov."""
        rng = np.random.RandomState(77)
        n_snp = 50
        ld_np = np.abs(rng.normal(size=n_snp * 2).reshape((n_snp, 2))) + 2
        z1_np = (np.sum(ld_np, axis=1) * 10).reshape((n_snp, 1))

        ld = torch.from_numpy(ld_np)
        w_ld = torch.from_numpy(rng.uniform(0.5, 2, size=(n_snp, 1)))
        N1 = 9 * torch.ones(n_snp, 1, dtype=torch.float64)
        M = torch.tensor([[700.0, 222.0]], dtype=torch.float64)
        z1 = torch.from_numpy(z1_np).requires_grad_(True)
        z2 = -z1.detach().clone().requires_grad_(True)

        rg = rg_loss(z1, z2, ld, w_ld, N1, N1, M)
        rg.backward()

        assert z1.grad is not None


class TestGradcheck:

    @pytest.mark.slow
    def test_h2_gradcheck(self):
        """torch.autograd.gradcheck on h2_loss with small inputs."""
        n_snp = 15
        ld = torch.ones(n_snp, 1, dtype=torch.float64) * 50
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.ones(n_snp, 1, dtype=torch.float64) * 1e5
        M = torch.tensor([[1e7]], dtype=torch.float64)
        chisq = torch.ones(n_snp, 1, dtype=torch.float64) * 1.5
        chisq.requires_grad_(True)

        def fn(x):
            return h2_loss(x, ld, w_ld, N, M, intercept=1.0).unsqueeze(0)

        torch.autograd.gradcheck(fn, (chisq,), eps=1e-5, atol=1e-3, rtol=1e-3)
