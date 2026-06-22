"""Optimization smoke tests — verify that h2_loss can be minimized with SGD."""

import numpy as np
import torch
import pytest

from h2vae.ldsc_torch import h2_loss


class TestOptimization:

    def test_optimize_toward_target_h2(self):
        """Start with wrong chi-sq, optimize toward target h2 = 0.5."""
        n_snp = 200
        target_h2 = 0.5
        M_val = 1e7
        N_val = 1e5

        ld = torch.ones(n_snp, 1, dtype=torch.float64) * 100
        w_ld = torch.ones(n_snp, 1, dtype=torch.float64)
        N = torch.ones(n_snp, 1, dtype=torch.float64) * N_val
        M = torch.tensor([[M_val]], dtype=torch.float64)

        # Initialize chi-sq to give h2 ≈ 0 (all 1s)
        chisq = torch.ones(n_snp, 1, dtype=torch.float64) * 1.01
        chisq = chisq.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([chisq], lr=0.01)
        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()
            h2 = h2_loss(chisq.clamp(min=1.0), ld, w_ld, N, M, intercept=1.0)
            loss = (h2 - target_h2) ** 2
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )
