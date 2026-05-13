"""Tests for the rank-B heritability prototype.

Compares the rank-B incremental computation against the full
``||X^T Z||_F^2 / m`` ground truth across a sequence of randomized
minibatches. Checks both the scalar value and the gradient flowing
back to the latent (which is what backprop into the encoder will
actually receive).
"""
from __future__ import annotations

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from h2vae.rank_b_heritability import RankBHeritability  # noqa: E402


def make_X(n: int, m: int, seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Standardized ternary genotype matrix.

    Genotypes drawn from {0, 1, 2} uniformly; columns z-scored.
    fp64 by default to keep numerical noise out of equivalence checks.
    """
    g = torch.randint(0, 3, (n, m), generator=torch.Generator().manual_seed(seed)).to(dtype)
    mu = g.mean(dim=0, keepdim=True)
    sd = g.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (g - mu) / sd


def make_Z(n: int, zdim: int, seed: int, dtype: torch.dtype = torch.float64,
           requires_grad: bool = False) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(n, zdim, generator=rng, dtype=dtype)
    if requires_grad:
        z.requires_grad_(True)
    return z


def test_rebuild_matches_full() -> None:
    """After rebuild(), update_and_loss with empty delta == full_loss(Z)."""
    n, m, zdim, B = 64, 32, 4, 8
    X = make_X(n, m, seed=1)
    Z = make_Z(n, zdim, seed=2)

    her = RankBHeritability(X)
    her.rebuild(Z)

    # Idx subset; Z_batch equal to current Z_prev[idxs] → delta = 0,
    # u unchanged, loss == full_loss(Z).
    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(3))[:B]
    Z_batch = Z[idxs].clone()
    loss_inc = her.update_and_loss(Z_batch, idxs)
    loss_full = her.full_loss(Z)
    assert torch.allclose(loss_inc, loss_full, atol=1e-9), (
        f"empty-delta update gave {loss_inc.item()} vs full {loss_full.item()}"
    )
    print(f"  rebuild + empty-delta:  inc={loss_inc.item():.6g}  full={loss_full.item():.6g}  ok")


def test_single_step_value_matches_full() -> None:
    """One rank-B update produces the same scalar as a from-scratch full pass."""
    n, m, zdim, B = 64, 32, 4, 8
    X = make_X(n, m, seed=10)
    Z0 = make_Z(n, zdim, seed=20)

    her = RankBHeritability(X)
    her.rebuild(Z0)

    # Move on to a new Z that differs from Z0 only in B rows.
    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(30))[:B]
    Z1 = Z0.clone()
    Z1[idxs] = make_Z(B, zdim, seed=40)

    loss_inc = her.update_and_loss(Z1[idxs].clone(), idxs)
    loss_full = her.full_loss(Z1)
    assert torch.allclose(loss_inc, loss_full, atol=1e-9), (
        f"single-step: inc={loss_inc.item()} full={loss_full.item()}"
    )
    print(f"  single rank-B step:     inc={loss_inc.item():.6g}  full={loss_full.item():.6g}  ok")


def test_many_steps_match_full() -> None:
    """Many overlapping minibatches stay aligned with the full ground truth."""
    n, m, zdim, B, T = 100, 50, 5, 12, 30
    X = make_X(n, m, seed=100)
    Z = make_Z(n, zdim, seed=200)

    her = RankBHeritability(X)
    her.rebuild(Z)

    gen = torch.Generator().manual_seed(300)
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, generator=gen, dtype=Z.dtype)
        Z = Z.clone()
        Z[idxs] = Z_new
        loss_inc = her.update_and_loss(Z_new, idxs)
        loss_full = her.full_loss(Z)
        rel = (loss_inc - loss_full).abs() / loss_full.abs().clamp_min(1e-12)
        assert rel.item() < 1e-9, (
            f"step {t}: inc={loss_inc.item()} full={loss_full.item()} rel={rel.item()}"
        )
    print(f"  {T} steps, drift bounded: final inc={loss_inc.item():.6g}  full={loss_full.item():.6g}  ok")


def test_gradient_matches_full() -> None:
    """Gradient w.r.t. Z_batch matches a fresh autograd through ||X^T Z||^2 / m."""
    n, m, zdim, B = 64, 32, 4, 8
    X = make_X(n, m, seed=1000)
    Z0 = make_Z(n, zdim, seed=2000)

    her = RankBHeritability(X)
    her.rebuild(Z0)

    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(3000))[:B]

    # Path A: rank-B update, autograd through Z_batch.
    Z_batch_A = Z0[idxs].detach().clone().requires_grad_(True)
    loss_A = her.update_and_loss(Z_batch_A, idxs)
    grad_A = torch.autograd.grad(loss_A, Z_batch_A)[0]

    # Path B: rebuild fresh full Z (with the same B rows live), compute
    # full loss, autograd back to those rows.
    Z_full = Z0.detach().clone()
    Z_batch_B = Z0[idxs].detach().clone().requires_grad_(True)
    Z_full = Z_full.clone()
    Z_full[idxs] = Z_batch_B
    u_full = X.T @ Z_full
    loss_B = (u_full ** 2).sum() / m
    grad_B = torch.autograd.grad(loss_B, Z_batch_B)[0]

    assert torch.allclose(loss_A, loss_B, atol=1e-9), (
        f"loss mismatch: A={loss_A.item()} B={loss_B.item()}"
    )
    assert torch.allclose(grad_A, grad_B, atol=1e-9), (
        f"grad mismatch: max abs diff = {(grad_A - grad_B).abs().max().item()}"
    )
    print(f"  forward + backward:     "
          f"loss A={loss_A.item():.6g} B={loss_B.item():.6g}  "
          f"grad max-abs-diff={(grad_A - grad_B).abs().max().item():.2e}  ok")


def test_gradient_through_encoder() -> None:
    """End-to-end: gradient flows through a fake encoder via rank-B update.

    Confirms the rank-B path can be plugged into a real training loop
    where Z_batch is the output of an `nn.Module`, not a leaf tensor.
    """
    n, m, zdim, B = 64, 32, 4, 8
    X = make_X(n, m, seed=10000)
    Z0 = make_Z(n, zdim, seed=20000)

    her = RankBHeritability(X)
    her.rebuild(Z0)

    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(30000))[:B]

    # Fake encoder: a single linear layer feeding the minibatch.
    # Input: minibatch indices encoded as one-hot rows of dim n.
    enc = torch.nn.Linear(n, zdim, bias=False).to(dtype=torch.float64)
    one_hot = torch.zeros(B, n, dtype=torch.float64)
    one_hot[torch.arange(B), idxs] = 1.0

    # Rank-B path through the encoder.
    Z_batch_A = enc(one_hot)
    loss_A = her.update_and_loss(Z_batch_A, idxs)
    grad_A = torch.autograd.grad(loss_A, enc.weight, retain_graph=False)[0]

    # Re-init module for the full path (so Z_prev is fresh).
    her_full = RankBHeritability(X)
    her_full.rebuild(Z0)
    Z_full = Z0.detach().clone()
    Z_batch_B = enc(one_hot)
    Z_full = Z_full.clone()
    Z_full[idxs] = Z_batch_B
    u_full = X.T @ Z_full
    loss_B = (u_full ** 2).sum() / m
    grad_B = torch.autograd.grad(loss_B, enc.weight)[0]

    assert torch.allclose(loss_A, loss_B, atol=1e-9), (
        f"end-to-end loss mismatch: A={loss_A.item()} B={loss_B.item()}"
    )
    assert torch.allclose(grad_A, grad_B, atol=1e-9), (
        f"end-to-end grad max diff = {(grad_A - grad_B).abs().max().item()}"
    )
    print(f"  encoder-side gradient:  "
          f"loss A={loss_A.item():.6g} B={loss_B.item():.6g}  "
          f"grad max-abs-diff={(grad_A - grad_B).abs().max().item():.2e}  ok")


def test_repeated_idxs_in_consecutive_batches() -> None:
    """If a row is in two consecutive minibatches, the update still tracks."""
    n, m, zdim = 40, 20, 3
    X = make_X(n, m, seed=999)
    Z = make_Z(n, zdim, seed=888)

    her = RankBHeritability(X)
    her.rebuild(Z)

    # Step 1: update rows {0, 1, 2, 3}.
    idxs1 = torch.tensor([0, 1, 2, 3])
    Z_new_1 = make_Z(4, zdim, seed=111)
    Z = Z.clone()
    Z[idxs1] = Z_new_1
    her.update_and_loss(Z_new_1, idxs1)

    # Step 2: overlap — rows {2, 3, 4, 5}.
    idxs2 = torch.tensor([2, 3, 4, 5])
    Z_new_2 = make_Z(4, zdim, seed=222)
    Z = Z.clone()
    Z[idxs2] = Z_new_2
    loss_inc = her.update_and_loss(Z_new_2, idxs2)

    loss_full = her.full_loss(Z)
    assert torch.allclose(loss_inc, loss_full, atol=1e-9), (
        f"overlapping batches: inc={loss_inc.item()} full={loss_full.item()}"
    )
    print(f"  overlapping minibatches: inc={loss_inc.item():.6g}  full={loss_full.item():.6g}  ok")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("RankBHeritability tests:")
    test_rebuild_matches_full()
    test_single_step_value_matches_full()
    test_many_steps_match_full()
    test_gradient_matches_full()
    test_gradient_through_encoder()
    test_repeated_idxs_in_consecutive_batches()
    print("all tests passed.")
