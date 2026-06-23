"""2D convolutional VAE with optional covariate conditioning at decode time."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from h2vae.models.base import BaseVAE
from h2vae.models import register


def _make_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    if name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


class Conv2dCellDown(nn.Module):
    """Two-conv downsampling cell: conv(stride=1) -> conv(stride=2)."""

    def __init__(self, ni: int, no: int, ks: int = 3, act: str = "elu"):
        super().__init__()
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=2, padding=1)
        self.act1 = _make_activation(act)
        self.act2 = _make_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class Conv2dCellUp(nn.Module):
    """Two-conv upsampling cell: interpolate(2x) -> conv -> conv."""

    def __init__(
        self, ni: int, no: int, ks: int = 3, act1: str = "elu", act2: str = "elu"
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=1, padding=1)
        self.act1 = _make_activation(act1)
        self.act2 = _make_activation(act2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


@register("vae2d")
class VAE(BaseVAE):
    """Convolutional variational autoencoder with beta-VAE weighting.

    The ELBO is computed as: loss = MSE + beta * KLD

    Args:
        img_size: Spatial resolution of input images (assumed square).
        nf: Number of convolutional filters per layer.
        zdim: Dimensionality of the latent space.
        steps: Number of downsampling / upsampling stages.
        colors: Number of input image channels.
        external: Dimensionality of external covariates concatenated to z
            before decoding. Set to 0 for no conditioning.
        act: Activation function name ("elu", "relu", or "linear").
        beta: Weight on KL divergence relative to reconstruction.
    """

    ndim = 2
    data_format = "image"

    def __init__(
        self,
        img_size: int = 256,
        nf: int = 32,
        zdim: int = 2,
        steps: int = 5,
        colors: int = 1,
        external: int = 0,
        act: str = "elu",
        beta: float = 1.0,
        zs_floor: float = 0.0,
    ):
        super().__init__()

        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.beta = beta
        self.zs_floor = float(zs_floor)
        ks = 3

        # Encoder
        self.econv = nn.ModuleList()
        self.econv.append(Conv2dCellDown(colors, nf, ks, act))
        for _ in range(steps - 1):
            self.econv.append(Conv2dCellDown(nf, nf, ks, act))

        # Decoder
        self.dconv = nn.ModuleList()
        for _ in range(steps - 1):
            self.dconv.append(Conv2dCellUp(nf, nf, ks, act1=act, act2=act))
        self.dconv.append(Conv2dCellUp(nf, colors, ks, act1=act, act2="linear"))

        # Latent projections
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)
        self.dense_dec = nn.Linear(zdim + external, self.size_flat)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for cell in self.econv:
            x = cell(x)
        x = x.view(-1, self.size_flat)
        zm = self.dense_zm(x)
        zs = F.softplus(self.dense_zs(x))
        # Optional posterior-std floor (--zs-floor, default off); keeps log(zs)
        # finite when the encoder drives zs->0 (see vae3d for the empirical detail).
        if self.zs_floor > 0:
            zs = zs.clamp_min(self.zs_floor)
        return zm, zs

    def decode(self, z: Tensor, external: Tensor | None = None) -> Tensor:
        if external is not None:
            z = torch.hstack((z, external))
        x = self.dense_dec(z)
        x = x.view(-1, self.nf, self.red_img_size, self.red_img_size)
        for cell in self.dconv:
            x = cell(x)
        return x

    def mse(self, x: Tensor, xr: Tensor) -> Tensor:
        return ((xr - x) ** 2).view(x.shape[0], self.K).mean(1)[:, None]

    def forward(
        self, x: Tensor, eps: Tensor, external: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        zm, zs = self.encode(x)
        z = zm + eps * zs
        xr = self.decode(z, external)
        mse = self.mse(x, xr)
        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None]
            / self.K
        )
        loss = mse + self.beta * kld
        return loss, mse, kld

    def sample(self, x: Tensor, eps: Tensor) -> Tensor:
        zm, zs = self.encode(x)
        return zm + eps * zs
