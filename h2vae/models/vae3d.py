"""3D convolutional VAE for volumetric data (e.g. brain MRI).

Based on the DeepENDO autoencoder architecture (Patel et al.), converted to a
proper VAE with KL divergence and the standard h2vae interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from h2vae.models.base import BaseVAE
from h2vae.models import register


def _make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


class Conv3dCellDown(nn.Module):
    """Two-conv downsampling cell: Conv3d(stride=1) -> Conv3d(stride=2).

    Reduces spatial dimensions by 2x per cell.
    """

    def __init__(self, ni: int, no: int, ks: int = 3, act: str = "leaky_relu",
                 use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv3d(no, no, kernel_size=ks, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(no) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm3d(no) if use_bn else nn.Identity()
        self.act1 = _make_activation(act)
        self.act2 = _make_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class Conv3dCellUp(nn.Module):
    """Two-conv upsampling cell: trilinear interp(2x) -> Conv3d -> Conv3d.

    Increases spatial dimensions by 2x per cell.
    """

    def __init__(self, ni: int, no: int, ks: int = 3, act1: str = "leaky_relu",
                 act2: str = "leaky_relu", use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv3d(no, no, kernel_size=ks, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(no) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm3d(no) if use_bn else nn.Identity()
        self.act1 = _make_activation(act1)
        self.act2 = _make_activation(act2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


@register("vae3d")
class VAE3D(BaseVAE):
    """3D convolutional VAE with channel-doubling encoder and beta-VAE weighting.

    Channel progression through the encoder (with ``steps=4``, ``nf=16``)::

        colors(1) -> nf(16) -> 2*nf(32) -> 4*nf(64) -> 8*nf(128)

    The decoder mirrors this in reverse, ending with a 1x1x1 conv to produce
    ``colors`` output channels.

    Args:
        img_size: Cubic spatial resolution. Must be divisible by ``2**steps``.
        nf: Base number of filters (doubled at each encoder stage).
        zdim: Latent dimensionality.
        steps: Number of downsampling / upsampling stages.
        colors: Number of input channels (1 for grayscale MRI).
        external: Dimensionality of external covariates for decode conditioning.
        act: Activation function name.
        beta: KL divergence weight.
        gradient_checkpointing: Trade ~30-40% more compute for ~3x less
            activation memory by recomputing intermediate activations during
            the backward pass.
    """

    ndim = 3
    data_format = "nifti"

    def __init__(
        self,
        img_size: int = 256,
        nf: int = 16,
        zdim: int = 128,
        steps: int = 4,
        colors: int = 1,
        external: int = 0,
        act: str = "leaky_relu",
        beta: float = 1.0,
        gradient_checkpointing: bool = False,
        zs_floor: float = 0.0,
    ):
        super().__init__()

        if img_size % (2 ** steps) != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by 2**steps ({2**steps})"
            )

        self.gradient_checkpointing = gradient_checkpointing
        self.red_img_size = img_size // (2 ** steps)
        self.K = img_size ** 3 * colors
        self.beta = beta
        self.zs_floor = float(zs_floor)
        ks = 3

        # Build channel schedule: [colors, nf, 2*nf, 4*nf, ...]
        channels = [colors] + [nf * (2 ** i) for i in range(steps)]
        self.nf_final = channels[-1]
        self.size_flat = self.red_img_size ** 3 * self.nf_final

        # Encoder
        self.econv = nn.ModuleList()
        for i in range(steps):
            self.econv.append(Conv3dCellDown(channels[i], channels[i + 1], ks, act))

        # Decoder — reverse channel schedule, final cell outputs colors
        self.dconv = nn.ModuleList()
        for i in range(steps - 1, 0, -1):
            self.dconv.append(Conv3dCellUp(channels[i + 1], channels[i], ks, act, act))
        self.dconv.append(
            Conv3dCellUp(channels[1], colors, ks, act1=act, act2="linear", use_bn=False)
        )

        # Latent projections
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)
        self.dense_dec = nn.Linear(zdim + external, self.size_flat)

    def _run_cells(self, cells: nn.ModuleList, x: Tensor) -> Tensor:
        for cell in cells:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(cell, x, use_reentrant=False)
            else:
                x = cell(x)
        return x

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode volumes to latent mean and std.

        Args:
            x: Volumes of shape ``(batch, colors, D, H, W)``.

        Returns:
            zm, zs: Each ``(batch, zdim)``.
        """
        x = self._run_cells(self.econv, x)
        x = x.view(-1, self.size_flat)
        zm = self.dense_zm(x)
        zs = F.softplus(self.dense_zs(x))
        # Optional posterior-std floor (--zs-floor, default off). Empirically
        # (instrumented β=1 spectrum probe, 2026-06-12) the encoder drives zs
        # progressively to exactly 0.0 for several dims; log(zs)=-inf in the KL then
        # produces a NaN that propagates through the latent means and surfaces as the
        # spectrum Cholesky failure. A small floor (e.g. 1e-8) keeps log(zs) finite
        # and prevents that NaN cascade. zs_floor=0 reproduces legacy (no clamp).
        if self.zs_floor > 0:
            zs = zs.clamp_min(self.zs_floor)
        return zm, zs

    def decode(self, z: Tensor, external: Tensor | None = None) -> Tensor:
        """Decode latent vectors to volumes.

        Args:
            z: Latent vectors ``(batch, zdim)``.
            external: Optional covariates ``(batch, external_dim)``.

        Returns:
            Reconstructed volumes ``(batch, colors, D, H, W)``.
        """
        if external is not None:
            z = torch.hstack((z, external))
        x = self.dense_dec(z)
        x = x.view(-1, self.nf_final, self.red_img_size, self.red_img_size, self.red_img_size)
        x = self._run_cells(self.dconv, x)
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
