"""Abstract base class defining the VAE interface for the h2vae pipeline."""

from __future__ import annotations

import abc

import torch.nn as nn
from torch import Tensor


class BaseVAE(nn.Module, abc.ABC):
    """Interface contract for all VAE architectures.

    Subclasses **must** set these instance attributes in ``__init__``:

    * ``K`` (int) — total element count per sample (pixels, voxels, time
      steps * channels).  Used to normalise the KL divergence term.
    * ``beta`` (float) — weight on the KL divergence relative to
      reconstruction.

    Subclasses **must** define these class-level attributes:

    * ``ndim`` (int) — spatial dimensionality of the input data (1, 2,
      or 3).
    * ``data_format`` (str) — drives streaming-dataset selection in the
      training script.  One of ``"image"``, ``"nifti"``, or
      ``"timeseries"``.
    """

    ndim: int
    data_format: str

    @abc.abstractmethod
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs to latent mean and std.

        Args:
            x: Input tensor whose layout depends on ``ndim``.

        Returns:
            zm: Latent means of shape ``(batch, zdim)``.
            zs: Latent stds of shape ``(batch, zdim)``.
        """
        ...

    @abc.abstractmethod
    def decode(self, z: Tensor, external: Tensor | None = None) -> Tensor:
        """Decode latent vectors back to input space.

        Args:
            z: Latent vectors of shape ``(batch, zdim)``.
            external: Optional covariates concatenated to *z* before
                the dense projection.

        Returns:
            Reconstructed tensor in the same layout as ``encode`` input.
        """
        ...

    @abc.abstractmethod
    def mse(self, x: Tensor, xr: Tensor) -> Tensor:
        """Per-sample mean squared error.

        Returns:
            Tensor of shape ``(batch, 1)``.
        """
        ...

    @abc.abstractmethod
    def forward(
        self, x: Tensor, eps: Tensor, external: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass: encode, sample, decode, compute loss.

        Returns:
            loss: Per-sample ELBO of shape ``(batch, 1)``.
            mse: Per-sample MSE of shape ``(batch, 1)``.
            kld: Per-sample KL divergence of shape ``(batch, 1)``.
        """
        ...

    @abc.abstractmethod
    def sample(self, x: Tensor, eps: Tensor) -> Tensor:
        """Encode and reparameterize.

        Returns:
            Sampled latent vectors of shape ``(batch, zdim)``.
        """
        ...
