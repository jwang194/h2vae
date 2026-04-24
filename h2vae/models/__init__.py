"""Model registry and public re-exports for the h2vae models sub-package."""

from __future__ import annotations

from typing import Type

from h2vae.models.base import BaseVAE

_REGISTRY: dict[str, Type[BaseVAE]] = {}


def register(name: str):
    """Class decorator that registers a model under *name*."""

    def wrapper(cls: Type[BaseVAE]) -> Type[BaseVAE]:
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_model_class(name: str) -> Type[BaseVAE]:
    """Look up a registered model class by name.

    Raises :class:`ValueError` if *name* is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return list(_REGISTRY.keys())


# Import submodules to trigger registration.
from h2vae.models.vae2d import VAE  # noqa: E402, F401
from h2vae.models.vae3d import VAE3D  # noqa: E402, F401
from h2vae.models.vae1d import VAE1D  # noqa: E402, F401
