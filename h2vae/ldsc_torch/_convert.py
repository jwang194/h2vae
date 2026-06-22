"""numpy <-> torch conversion helpers."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class InputContext:
    """Records whether original inputs were numpy, and the target device/dtype."""

    was_numpy: bool
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def from_inputs(cls, *args):
        was_numpy = any(isinstance(a, np.ndarray) for a in args if a is not None)
        if was_numpy:
            return cls(was_numpy=True, device=torch.device("cpu"), dtype=torch.float64)
        for a in args:
            if isinstance(a, torch.Tensor):
                return cls(was_numpy=False, device=a.device, dtype=torch.float64)
        return cls(was_numpy=True, device=torch.device("cpu"), dtype=torch.float64)


def to_tensor(x, ctx: InputContext) -> torch.Tensor:
    """Convert numpy array, scalar, or tensor to torch.Tensor with the context's device/dtype."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float64)).to(device=ctx.device)
    if isinstance(x, torch.Tensor):
        return x.to(device=ctx.device, dtype=ctx.dtype)
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=ctx.dtype, device=ctx.device)
    if isinstance(x, np.matrix):
        return torch.from_numpy(np.asarray(x, dtype=np.float64)).to(device=ctx.device)
    raise TypeError(f"Cannot convert {type(x)} to tensor")


def maybe_to_numpy(x, ctx: InputContext):
    """Convert tensor back to numpy if the original inputs were numpy."""
    if ctx.was_numpy and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
