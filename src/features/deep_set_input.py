from __future__ import annotations

from torch import Tensor


def validate_deep_set_input(x: Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"Deep Sets input must be [N,N-1,F], got {tuple(x.shape)}")
