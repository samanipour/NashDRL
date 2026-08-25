from __future__ import annotations

import torch
from torch import Tensor


def add_gaussian_exploration(mu: Tensor, sigma: float, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator) * sigma
    return mu + noise
