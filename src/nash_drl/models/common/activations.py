from __future__ import annotations

from torch import Tensor

import torch.nn.functional as F


def strictly_positive(x: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Map real values to strictly positive values with softplus."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return F.softplus(x) + epsilon
