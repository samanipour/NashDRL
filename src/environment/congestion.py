from __future__ import annotations

import torch
from torch import Tensor


def congestion_multiplier(flow: Tensor, capacity: Tensor, alpha: float, beta: float) -> Tensor:
    """Returns 1 + alpha * (flow/capacity)^beta."""
    safe_capacity = torch.clamp(capacity, min=torch.finfo(capacity.dtype).eps)
    return 1.0 + alpha * torch.pow(flow / safe_capacity, beta)
