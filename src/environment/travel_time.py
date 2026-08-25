from __future__ import annotations

import torch
from torch import Tensor

from .congestion import congestion_multiplier


def edge_travel_time(
    edge_length_km: Tensor,
    free_flow_speed_kmh: Tensor,
    flow: Tensor,
    capacity: Tensor,
    alpha: float,
    beta: float,
) -> Tensor:
    base_hours = edge_length_km / torch.clamp(free_flow_speed_kmh, min=1e-8)
    return base_hours * congestion_multiplier(flow, capacity, alpha, beta)
