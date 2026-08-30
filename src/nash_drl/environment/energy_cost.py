from __future__ import annotations

import torch
from torch import Tensor


def charging_price(
    flow: Tensor,
    overhead: float,
    fixed_unit_cost: float,
    floor_price: float,
) -> Tensor:
    eps = torch.finfo(flow.dtype).eps
    active_price = overhead / torch.clamp(flow, min=eps) + fixed_unit_cost
    return torch.where(flow > 0, torch.maximum(active_price, torch.tensor(floor_price, dtype=flow.dtype, device=flow.device)), torch.zeros_like(flow))


def edge_energy_cost(
    edge_length_km: Tensor,
    flow: Tensor,
    energy_rate_kwh_per_km: float,
    overhead: float,
    fixed_unit_cost: float,
    floor_price: float,
) -> Tensor:
    price = charging_price(flow, overhead, fixed_unit_cost, floor_price)
    return energy_rate_kwh_per_km * edge_length_km * price
