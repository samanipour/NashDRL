from __future__ import annotations

import torch
from torch import Tensor

from domain import Vehicle

# [current node, next destination, final destination, remaining trip count,
#  remaining budget, free-flow speed]
DEFAULT_AGENT_FEATURE_DIM = 6


def encode_agent_features(vehicles: list[Vehicle], num_nodes: int, *, dtype: torch.dtype) -> Tensor:
    rows: list[list[float]] = []
    node_scale = max(1, num_nodes - 1)
    for v in vehicles:
        rows.append([
            float(v.current_node or 0) / node_scale,
            float(v.next_destination) / node_scale if v.trips else 0.0,
            float(v.final_destination) / node_scale if v.trips else 0.0,
            float(v.remaining_trip_count),
            float(v.remaining_budget or 0.0),
            float(v.free_flow_speed_kmh),
        ])
    return torch.tensor(rows, dtype=dtype)
