from __future__ import annotations

import torch
from torch import Tensor

from nash_drl.domain import Vehicle

DEFAULT_AGENT_FEATURE_DIM = 6


def encode_agent_features(
    vehicles: list[Vehicle],
    num_nodes: int,
    *,
    dtype: torch.dtype,
    max_trips: int | None = None,
    max_budget: float | None = None,
    max_speed_kmh: float | None = None,
) -> Tensor:
    """Encode the six Section-3.2.1 vehicle features in comparable ranges.

    Node identifiers are normalized to [0,1].  Remaining trips, remaining
    budget, and free-flow speed are normalized against dataset-level maxima so
    the neural networks do not receive features spanning orders of magnitude.
    """
    if not vehicles:
        return torch.empty((0, DEFAULT_AGENT_FEATURE_DIM), dtype=dtype)
    node_scale = max(1, num_nodes - 1)
    trip_scale = max(1, max_trips if max_trips is not None else max(len(v.trips) for v in vehicles))
    budget_scale = max(1.0, max_budget if max_budget is not None else max(v.budget for v in vehicles))
    speed_scale = max(1.0, max_speed_kmh if max_speed_kmh is not None else max(v.free_flow_speed_kmh for v in vehicles))

    rows: list[list[float]] = []
    for v in vehicles:
        rows.append([
            float(v.current_node if v.current_node is not None else 0) / node_scale,
            float(v.next_destination) / node_scale if v.trips else 0.0,
            float(v.final_destination) / node_scale if v.trips else 0.0,
            float(v.remaining_trip_count) / trip_scale,
            float(v.remaining_budget if v.remaining_budget is not None else v.budget) / budget_scale,
            float(v.free_flow_speed_kmh) / speed_scale,
        ])
    return torch.tensor(rows, dtype=dtype)
