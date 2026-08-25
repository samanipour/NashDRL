from __future__ import annotations

from torch import Tensor


def edge_flow_features(edge_flow: Tensor) -> Tensor:
    if edge_flow.ndim != 1:
        raise ValueError("edge_flow must be [E]")
    return edge_flow
