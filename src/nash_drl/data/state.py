from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .csr_map import CSRMap


@dataclass(slots=True)
class GlobalState:
    """Environment state exposed to the feature extractor.

    Network features remain `[N,F]`; explicit node tensors are retained so the
    deterministic route mapper does not have to reverse-engineer semantic state
    from normalized feature vectors.
    """

    csr_map: CSRMap
    agent_features: Tensor  # [N, F]
    edge_flow: Tensor  # [E], dynamic traffic flow observed from SUMO at state time
    current_nodes: Tensor  # [N]
    next_destinations: Tensor  # [N]
    final_destinations: Tensor  # [N]
    flow_time_s: float = 0.0
    edge_flow_source: str = "unknown"

    def validate(self) -> None:
        n = self.agent_features.shape[0]
        if self.agent_features.ndim != 2:
            raise ValueError(f"agent_features must be [N,F], got {self.agent_features.shape}")
        if self.edge_flow.ndim != 1:
            raise ValueError(f"edge_flow must be [E], got {self.edge_flow.shape}")
        if self.edge_flow.shape[0] != self.csr_map.num_edges:
            raise ValueError("edge_flow length must equal number of graph edges")
        for name, tensor in (
            ("current_nodes", self.current_nodes),
            ("next_destinations", self.next_destinations),
            ("final_destinations", self.final_destinations),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != n:
                raise ValueError(f"{name} must have shape [N]")
