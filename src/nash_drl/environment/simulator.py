from __future__ import annotations

from dataclasses import dataclass

from nash_drl.domain import Vehicle

from ..data import CSRMap, Paths


@dataclass(slots=True)
class SimulationState:
    csr_map: CSRMap
    vehicles: list[Vehicle]


class RoadSimulator:
    """Minimal deterministic route executor; physics can be expanded without touching networks."""

    def apply_paths(self, state: SimulationState, paths: Paths) -> SimulationState:
        edge_flow = state.csr_map.edge_flow.clone().zero_()
        for path in paths.edge_ids:
            for edge_id in path:
                edge_flow[edge_id] += 1.0
        state.csr_map.edge_flow = edge_flow
        return state
