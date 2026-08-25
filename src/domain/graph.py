from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Edge:
    id: int
    source: int
    destination: int
    length_km: float
    capacity_vph: float


@dataclass(frozen=True, slots=True)
class DirectedGraph:
    num_nodes: int
    edges: tuple[Edge, ...]

    @property
    def num_edges(self) -> int:
        return len(self.edges)
