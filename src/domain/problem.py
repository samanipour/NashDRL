from __future__ import annotations

from dataclasses import dataclass

from .graph import DirectedGraph
from .vehicle import Vehicle


@dataclass(slots=True)
class ProblemDefinition:
    graph: DirectedGraph
    vehicles: list[Vehicle]
