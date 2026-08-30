from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle


@dataclass(frozen=True, slots=True)
class MockDataConfig:
    seed: int = 42
    num_vehicles: int = 12
    min_trips_per_vehicle: int = 1
    max_trips_per_vehicle: int = 3
    graph_nodes: int = 12
    graph_extra_edges: int = 10
    road_min_length_km: float = 1.0
    road_max_length_km: float = 8.0
    capacity_min_vph: int = 20
    capacity_max_vph: int = 80
    speed_min_kmh: float = 30.0
    speed_max_kmh: float = 70.0
    budget_min: float = 75.0
    budget_max: float = 300.0
    energy_rate_kwh_per_km: float = 1.0
    charging_overhead: float = 6.0
    charging_fixed_cost: float = 1.0
    charging_floor_price: float = 0.0
    congestion_alpha: float = 0.15
    congestion_beta: float = 4.0
    vehicle_departure_gap_s: float = 2.0


class MockDatasetGenerator:
    """Generate a deterministic, connected Nash-DRL dataset."""

    def __init__(self, config: MockDataConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(self) -> ProblemDefinition:
        n = self.config.graph_nodes
        if n < 3:
            raise ValueError("graph_nodes must be >= 3")
        if self.config.min_trips_per_vehicle < 1 or self.config.max_trips_per_vehicle < self.config.min_trips_per_vehicle:
            raise ValueError("Invalid trip range")

        edges: list[Edge] = []
        edge_id = 0
        # Directed ring guarantees strong connectivity.
        for u in range(n):
            v = (u + 1) % n
            edges.append(self._edge(edge_id, u, v))
            edge_id += 1
            edges.append(self._edge(edge_id, v, u))
            edge_id += 1

        existing = {(e.source, e.destination) for e in edges}
        candidates = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in existing]
        self.rng.shuffle(candidates)
        for u, v in candidates[: self.config.graph_extra_edges]:
            edges.append(self._edge(edge_id, u, v))
            edge_id += 1

        vehicles: list[Vehicle] = []
        for vehicle_id in range(self.config.num_vehicles):
            trip_count = self.rng.randint(self.config.min_trips_per_vehicle, self.config.max_trips_per_vehicle)
            nodes = [self.rng.randrange(n) for _ in range(trip_count + 1)]
            for j in range(1, len(nodes)):
                if nodes[j] == nodes[j - 1]:
                    nodes[j] = (nodes[j] + 1) % n
            trips = [Trip(nodes[j], nodes[j + 1]) for j in range(trip_count)]
            vehicles.append(
                Vehicle(
                    id=vehicle_id,
                    trips=trips,
                    free_flow_speed_kmh=self.rng.uniform(self.config.speed_min_kmh, self.config.speed_max_kmh),
                    budget=self.rng.uniform(self.config.budget_min, self.config.budget_max),
                )
            )
        return ProblemDefinition(graph=DirectedGraph(num_nodes=n, edges=tuple(edges)), vehicles=vehicles)

    def _edge(self, edge_id: int, source: int, destination: int) -> Edge:
        return Edge(
            id=edge_id,
            source=source,
            destination=destination,
            length_km=self.rng.uniform(self.config.road_min_length_km, self.config.road_max_length_km),
            capacity_vph=float(self.rng.randint(self.config.capacity_min_vph, self.config.capacity_max_vph)),
        )

    def save(self, problem: ProblemDefinition, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "generator": "MockDatasetGenerator",
            "seed": self.config.seed,
            "config": asdict(self.config),
            "graph": {
                "num_nodes": problem.graph.num_nodes,
                "edges": [asdict(edge) for edge in problem.graph.edges],
            },
            "vehicles": [
                {
                    "id": v.id,
                    "free_flow_speed_kmh": v.free_flow_speed_kmh,
                    "budget": v.budget,
                    "trips": [asdict(t) for t in v.trips],
                }
                for v in problem.vehicles
            ],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output


def load_problem(path: str | Path) -> tuple[ProblemDefinition, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = DirectedGraph(
        num_nodes=int(payload["graph"]["num_nodes"]),
        edges=tuple(Edge(**e) for e in payload["graph"]["edges"]),
    )
    vehicles = [
        Vehicle(
            id=int(v["id"]),
            trips=[Trip(**trip) for trip in v["trips"]],
            free_flow_speed_kmh=float(v["free_flow_speed_kmh"]),
            budget=float(v["budget"]),
        )
        for v in payload["vehicles"]
    ]
    return ProblemDefinition(graph=graph, vehicles=vehicles), payload
