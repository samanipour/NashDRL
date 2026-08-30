from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle


@dataclass(frozen=True, slots=True)
class MockDataConfig:
    """Parameters for reproducible stochastic mock-data generation.

    Numeric entity attributes use clipped normal distributions. Integer counts,
    including the number of trips per vehicle, are sampled from a normal
    distribution and clipped/rounded to the configured bounds.
    """

    seed: int = 42
    num_vehicles: int = 12
    min_trips_per_vehicle: int = 1
    max_trips_per_vehicle: int = 3
    graph_nodes: int = 12
    graph_extra_edges: int = 10
    road_min_length_km: float = 1.0
    road_max_length_km: float = 8.0
    road_length_std_km: float = 1.5
    capacity_min_vph: int = 20
    capacity_max_vph: int = 80
    capacity_std_vph: float = 15.0
    speed_min_kmh: float = 30.0
    speed_max_kmh: float = 70.0
    speed_std_kmh: float = 10.0
    budget_min: float = 75.0
    budget_max: float = 300.0
    budget_std: float = 45.0
    trip_count_mean: float | None = None
    trip_count_std: float = 0.9
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
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> ProblemDefinition:
        c = self.config
        if c.graph_nodes < 3:
            raise ValueError("graph_nodes must be >= 3")
        if c.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if c.min_trips_per_vehicle < 1 or c.max_trips_per_vehicle < c.min_trips_per_vehicle:
            raise ValueError("Invalid trip range")

        edges: list[Edge] = []
        edge_id = 0
        # Bidirectional ring guarantees strong connectivity.
        for u in range(c.graph_nodes):
            v = (u + 1) % c.graph_nodes
            edges.append(self._edge(edge_id, u, v))
            edge_id += 1
            edges.append(self._edge(edge_id, v, u))
            edge_id += 1

        existing = {(e.source, e.destination) for e in edges}
        candidates = [
            (u, v)
            for u in range(c.graph_nodes)
            for v in range(c.graph_nodes)
            if u != v and (u, v) not in existing
        ]
        self.rng.shuffle(candidates)
        for u, v in candidates[: min(c.graph_extra_edges, len(candidates))]:
            edges.append(self._edge(edge_id, u, v))
            edge_id += 1

        vehicles: list[Vehicle] = []
        mean = c.trip_count_mean if c.trip_count_mean is not None else (c.min_trips_per_vehicle + c.max_trips_per_vehicle) / 2.0
        for vehicle_id in range(c.num_vehicles):
            trip_count = self._normal_int(mean, c.trip_count_std, c.min_trips_per_vehicle, c.max_trips_per_vehicle)
            nodes = self.rng.integers(0, c.graph_nodes, size=trip_count + 1).tolist()
            for j in range(1, len(nodes)):
                if nodes[j] == nodes[j - 1]:
                    nodes[j] = (nodes[j] + 1) % c.graph_nodes
            trips = [Trip(int(nodes[j]), int(nodes[j + 1])) for j in range(trip_count)]
            vehicles.append(
                Vehicle(
                    id=vehicle_id,
                    trips=trips,
                    free_flow_speed_kmh=self._normal(c.speed_min_kmh, c.speed_std_kmh, c.speed_min_kmh, c.speed_max_kmh),
                    budget=self._normal(c.budget_min, c.budget_std, c.budget_min, c.budget_max),
                )
            )
        return ProblemDefinition(graph=DirectedGraph(num_nodes=c.graph_nodes, edges=tuple(edges)), vehicles=vehicles)

    def _edge(self, edge_id: int, source: int, destination: int) -> Edge:
        c = self.config
        return Edge(
            id=edge_id,
            source=source,
            destination=destination,
            length_km=self._normal(c.road_min_length_km, c.road_length_std_km, c.road_min_length_km, c.road_max_length_km),
            capacity_vph=self._normal(c.capacity_min_vph, c.capacity_std_vph, c.capacity_min_vph, c.capacity_max_vph),
        )

    def _normal(self, mean: float, std: float, low: float, high: float) -> float:
        return float(np.clip(self.rng.normal(mean, max(std, 1e-9)), low, high))

    def _normal_int(self, mean: float, std: float, low: int, high: int) -> int:
        return int(np.clip(np.rint(self.rng.normal(mean, max(std, 1e-9))), low, high))

    def save(self, problem: ProblemDefinition, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": 2,
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
