from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle


@dataclass(frozen=True, slots=True)
class MockDataConfig:
    """Parameters for reproducible stochastic mock-data generation."""

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
    budget_feasibility_multiplier: float = 1.20
    budget_resample_limit: int = 100
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
    """Generate reproducible data with budgets conditioned on feasibility.

    The graph and trip-sets are generated stochastically from the configured
    seed.  Vehicle budgets are still normal-distributed, but their mean is
    anchored to the cost of a deterministic shortest-path baseline so that the
    generated dataset contains at least one budget-feasible reference policy.
    """

    def __init__(self, config: MockDataConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> ProblemDefinition:
        c = self.config
        self._validate_config()
        edges: list[Edge] = []
        edge_id = 0

        # Bidirectional ring guarantees strong connectivity and realistic
        # turnaround-capable synthetic routing.
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

        graph = DirectedGraph(num_nodes=c.graph_nodes, edges=tuple(edges))
        vehicles: list[Vehicle] = []
        mean = (
            c.trip_count_mean
            if c.trip_count_mean is not None
            else (c.min_trips_per_vehicle + c.max_trips_per_vehicle) / 2.0
        )

        for vehicle_id in range(c.num_vehicles):
            trip_count = self._normal_int(
                mean, c.trip_count_std, c.min_trips_per_vehicle, c.max_trips_per_vehicle
            )
            nodes = self.rng.integers(0, c.graph_nodes, size=trip_count + 1).tolist()
            for j in range(1, len(nodes)):
                if nodes[j] == nodes[j - 1]:
                    nodes[j] = (nodes[j] + 1) % c.graph_nodes
            trips = [Trip(int(nodes[j]), int(nodes[j + 1])) for j in range(trip_count)]
            vehicles.append(
                Vehicle(
                    id=vehicle_id,
                    trips=trips,
                    free_flow_speed_kmh=self._normal(
                        c.speed_min_kmh, c.speed_std_kmh, c.speed_min_kmh, c.speed_max_kmh
                    ),
                    budget=0.0,
                )
            )

        baseline_costs = self._baseline_feasible_costs(graph, vehicles)
        for vehicle in vehicles:
            required = baseline_costs[vehicle.id]
            if required > c.budget_max:
                raise ValueError(
                    f"Mock generator cannot guarantee a feasible budget for vehicle {vehicle.id}: "
                    f"minimum baseline cost {required:.2f} exceeds budget_max={c.budget_max:.2f}. "
                    "Increase budget_max or reduce trip/road-cost parameters."
                )
            center = max(c.budget_min, required * c.budget_feasibility_multiplier)
            # Reject only the lower tail; this preserves a normal distribution
            # while guaranteeing the reference policy fits inside the budget.
            vehicle.budget = self._feasible_normal(center, c.budget_std, required, c.budget_min, c.budget_max)
            vehicle.remaining_budget = vehicle.budget

        return ProblemDefinition(graph=graph, vehicles=vehicles)

    def _baseline_feasible_costs(
        self,
        graph: DirectedGraph,
        vehicles: list[Vehicle],
    ) -> dict[int, float]:
        edge_by_id = {e.id: e for e in graph.edges}
        adjacency: dict[int, list[tuple[int, float, int]]] = {u: [] for u in range(graph.num_nodes)}
        for e in graph.edges:
            adjacency[e.source].append((e.destination, e.length_km, e.id))

        def shortest(source: int, destination: int) -> list[int]:
            if source == destination:
                return []
            dist = {source: 0.0}
            prev: dict[int, tuple[int, int]] = {}
            heap = [(0.0, source)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float("inf")):
                    continue
                if node == destination:
                    break
                for nxt, length, edge_id in adjacency[node]:
                    nd = d + length
                    if nd < dist.get(nxt, float("inf")):
                        dist[nxt] = nd
                        prev[nxt] = (node, edge_id)
                        heapq.heappush(heap, (nd, nxt))
            if destination not in prev:
                raise ValueError(f"No path from node {source} to node {destination}")
            path: list[int] = []
            node = destination
            while node != source:
                parent, edge_id = prev[node]
                path.append(edge_id)
                node = parent
            path.reverse()
            return path

        max_steps = max((len(v.trips) for v in vehicles), default=0)
        cumulative = {v.id: 0.0 for v in vehicles}
        for step in range(max_steps):
            paths: dict[int, list[int]] = {}
            flows: dict[int, int] = {}
            for vehicle in vehicles:
                if step >= len(vehicle.trips):
                    continue
                trip = vehicle.trips[step]
                path = shortest(trip.origin, trip.destination)
                paths[vehicle.id] = path
                for eid in path:
                    flows[eid] = flows.get(eid, 0) + 1

            for vehicle in vehicles:
                for eid in paths.get(vehicle.id, []):
                    edge = edge_by_id[eid]
                    flow = max(1, flows[eid])
                    price = max(
                        self.config.charging_floor_price,
                        self.config.charging_overhead / flow + self.config.charging_fixed_cost,
                    )
                    cumulative[vehicle.id] += (
                        self.config.energy_rate_kwh_per_km * edge.length_km * price
                    )
        return cumulative

    def _feasible_normal(
        self,
        mean: float,
        std: float,
        minimum: float,
        low: float,
        high: float,
    ) -> float:
        for _ in range(self.config.budget_resample_limit):
            value = self.rng.normal(mean, max(std, 1e-9))
            if minimum <= value <= high:
                return float(max(low, value))
        # Deterministic fallback still respects the generated-normal design's
        # lower feasibility bound.
        return float(np.clip(max(mean, minimum), low, high))

    def _edge(self, edge_id: int, source: int, destination: int) -> Edge:
        c = self.config
        return Edge(
            id=edge_id,
            source=source,
            destination=destination,
            length_km=self._normal(
                c.road_min_length_km,
                c.road_length_std_km,
                c.road_min_length_km,
                c.road_max_length_km,
            ),
            capacity_vph=self._normal(
                c.capacity_min_vph,
                c.capacity_std_vph,
                c.capacity_min_vph,
                c.capacity_max_vph,
            ),
        )

    def _normal(self, mean: float, std: float, low: float, high: float) -> float:
        return float(np.clip(self.rng.normal(mean, max(std, 1e-9)), low, high))

    def _normal_int(self, mean: float, std: float, low: int, high: int) -> int:
        return int(np.clip(np.rint(self.rng.normal(mean, max(std, 1e-9))), low, high))

    def _validate_config(self) -> None:
        c = self.config
        if c.graph_nodes < 3:
            raise ValueError("graph_nodes must be >= 3")
        if c.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if c.min_trips_per_vehicle < 1 or c.max_trips_per_vehicle < c.min_trips_per_vehicle:
            raise ValueError("Invalid trip range")
        if c.budget_feasibility_multiplier < 1.0:
            raise ValueError("budget_feasibility_multiplier must be >= 1")
        if c.budget_resample_limit < 1:
            raise ValueError("budget_resample_limit must be >= 1")

    def save(self, problem: ProblemDefinition, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": 3,
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
