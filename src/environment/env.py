from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from data import GlobalState, Paths
from domain import ProblemDefinition
from features.agent_features import encode_agent_features

from .energy_cost import edge_energy_cost
from .reward import RewardModel, RewardResult
from .simulator import RoadSimulator, SimulationState
from .travel_time import edge_travel_time


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    energy_rate_kwh_per_km: float = 1.0
    charging_overhead: float = 6.0
    charging_fixed_cost: float = 1.0
    charging_floor_price: float = 0.0
    congestion_alpha: float = 0.15
    congestion_beta: float = 4.0
    max_steps: int = 100


class NashEnvironment:
    """Environment boundary for the mathematical road/vehicle model."""

    def __init__(self, problem: ProblemDefinition, config: EnvironmentConfig, reward_model: RewardModel) -> None:
        self.problem = problem
        self.config = config
        self.reward_model = reward_model
        self.simulator: RoadSimulator | None = None
        self.state: GlobalState | None = None
        self._simulation_state: SimulationState | None = None
        self.step_count = 0

    def reset(self) -> GlobalState:
        from data import CSRMap

        csr = CSRMap.from_graph(self.problem.graph)
        self.simulator = RoadSimulator()
        self._simulation_state = SimulationState(csr_map=csr, vehicles=self.problem.vehicles)
        self.step_count = 0

        current_nodes = torch.tensor([int(v.current_node or 0) for v in self.problem.vehicles], dtype=torch.long)
        next_destinations = torch.tensor(
            [int(v.next_destination) if v.trips else 0 for v in self.problem.vehicles], dtype=torch.long
        )
        final_destinations = torch.tensor(
            [int(v.final_destination) if v.trips else 0 for v in self.problem.vehicles], dtype=torch.long
        )
        features = encode_agent_features(self.problem.vehicles, csr.num_nodes, dtype=torch.float32)
        self.state = GlobalState(
            csr_map=csr,
            agent_features=features,
            edge_flow=csr.edge_flow,
            current_nodes=current_nodes,
            next_destinations=next_destinations,
            final_destinations=final_destinations,
        )
        self.state.validate()
        return self.state

    def step(self, paths: Paths) -> tuple[GlobalState, RewardResult, bool, dict[str, object]]:
        if self.state is None or self.simulator is None or self._simulation_state is None:
            raise RuntimeError("Environment must be reset before step().")
        if len(paths.edge_ids) != len(self.problem.vehicles):
            raise ValueError("One path is required per vehicle")

        self._simulation_state = self.simulator.apply_paths(self._simulation_state, paths)
        self.state.edge_flow = self.state.csr_map.edge_flow

        n = len(self.problem.vehicles)
        travel_times = torch.zeros(n, dtype=torch.float32)
        costs = torch.zeros(n, dtype=torch.float32)
        for i, vehicle in enumerate(self.problem.vehicles):
            path = paths.edge_ids[i]
            if not path:
                continue
            speed = torch.full((len(path),), float(vehicle.free_flow_speed_kmh))
            edge_ids = torch.tensor(path, dtype=torch.long)
            lengths = self.state.csr_map.edge_len[edge_ids]
            caps = self.state.csr_map.edge_cap[edge_ids]
            flows = self.state.csr_map.edge_flow[edge_ids]
            travel_times[i] = edge_travel_time(
                lengths, speed, flows, caps, self.config.congestion_alpha, self.config.congestion_beta
            ).sum()
            costs[i] = edge_energy_cost(
                lengths,
                flows,
                self.config.energy_rate_kwh_per_km,
                self.config.charging_overhead,
                self.config.charging_fixed_cost,
                self.config.charging_floor_price,
            ).sum()

        budgets = torch.tensor([float(v.remaining_budget or v.budget) for v in self.problem.vehicles])
        reward = self.reward_model.compute(travel_times, costs, budgets)
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        return self.state, reward, done, {"paths": paths}
