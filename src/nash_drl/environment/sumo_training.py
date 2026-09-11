from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import torch

from nash_drl.data import Action, CSRMap, GlobalState, Paths
from nash_drl.domain import ProblemDefinition, Vehicle
from nash_drl.features.agent_features import encode_agent_features
from nash_drl.routing import ActionToPathMapper

from .reward import RewardModel, RewardResult
from .env import EnvironmentConfig
from .sumo import SumoConfig, SumoScenarioBuilder, SumoTrafficSession, run_sumo_trip


@dataclass(slots=True)
class SUMOStepResult:
    step_index: int
    sumo_time_s: float
    vehicle_metrics: list[dict[str, Any]]
    edge_metrics: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SumoTrainingEnvironmentConfig:
    sumo: SumoConfig
    environment: EnvironmentConfig


class NashSUMOTrainingEnvironment:
    """Episode environment that executes exactly one trip-set leg per step in SUMO.

    Every episode reuses the same problem dataset, while mutable vehicle state is
    reset. At step t, the Actor/mapper chooses one path for each unfinished vehicle
    from its current stop to its current trip destination. SUMO executes those paths
    to completion. The environment then advances each completed vehicle to the next
    trip and exposes the resulting state.
    """

    def __init__(
        self,
        problem: ProblemDefinition,
        config: Any,
        reward_model: RewardModel,
        mapper: ActionToPathMapper,
        *,
        output_root: str | Path,
        use_gui: bool = False,
    ) -> None:
        self.original_problem = deepcopy(problem)
        self.problem = deepcopy(problem)
        self.config = config
        self.reward_model = reward_model
        self.mapper = mapper
        self.output_root = Path(output_root)
        self.use_gui = use_gui
        self.episode_index = 0
        self.step_index = 0
        self.state: GlobalState | None = None
        self.csr: CSRMap | None = None
        self.cumulative_costs: dict[int, float] = {}
        self.cumulative_travel_times: dict[int, float] = {}
        self.sumo_scenario = None
        self.sumo_session: SumoTrafficSession | None = None
        self.flow_history: dict[int, torch.Tensor] = {}

    @property
    def num_agents(self) -> int:
        return len(self.problem.vehicles)

    @property
    def num_edges(self) -> int:
        return self.problem.graph.num_edges

    @property
    def max_steps(self) -> int:
        return max((len(v.trips) for v in self.problem.vehicles), default=0)

    def reset(self, *, episode_index: int = 0) -> GlobalState:
        # A training environment owns exactly one persistent SUMO session per
        # episode. Ensure a previous episode cannot leak its TraCI connection.
        self.close()
        self.episode_index = episode_index
        self.step_index = 0
        self.problem = deepcopy(self.original_problem)
        self.cumulative_costs = {v.id: 0.0 for v in self.problem.vehicles}
        self.cumulative_travel_times = {v.id: 0.0 for v in self.problem.vehicles}

        self.csr = CSRMap.from_graph(self.problem.graph)
        episode_dir = self.output_root / f"episode_{episode_index:05d}"
        if episode_dir.exists() and self.step_index == 0:
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        # Build the network once per episode; route files are replaced per step.
        self.sumo_scenario = SumoScenarioBuilder(self.problem, self.config.sumo).build_network(episode_dir / "network")
        self.sumo_session = SumoTrafficSession(
            self.sumo_scenario,
            self.config.sumo,
            use_gui=self.use_gui,
            label=f"nash_drl_episode_{self.episode_index}",
        )
        self.sumo_session.start()

        # t=0 traffic-flow snapshot comes from the live SUMO simulation.  This
        # snapshot is the edge-flow component of the first neural-network state.
        initial_flow = self.sumo_session.snapshot_edge_flow(self.num_edges)
        self.csr.edge_flow = initial_flow.clone()
        self.flow_history[0] = initial_flow.clone()
        self.state = self._make_state(flow_time_s=self.sumo_session.current_time())
        self.state.validate()
        return self.state

    def _make_state(self, *, flow_time_s: float = 0.0) -> GlobalState:
        assert self.csr is not None

        def current_node_for_state(vehicle: Vehicle) -> int:
            # A vehicle may have completed its entire trip-set, so
            # current_trip_index can legitimately equal len(trips). Never
            # index trips using the completed index. Also do not use
            # ``current_node or ...`` because node 0 is a valid node ID.
            if vehicle.current_node is not None:
                return int(vehicle.current_node)
            if vehicle.trips and vehicle.current_trip_index < len(vehicle.trips):
                return int(vehicle.trips[vehicle.current_trip_index].origin)
            if vehicle.trips:
                return int(vehicle.trips[-1].destination)
            return 0
        current_nodes = torch.tensor(
            [current_node_for_state(v) for v in self.problem.vehicles],
            dtype=torch.long,
        )
        next_destinations = torch.tensor(
            [
                int(v.next_destination)
                if v.trips and v.current_trip_index < len(v.trips)
                else current_node_for_state(v)
                for v in self.problem.vehicles
            ],
            dtype=torch.long,
        )
        final_destinations = torch.tensor(
            [
                int(v.final_destination)
                if v.trips
                else current_node_for_state(v)
                for v in self.problem.vehicles
            ],
            dtype=torch.long,
        )
        features = encode_agent_features(
            self.problem.vehicles,
            self.csr.num_nodes,
            dtype=torch.float32,
            max_trips=max((len(v.trips) for v in self.problem.vehicles), default=1),
            max_budget=max((v.budget for v in self.problem.vehicles), default=1.0),
            max_speed_kmh=max((v.free_flow_speed_kmh for v in self.problem.vehicles), default=1.0),
        )
        remaining_budgets = torch.tensor(
            [float(v.remaining_budget if v.remaining_budget is not None else v.budget) for v in self.problem.vehicles],
            dtype=torch.float32,
        )
        active_mask = torch.tensor(
            [float(v.current_trip_index < len(v.trips)) for v in self.problem.vehicles],
            dtype=torch.float32,
        )
        return GlobalState(
            csr_map=self.csr,
            agent_features=features,
            edge_flow=self.csr.edge_flow,
            current_nodes=current_nodes,
            next_destinations=next_destinations,
            final_destinations=final_destinations,
            remaining_budgets=remaining_budgets,
            active_mask=active_mask,
            flow_time_s=flow_time_s,
            edge_flow_source="sumo",
        )

    def step(self, action: Action) -> tuple[GlobalState, RewardResult, bool, dict[str, Any]]:
        if self.state is None or self.sumo_scenario is None or self.csr is None or self.sumo_session is None:
            raise RuntimeError("Environment must be reset before step().")
        if action.edge_weights.shape != (self.num_agents, self.num_edges):
            raise ValueError(
                f"Expected action shape [{self.num_agents},{self.num_edges}], got {tuple(action.edge_weights.shape)}"
            )

        paths = self.mapper.map(action, self.state)
        active = [
            i for i, v in enumerate(self.problem.vehicles)
            if v.trips and v.current_trip_index < len(v.trips)
        ]
        # Trips whose origin equals their destination complete without a SUMO run.
        instant = [i for i in active if self.problem.vehicles[i].current_node == self.problem.vehicles[i].next_destination]
        active_for_sumo = [i for i in active if i not in instant]
        for i in range(self.num_agents):
            if i not in active_for_sumo:
                paths.edge_ids[i] = []

        # Advance zero-length trips immediately.
        for i in instant:
            v = self.problem.vehicles[i]
            v.current_node = v.next_destination

        active_routes = {self.problem.vehicles[i].id: paths.edge_ids[i] for i in active_for_sumo if paths.edge_ids[i]}
        if active_routes:
            # Continue below through SUMO for non-empty paths.
            pass
        elif not active_for_sumo:
            self.step_index += 1
            for i in instant:
                self.problem.vehicles[i].current_trip_index += 1
            done = self.step_index >= self.max_steps
            self.state = self._make_state(flow_time_s=self.sumo_session.current_time())
            agent_done = [bool(v.current_trip_index >= len(v.trips)) for v in self.problem.vehicles]
            if done:
                agent_done = [True] * self.num_agents
            return self.state, self._zero_reward(), done, {"paths": paths, "sumo": None, "agent_done": agent_done}

        step_dir = self.sumo_scenario.directory.parent / f"step_{self.step_index:03d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        # Refresh t-flow immediately before executing the selected routes.  This
        # is the authoritative SUMO traffic state available to the environment.
        flow_t = self.sumo_session.snapshot_edge_flow(self.num_edges)
        flow_time_t = self.sumo_session.current_time()
        self.csr.edge_flow = flow_t.clone()
        self.state.edge_flow = flow_t.clone()
        self.state.csr_map.edge_flow = flow_t.clone()
        self.flow_history[self.step_index] = flow_t.clone()

        trip_metrics = self.sumo_session.execute_routes(
            active_routes,
            step_index=self.step_index,
        ) if active_routes else None

        # Reward-model flow includes the observed background traffic at time t
        # plus the controlled routes selected for this decision.  The resulting
        # post-step SUMO flow is stored separately and becomes state t+1.
        planned_flow = torch.zeros(self.num_edges, dtype=torch.float32)
        for route in paths.edge_ids:
            for edge_id in route:
                planned_flow[edge_id] += 1.0
        edge_flow = flow_t + planned_flow

        n = self.num_agents
        travel_times = torch.zeros(n, dtype=torch.float32)
        charging_costs = torch.zeros(n, dtype=torch.float32)
        budgets = torch.zeros(n, dtype=torch.float32)
        vehicle_metrics: list[dict[str, Any]] = []
        metrics_by_id = {int(m["vehicle_id"]): m for m in (trip_metrics.vehicle_metrics if trip_metrics else [])}

        for i, vehicle in enumerate(self.problem.vehicles):
            budgets[i] = float(vehicle.remaining_budget if vehicle.remaining_budget is not None else vehicle.budget)
            if i not in active:
                vehicle_metrics.append({"vehicle_id": vehicle.id, "inactive": True})
                continue
            path = paths.edge_ids[i]
            telemetry = metrics_by_id.get(vehicle.id, {})
            edge_ids = torch.tensor(path, dtype=torch.long)
            model_travel_time_h = 0.0
            if len(path) > 0:
                lengths = self.csr.edge_len[edge_ids]
                flows = edge_flow[edge_ids]
                caps = self.csr.edge_cap[edge_ids]
                speeds = torch.full_like(lengths, float(vehicle.free_flow_speed_kmh))
                from .energy_cost import edge_energy_cost
                from .travel_time import edge_travel_time

                charging_costs[i] = edge_energy_cost(
                    lengths,
                    flows,
                    self.config.environment.energy_rate_kwh_per_km,
                    self.config.environment.charging_overhead,
                    self.config.environment.charging_fixed_cost,
                    self.config.environment.charging_floor_price,
                ).sum()
                model_travel_time_h = float(
                    edge_travel_time(
                        lengths,
                        speeds,
                        flows,
                        caps,
                        self.config.environment.congestion_alpha,
                        self.config.environment.congestion_beta,
                    ).sum()
                )

            # SUMO travel time is retained for diagnostics.  The NashDRL
            # learning reward can use the mathematical Eq. (5) travel time
            # through EnvironmentConfig.use_model_travel_time.
            sumo_travel_time_h = float(telemetry.get("travel_time_s", 0.0)) / 3600.0
            travel_times[i] = (
                model_travel_time_h if self.config.environment.use_model_travel_time
                else sumo_travel_time_h
            )

            self.cumulative_costs[vehicle.id] += float(charging_costs[i])
            self.cumulative_travel_times[vehicle.id] += float(travel_times[i])
            vehicle.remaining_budget = vehicle.budget - self.cumulative_costs[vehicle.id]
            vehicle.current_node = vehicle.trips[vehicle.current_trip_index].destination

            vehicle_metrics.append({
                "vehicle_id": vehicle.id,
                "trip_index": vehicle.current_trip_index,
                "origin_node": vehicle.trips[vehicle.current_trip_index].origin,
                "destination_node": vehicle.trips[vehicle.current_trip_index].destination,
                "travel_time_h": float(travel_times[i]),
                "travel_time_s": float(travel_times[i]) * 3600.0,
                "model_travel_time_h": model_travel_time_h,
                "model_travel_time_s": model_travel_time_h * 3600.0,
                "sumo_travel_time_h": sumo_travel_time_h,
                "sumo_travel_time_s": sumo_travel_time_h * 3600.0,
                "charging_cost": float(charging_costs[i]),
                "cumulative_travel_time_s": self.cumulative_travel_times[vehicle.id],
                "cumulative_charging_cost": self.cumulative_costs[vehicle.id],
                "remaining_budget": vehicle.remaining_budget,
                "sumo_waiting_time_s": float(telemetry.get("waiting_time_s", 0.0)),
                "sumo_time_loss_s": float(telemetry.get("time_loss_s", 0.0)),
                "route": " ".join(f"e{e}" for e in path),
            })

        reward = self.reward_model.compute(
            travel_times,
            charging_costs,
            budgets,
        )

        # Attach per-vehicle reward and hard-constraint outcome to report rows.
        for i, row in enumerate(vehicle_metrics):
            if "inactive" not in row:
                row["step_reward"] = float(reward.vehicle_rewards[i])
                row["hard_constraint_violation"] = bool(reward.budget_violations[i])
                row["budget_before_trip"] = float(budgets[i])
                row["budget_after_trip"] = float(self.problem.vehicles[i].remaining_budget or 0.0)

        # Enrich SUMO edge telemetry with model parameters.
        edge_by_id = {e.id: e for e in self.problem.graph.edges}
        for edge_row in (trip_metrics.edge_metrics if trip_metrics else []):
            eid = edge_row.get("entity_id", "")
            if isinstance(eid, str) and eid.startswith("e") and eid[1:].isdigit():
                edge = edge_by_id[int(eid[1:])]
                flow = float(edge_row.get("vehicle_count", 0.0))
                price = self.config.environment.charging_overhead / flow + self.config.environment.charging_fixed_cost if flow > 0 else 0.0
                edge_row.update({
                    "edge_id": edge.id,
                    "from_node": edge.source,
                    "to_node": edge.destination,
                    "length_km": edge.length_km,
                    "capacity_vph": edge.capacity_vph,
                    "charging_unit_price": max(self.config.environment.charging_floor_price, price) if flow > 0 else 0.0,
                    "congestion_ratio": flow / max(edge.capacity_vph, 1e-12),
                })

        # A trip leg is completed for every active vehicle in this step.
        for i in active:
            self.problem.vehicles[i].current_trip_index += 1
            if self.problem.vehicles[i].current_trip_index < len(self.problem.vehicles[i].trips):
                self.problem.vehicles[i].current_node = self.problem.vehicles[i].trips[self.problem.vehicles[i].current_trip_index].origin

        # Query SUMO once more after executing the current decision. This is the
        # authoritative t+1 traffic-flow snapshot and is used by the next state.
        flow_t1 = self.sumo_session.snapshot_edge_flow(self.num_edges)
        self.csr.edge_flow = flow_t1.clone()

        self.step_index += 1
        done = self.step_index >= self.max_steps
        self.flow_history[self.step_index] = flow_t1.clone()
        self.state = self._make_state(flow_time_s=self.sumo_session.current_time())
        agent_done = [
            bool(v.current_trip_index >= len(v.trips)) for v in self.problem.vehicles
        ]
        if done:
            agent_done = [True] * self.num_agents
        info = {
            "paths": paths,
            "vehicle_metrics": vehicle_metrics,
            "edge_metrics": trip_metrics.edge_metrics if trip_metrics else [],
            "sumo_time_s": float(trip_metrics.sumo_time_s) if trip_metrics else self.sumo_session.current_time(),
            "completed_trips": sum(min(self.step_index, len(v.trips)) for v in self.problem.vehicles),
            "agent_done": agent_done,
            "flow_time_t_s": float(flow_time_t),
            "flow_time_t1_s": float(self.state.flow_time_s),
            "edge_flow_t": flow_t.tolist(),
            "edge_flow_t1": flow_t1.tolist(),
        }
        return self.state, reward, done, info

    def close(self) -> None:
        """Close the persistent SUMO/TraCI session for the current episode."""
        if self.sumo_session is not None:
            self.sumo_session.close()
            self.sumo_session = None

    def _zero_reward(self) -> RewardResult:
        n = self.num_agents
        zeros = torch.zeros(n, dtype=torch.float32)
        return RewardResult(
            vehicle_rewards=zeros,
            total_reward=zeros.sum(),
            travel_times=zeros,
            charging_costs=zeros,
            budget_violations=torch.zeros(n, dtype=torch.bool),
        )
