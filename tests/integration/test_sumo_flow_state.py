from pathlib import Path

import torch

import nash_drl.environment.sumo_training as st
from nash_drl.config import load_yaml
from nash_drl.environment.reward import RewardConfig, RewardModel
from nash_drl.environment.sumo import SumoScenario, SumoTripResult
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from nash_drl.cli import build_demo_problem
from nash_drl.routing import DijkstraMapper


class FakeSession:
    def __init__(self, scenario, config, *, use_gui=False, label="test"):
        self.scenario = scenario
        self.config = config
        self.t = 0.0
        self.closed = False
        self.snapshots = [torch.tensor([2.0, 0.0, 1.0, 0.0, 0.0]), torch.tensor([1.0, 3.0, 0.0, 2.0, 0.0])]
        self.index = 0

    def start(self):
        self.closed = False

    def current_time(self):
        return self.t

    def snapshot_edge_flow(self, edge_count):
        x = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        return x[:edge_count].clone()

    def execute_routes(self, routes, *, step_index, max_sim_steps=None):
        self.t += 10.0
        self.index = min(self.index + 1, len(self.snapshots) - 1)
        vehicle_metrics = [
            {
                "vehicle_id": vid,
                "travel_time_s": 10.0,
                "waiting_time_s": 0.0,
                "time_loss_s": 0.0,
                "distance_m": 1000.0,
            }
            for vid in routes
        ]
        return SumoTripResult(self.t, vehicle_metrics, [], None)

    def close(self):
        self.closed = True


def test_sumo_flow_is_loaded_into_state_and_carried_to_next_step(tmp_path, monkeypatch):
    problem = build_demo_problem()
    cfg = load_yaml("configs/experiments/small.yaml")
    cfg["simulation"]["sumo"]["end_time_s"] = 20
    env_cfg = EnvironmentConfig(**cfg["environment"])
    sumo_cfg = st.SumoConfig(**cfg["simulation"]["sumo"])

    def fake_build_network(self, directory):
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        return SumoScenario(p, p / "network.net.xml", p / "routes.rou.xml", p / "simulation.sumocfg", {})

    monkeypatch.setattr(st.SumoScenarioBuilder, "build_network", fake_build_network)
    monkeypatch.setattr(st, "SumoTrafficSession", FakeSession)

    env = NashSUMOTrainingEnvironment(
        problem,
        SumoTrainingEnvironmentConfig(sumo=sumo_cfg, environment=env_cfg),
        RewardModel(RewardConfig(**cfg["reward"])),
        DijkstraMapper(),
        output_root=tmp_path / "sumo_runs",
    )

    state0 = env.reset()
    assert torch.equal(state0.edge_flow, torch.tensor([2., 0., 1., 0., 0.]))
    assert state0.edge_flow_source == "sumo"
    assert state0.flow_time_s == 0.0

    action = st.Action(torch.ones((len(problem.vehicles), problem.graph.num_edges)))
    state1, _, _, info = env.step(action)
    assert torch.equal(state1.edge_flow, torch.tensor([1., 3., 0., 2., 0.]))
    assert state1.flow_time_s == 10.0
    assert info["edge_flow_t"] == [2.0, 0.0, 1.0, 0.0, 0.0]
    assert info["edge_flow_t1"] == [1.0, 3.0, 0.0, 2.0, 0.0]
    env.close()
