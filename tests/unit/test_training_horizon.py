from types import SimpleNamespace
from pathlib import Path

from nash_drl.data import Action
from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel
from nash_drl.environment.sumo import SumoConfig, SumoScenario, SumoScenarioBuilder
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from nash_drl.routing import DijkstraMapper


class FakeSession:
    def __init__(self, scenario, config, *, use_gui=False, label="test"):
        self.t = 0.0
        self.closed = False

    def start(self):
        pass

    def current_time(self):
        return self.t

    def snapshot_edge_flow(self, edge_count):
        import torch
        return torch.zeros(edge_count, dtype=torch.float32)

    def close(self):
        self.closed = True


def test_episode_horizon_is_max_trip_count(tmp_path, monkeypatch):
    graph = DirectedGraph(3, (Edge(0,0,1,1,100), Edge(1,1,2,1,100), Edge(2,2,0,1,100)))
    problem = ProblemDefinition(graph, [
        Vehicle(0, [Trip(0,1), Trip(1,2)], 50, 100),
        Vehicle(1, [Trip(0,1)], 50, 100),
    ])
    def fake_build_network(self, directory):
        from nash_drl.environment.sumo import SumoScenario
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return SumoScenario(
            path, path / "network.net.xml", path / "routes.rou.xml",
            path / "simulation.sumocfg", {}
        )

    monkeypatch.setattr(SumoScenarioBuilder, "build_network", fake_build_network)
    monkeypatch.setattr("nash_drl.environment.sumo_training.SumoTrafficSession", FakeSession)

    env = NashSUMOTrainingEnvironment(
        problem,
        SumoTrainingEnvironmentConfig(SumoConfig(), EnvironmentConfig()),
        RewardModel(RewardConfig()),
        DijkstraMapper(),
        output_root=tmp_path,
    )
    assert env.max_steps == 2


def test_make_state_handles_completed_vehicle_without_index_error(tmp_path, monkeypatch):
    graph = DirectedGraph(3, (
        Edge(0, 0, 1, 1, 100),
        Edge(1, 1, 2, 1, 100),
    ))
    problem = ProblemDefinition(graph, [
        Vehicle(0, [Trip(0, 1)], 50, 100),
        Vehicle(1, [Trip(0, 2), Trip(2, 1)], 50, 100),
    ])
    def fake_build_network(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return SumoScenario(
            path, path / "network.net.xml", path / "routes.rou.xml",
            path / "simulation.sumocfg", {}
        )

    monkeypatch.setattr(SumoScenarioBuilder, "build_network", fake_build_network)
    monkeypatch.setattr("nash_drl.environment.sumo_training.SumoTrafficSession", FakeSession)

    env = NashSUMOTrainingEnvironment(
        problem,
        SumoTrainingEnvironmentConfig(SumoConfig(), EnvironmentConfig()),
        RewardModel(RewardConfig()),
        DijkstraMapper(),
        output_root=tmp_path,
    )
    env.reset(episode_index=0)

    # Simulate vehicle 0 having completed its only trip. Its trip index now
    # equals len(trips), which must remain valid for the global state builder.
    env.problem.vehicles[0].current_trip_index = 1
    env.problem.vehicles[0].current_node = 1

    state = env._make_state()

    assert int(state.current_nodes[0]) == 1
    assert int(state.next_destinations[0]) == 1
    assert int(state.final_destinations[0]) == 1
