from types import SimpleNamespace

from nash_drl.data import Action
from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel
from nash_drl.environment.sumo import SumoConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from nash_drl.routing import DijkstraMapper


def test_episode_horizon_is_max_trip_count(tmp_path, monkeypatch):
    graph = DirectedGraph(3, (Edge(0,0,1,1,100), Edge(1,1,2,1,100), Edge(2,2,0,1,100)))
    problem = ProblemDefinition(graph, [
        Vehicle(0, [Trip(0,1), Trip(1,2)], 50, 100),
        Vehicle(1, [Trip(0,1)], 50, 100),
    ])
    env = NashSUMOTrainingEnvironment(
        problem,
        SumoTrainingEnvironmentConfig(SumoConfig(), EnvironmentConfig()),
        RewardModel(RewardConfig()),
        DijkstraMapper(),
        output_root=tmp_path,
    )
    assert env.max_steps == 2
