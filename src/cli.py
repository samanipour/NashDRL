from __future__ import annotations

import argparse

import torch

from config import load_yaml
from data import CSRMap
from domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle
from environment import EnvironmentConfig, NashEnvironment, RewardConfig, RewardModel
from models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from routing import DijkstraMapper
from training import NashDRLTrainer, TrainingConfig


def build_demo_problem() -> ProblemDefinition:
    edges = (
        Edge(0, 0, 1, 10.0, 100.0),
        Edge(1, 1, 3, 20.0, 80.0),
        Edge(2, 0, 2, 15.0, 120.0),
        Edge(3, 2, 3, 25.0, 60.0),
        Edge(4, 1, 2, 5.0, 50.0),
    )
    graph = DirectedGraph(num_nodes=4, edges=edges)
    vehicles = [
        Vehicle(0, [Trip(0, 3)], 60.0, 150.0),
        Vehicle(1, [Trip(0, 3)], 30.0, 150.0),
    ]
    return ProblemDefinition(graph=graph, vehicles=vehicles)


def build_trainer(config: dict) -> NashDRLTrainer:
    problem = build_demo_problem()
    env_cfg = EnvironmentConfig(**config.get("environment", {}))
    reward_cfg = RewardConfig(**config.get("reward", {}))
    env = NashEnvironment(problem, env_cfg, RewardModel(reward_cfg))
    state = env.reset()
    f = int(state.agent_features.shape[1])
    e = state.csr_map.num_edges
    net_cfg = config.get("network", {})
    actor = ActorNetwork(f, e, hidden_dim=int(net_cfg.get("hidden_dim", 32)), deep_set_dim=int(net_cfg.get("deep_set_dim", 64)), hidden_layers=int(net_cfg.get("actor_hidden_layers", 4)))
    critic = CriticNetwork(f, e, hidden_dim=int(net_cfg.get("hidden_dim", 32)), deep_set_dim=int(net_cfg.get("deep_set_dim", 64)), hidden_layers=int(net_cfg.get("critic_hidden_layers", 4)))
    target = TargetCriticNetwork(critic)
    train_cfg = TrainingConfig(**config.get("training", {}))
    return NashDRLTrainer(env, actor, critic, target, DijkstraMapper(), train_cfg)


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    trainer = build_trainer(cfg)
    for step in range(args.steps):
        metrics = trainer.train_step()
        print(step, metrics)


def evaluate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    trainer = build_trainer(load_yaml(args.config))
    state = trainer.env.reset()
    action = trainer.select_action(state, training=False)
    paths = trainer.mapper.map(action, state)
    _, reward, _, _ = trainer.env.step(paths)
    print({"total_reward": float(reward.total_reward), "violations": int(reward.budget_violations.sum())})


def infer() -> None:
    evaluate()
