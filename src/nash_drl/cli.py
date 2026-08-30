from __future__ import annotations

import argparse
from pathlib import Path

from nash_drl.config import load_yaml
from nash_drl.training.experiment import run_training
from nash_drl.evaluation.runner import EvaluationRunner
from nash_drl.environment.simulation import SimulationRunner


def _bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")



def build_demo_problem():
    from nash_drl.domain import DirectedGraph, Edge, ProblemDefinition, Trip, Vehicle
    edges = (
        Edge(0, 0, 1, 10.0, 100.0), Edge(1, 1, 3, 20.0, 80.0),
        Edge(2, 0, 2, 15.0, 120.0), Edge(3, 2, 3, 25.0, 60.0), Edge(4, 1, 2, 5.0, 50.0),
    )
    return ProblemDefinition(DirectedGraph(4, edges), [
        Vehicle(0, [Trip(0, 3)], 60.0, 150.0),
        Vehicle(1, [Trip(0, 3)], 30.0, 150.0),
    ])


def build_trainer(config: dict | None = None):
    from nash_drl.environment import EnvironmentConfig, NashEnvironment, RewardConfig, RewardModel
    from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork
    from nash_drl.routing import DijkstraMapper
    from nash_drl.training import NashDRLTrainer, TrainingConfig
    cfg = config or {}
    problem = build_demo_problem()
    env = NashEnvironment(problem, EnvironmentConfig(**cfg.get("environment", {})), RewardModel(RewardConfig(**cfg.get("reward", {}))))
    state = env.reset()
    f, e = int(state.agent_features.shape[1]), state.csr_map.num_edges
    net = cfg.get("network", {})
    actor = ActorNetwork(f, e, hidden_dim=int(net.get("hidden_dim", 32)), deep_set_dim=int(net.get("deep_set_dim", 64)), hidden_layers=int(net.get("actor_hidden_layers", 4)))
    critic = CriticNetwork(f, e, hidden_dim=int(net.get("hidden_dim", 32)), deep_set_dim=int(net.get("deep_set_dim", 64)), hidden_layers=int(net.get("critic_hidden_layers", 4)))
    target = TargetCriticNetwork(critic)
    train_raw = cfg.get("training", {})
    allowed = {f.name for f in __import__("dataclasses").fields(TrainingConfig)}
    train_cfg = TrainingConfig(**{k: v for k, v in train_raw.items() if k in allowed})
    return NashDRLTrainer(env, actor, critic, target, DijkstraMapper(), train_cfg)

def simulate() -> None:
    parser = argparse.ArgumentParser(description="Run NashDRL SUMO simulation without DRL learning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["mock", "real"])
    parser.add_argument("--visualization", type=_bool)
    parser.add_argument("--report", type=_bool, dest="generate_analytic_report")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config.setdefault("simulation", {})
    if args.mode:
        config["simulation"]["mode"] = args.mode
    if args.visualization is not None:
        config["simulation"]["visualization"] = args.visualization
    if args.generate_analytic_report is not None:
        config["simulation"]["generate_analytic_report"] = args.generate_analytic_report
    result = SimulationRunner(config).run()
    print("Simulation completed")
    print(f"dataset: {result.dataset_path}")
    print(f"trace:   {result.raw_trace}")
    print(f"report:  {result.analytic_report or 'disabled'}")
    print(f"SUMO:    {result.scenario_dir}")


def train() -> None:
    parser = argparse.ArgumentParser(description="Train NashDRL Actor/Critic networks over episodic SUMO trips")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["mock", "real"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--visualization", type=_bool)
    args = parser.parse_args()
    config = load_yaml(args.config)
    config.setdefault("training", {})
    if args.mode:
        config["training"]["mode"] = args.mode
    if args.episodes is not None:
        config["training"]["episodes"] = args.episodes
    if args.visualization is not None:
        config["training"]["visualization"] = args.visualization
    result = run_training(config, mode=args.mode)
    print("Training completed")
    print(f"dataset:          {result['dataset_path']}")
    print(f"output directory: {result['output_dir']}")
    print(f"episodes:         {len(result['episodes'])}")
    print(f"steps/episode:    {result['metadata']['steps_per_episode']}")


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Test a trained NashDRL policy")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["mock", "real"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config.setdefault("evaluation", {})
    if args.mode:
        config["evaluation"]["mode"] = args.mode
    if args.episodes is not None:
        config["evaluation"]["episodes"] = args.episodes
    results = EvaluationRunner(config, mode=args.mode, checkpoint=args.checkpoint).run()
    for result in results:
        print(result)


def infer() -> None:
    evaluate()


def visualize() -> None:
    parser = argparse.ArgumentParser(description="Run simulation with visualization enabled")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["mock", "real"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg.setdefault("simulation", {})["visualization"] = True
    if args.mode:
        cfg["simulation"]["mode"] = args.mode
    result = SimulationRunner(cfg).run()
    print(result.raw_trace)
