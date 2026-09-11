from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nash_drl.config import load_yaml
from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel, build_reward_config
from nash_drl.environment.sumo import SumoConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from nash_drl.routing import DijkstraMapper, BudgetAwareDijkstraMapper
from nash_drl.training.checkpoint import load_checkpoint
from nash_drl.training.trainer import NashDRLTrainer, TrainingConfig
from nash_drl.evaluation.evaluator import Evaluator
from nash_drl.visualization.training import plot_training_history


class EvaluationRunner:
    def __init__(self, config: dict[str, Any], *, mode: str | None = None, checkpoint: str | None = None) -> None:
        self.config = config
        self.mode_override = mode
        self.checkpoint_override = checkpoint

    def _dataset(self) -> tuple[Any, Path]:
        training = self.config.get("training", {})
        mode = (self.mode_override or training.get("mode") or "mock").lower()
        if mode == "mock":
            mock = MockDataConfig(**self.config.get("mock_data", {}))
            path = Path("datasets/generated") / f"mock_seed_{mock.seed}" / "dataset.json"
            MockDatasetGenerator(mock).save(MockDatasetGenerator(mock).generate(), path)
        else:
            path = Path(training.get("dataset_path") or self.config.get("simulation", {}).get("dataset_path"))
            if not path.exists():
                raise FileNotFoundError(path)
        return load_problem(path)[0], path

    def run(self) -> list:
        problem, dataset_path = self._dataset()
        state_features = 6
        edges = problem.graph.num_edges
        net = self.config.get("network", {})
        actor = ActorNetwork(state_features, edges, hidden_dim=int(net.get("hidden_dim", 32)), deep_set_dim=int(net.get("deep_set_dim", 64)), hidden_layers=int(net.get("actor_hidden_layers", 4)), interaction_coupling_ratio=float(net.get("interaction_coupling_ratio", 0.9)))
        critic = CriticNetwork(state_features, edges, hidden_dim=int(net.get("hidden_dim", 32)), deep_set_dim=int(net.get("deep_set_dim", 64)), hidden_layers=int(net.get("critic_hidden_layers", 4)))
        target = TargetCriticNetwork(critic)
        checkpoint = self.checkpoint_override or self.config.get("evaluation", {}).get("checkpoint") or self.config.get("training", {}).get("checkpoint")
        if checkpoint:
            payload = load_checkpoint(checkpoint)
            actor.load_state_dict(payload["actor_state_dict"])
            critic.load_state_dict(payload["critic_state_dict"])
            target.load_state_dict(payload["target_critic_state_dict"])
        env = NashSUMOTrainingEnvironment(
            problem,
            SumoTrainingEnvironmentConfig(
                sumo=SumoConfig(**self.config.get("simulation", {}).get("sumo", {})),
                environment=EnvironmentConfig(**self.config.get("environment", {})),
            ),
            RewardModel(build_reward_config(self.config, self.config.get("evaluation", {}).get("reward_profile", "common"))),
            BudgetAwareDijkstraMapper(
                energy_rate_kwh_per_km=float(self.config.get("environment", {}).get("energy_rate_kwh_per_km", 1.0)),
                charging_overhead=float(self.config.get("environment", {}).get("charging_overhead", 6.0)),
                charging_fixed_cost=float(self.config.get("environment", {}).get("charging_fixed_cost", 1.0)),
                charging_floor_price=float(self.config.get("environment", {}).get("charging_floor_price", 0.0)),
            ),
            output_root=Path(self.config.get("evaluation", {}).get("output_dir", "outputs/evaluation")) / "sumo_runs",
            use_gui=bool(self.config.get("evaluation", {}).get("visualization", False)),
        )
        train_cfg = TrainingConfig(episodes=1, device=self.config.get("training", {}).get("device", "cpu"))
        trainer = NashDRLTrainer(env, actor, critic, target, BudgetAwareDijkstraMapper(energy_rate_kwh_per_km=float(self.config.get("environment", {}).get("energy_rate_kwh_per_km", 1.0)), charging_overhead=float(self.config.get("environment", {}).get("charging_overhead", 6.0)), charging_fixed_cost=float(self.config.get("environment", {}).get("charging_fixed_cost", 1.0)), charging_floor_price=float(self.config.get("environment", {}).get("charging_floor_price", 0.0))), train_cfg)
        episodes = int(self.config.get("evaluation", {}).get("episodes", 1))
        output = Path(self.config.get("evaluation", {}).get("output_dir", "outputs/evaluation"))
        results = Evaluator(trainer).run(episodes, output)
        history = [{
            "episode": r.episode,
            "total_reward": r.total_reward,
            "actor_loss_mean": 0.0,
            "critic_loss_mean": 0.0,
            "budget_violations": r.budget_violations,
            "total_travel_time_s": r.total_travel_time_s,
            "total_charging_cost": r.total_charging_cost,
        } for r in results]
        plot_training_history(history, output / "plots")
        (output / "evaluation_metadata.json").write_text(
            __import__("json").dumps({"dataset_path": str(dataset_path), "episodes": episodes}, indent=2),
            encoding="utf-8",
        )
        return results
