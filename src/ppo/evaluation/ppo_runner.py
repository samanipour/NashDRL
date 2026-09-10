from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel
from nash_drl.environment.sumo import SumoConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from ppo.models import PPOActorCritic
from nash_drl.routing import DijkstraMapper
from nash_drl.training.checkpoint import load_checkpoint
from ppo.evaluation.ppo_evaluator import PPOEvaluator


class PPOEvaluationRunner:
    def __init__(self, config: dict[str, Any], *, mode: str | None = None, checkpoint: str | None = None) -> None:
        self.config = config
        self.mode_override = mode
        self.checkpoint_override = checkpoint

    def _dataset(self) -> tuple[Any, Path]:
        evaluation = self.config.get("evaluation", {})
        mode = (self.mode_override or evaluation.get("mode") or self.config.get("ppo", {}).get("mode") or "mock").lower()
        if mode == "mock":
            mock = MockDataConfig(**self.config.get("mock_data", {}))
            path = Path("datasets/generated") / f"mock_seed_{mock.seed}" / "dataset.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            MockDatasetGenerator(mock).save(MockDatasetGenerator(mock).generate(), path)
        else:
            path = Path(evaluation.get("dataset_path") or self.config.get("ppo", {}).get("dataset_path") or self.config.get("training", {}).get("dataset_path") or self.config.get("simulation", {}).get("dataset_path"))
            if not path.exists():
                raise FileNotFoundError(path)
        return load_problem(path)[0], path

    def run(self) -> list[PPOEvaluationResult]:
        from ppo.evaluation.ppo_evaluator import PPOEvaluationResult
        problem, dataset_path = self._dataset()
        state_features = 6
        edges = problem.graph.num_edges
        network = self.config.get("ppo_network", {})
        policy = PPOActorCritic(
            state_features,
            edges,
            hidden_dim=int(network.get("hidden_dim", 64)),
            deep_set_dim=int(network.get("deep_set_dim", 64)),
            hidden_layers=int(network.get("hidden_layers", 2)),
            deep_set_hidden_layers=int(network.get("deep_set_hidden_layers", 2)),
            initial_log_std=float(network.get("initial_log_std", -0.5)),
            min_log_std=float(network.get("min_log_std", -5.0)),
            max_log_std=float(network.get("max_log_std", 1.0)),
        )
        device = self.config.get("ppo", {}).get("device", "cpu")
        checkpoint = self.checkpoint_override or self.config.get("evaluation", {}).get("checkpoint") or self.config.get("ppo", {}).get("checkpoint")
        if checkpoint:
            payload = load_checkpoint(checkpoint, map_location=device)
            policy.load_state_dict(payload["policy_state_dict"])

        sumo_raw = dict(self.config.get("simulation", {}).get("sumo", {}))
        sumo_raw.setdefault("seed", int(self.config.get("project", {}).get("seed", 42)))
        env = NashSUMOTrainingEnvironment(
            problem,
            SumoTrainingEnvironmentConfig(
                sumo=SumoConfig(**sumo_raw),
                environment=EnvironmentConfig(**self.config.get("environment", {})),
            ),
            RewardModel(RewardConfig(**self.config.get("reward", {}))),
            DijkstraMapper(),
            output_root=Path(self.config.get("evaluation", {}).get("output_dir", "outputs/ppo_evaluation")) / "sumo_runs",
            use_gui=bool(self.config.get("evaluation", {}).get("visualization", False)),
        )
        output = Path(self.config.get("evaluation", {}).get("output_dir", "outputs/ppo_evaluation"))
        episodes = int(self.config.get("evaluation", {}).get("episodes", 5))
        try:
            results = [PPOEvaluator(env, policy, device=device).run_once(i) for i in range(episodes)]
        finally:
            env.close()
        with (output / "test_results.csv").open("w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=list(PPOEvaluationResult.__dataclass_fields__))
            writer.writeheader()
            for r in results:
                writer.writerow({name: getattr(r, name) for name in PPOEvaluationResult.__dataclass_fields__})
        (output / "evaluation_metadata.json").write_text(json.dumps({"algorithm": "ppo", "dataset_path": str(dataset_path), "episodes": episodes}, indent=2), encoding="utf-8")
        return results
