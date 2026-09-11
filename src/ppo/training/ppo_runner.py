from __future__ import annotations

import csv
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

from nash_drl.config import load_yaml
from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel, build_reward_config
from nash_drl.environment.sumo import SumoConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from ppo.models.ppo import PPOActorCritic
from nash_drl.routing import DijkstraMapper
from nash_drl.utils.seeding import seed_everything

from ppo.training.ppo_trainer import PPOTrainer, PPOTrainingConfig


class PPOTrainingRunner:
    """Build and run the PPO baseline against the same dataset/environment."""

    def __init__(self, config: dict[str, Any], *, mode: str | None = None) -> None:
        self.config = config
        self.mode_override = mode

    def run(self) -> dict[str, Any]:
        raw = self.config.get("ppo", self.config.get("training", {}))
        seed = int(raw.get("seed", self.config.get("project", {}).get("seed", 42)))
        seed_everything(seed)
        mode = (self.mode_override or raw.get("mode") or self.config.get("simulation", {}).get("mode") or "mock").lower()
        output_dir = Path(raw.get("output_dir", "outputs/ppo_training"))
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = self._prepare_dataset(mode, seed, output_dir)
        problem, _ = load_problem(dataset_path)
        max_steps = max((len(v.trips) for v in problem.vehicles), default=0)
        if max_steps < 1:
            raise ValueError("Training dataset contains no vehicle trips")

        env_cfg = EnvironmentConfig(**self.config.get("environment", {}))
        reward_cfg = build_reward_config(self.config, "ppo")
        benchmark_reward_cfg = build_reward_config(self.config, "common")
        sumo_raw = dict(self.config.get("simulation", {}).get("sumo", {}))
        sumo_raw["seed"] = int(sumo_raw.get("seed", seed))
        sumo_cfg = SumoConfig(**sumo_raw)
        env = NashSUMOTrainingEnvironment(
            problem,
            SumoTrainingEnvironmentConfig(sumo=sumo_cfg, environment=env_cfg),
            RewardModel(reward_cfg, profile_name="ppo"),
            DijkstraMapper(),
            benchmark_reward_model=RewardModel(benchmark_reward_cfg, profile_name="common"),
            output_root=output_dir / "sumo_runs",
            use_gui=bool(raw.get("visualization", False)),
        )
        state = env.reset(episode_index=0)
        f, e = int(state.agent_features.shape[1]), state.csr_map.num_edges
        net = self.config.get("network", {})
        ppo = self.config.get("ppo_network", {})
        policy = PPOActorCritic(
            f,
            e,
            hidden_dim=int(ppo.get("hidden_dim", net.get("hidden_dim", 64))),
            deep_set_dim=int(ppo.get("deep_set_dim", net.get("deep_set_dim", 64))),
            hidden_layers=int(ppo.get("hidden_layers", 2)),
            deep_set_hidden_layers=int(ppo.get("deep_set_hidden_layers", 2)),
            initial_log_std=float(ppo.get("initial_log_std", -0.5)),
            min_log_std=float(ppo.get("min_log_std", -5.0)),
            max_log_std=float(ppo.get("max_log_std", 1.0)),
        )
        allowed = {f.name for f in fields(PPOTrainingConfig)}
        train_values = {k: v for k, v in raw.items() if k in allowed}
        ppo_cfg = PPOTrainingConfig(**train_values)
        trainer = PPOTrainer(env, policy, DijkstraMapper(), ppo_cfg)

        all_steps: list[dict[str, Any]] = []
        all_vehicles: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        try:
            for episode in range(ppo_cfg.episodes):
                step_rows, vehicle_rows, edge_rows, episode_row = trainer.train_episode(episode)
                all_steps.extend(step_rows)
                all_vehicles.extend(vehicle_rows)
                all_edges.extend(edge_rows)
                episodes.append(episode_row)
                if ppo_cfg.save_checkpoints and (episode + 1) % ppo_cfg.checkpoint_interval == 0:
                    trainer.save_checkpoint(str(output_dir / "checkpoints" / f"episode_{episode + 1:05d}.pt"))
        finally:
            env.close()

        self._write_csv(output_dir / "step_results.csv", all_steps)
        self._write_csv(output_dir / "vehicle_results.csv", all_vehicles)
        self._write_csv(output_dir / "edge_results.csv", all_edges)
        self._write_csv(output_dir / "episode_results.csv", episodes)
        metadata = {
            "algorithm": "ppo",
            "mode": mode,
            "seed": seed,
            "dataset_path": str(dataset_path),
            "episodes": ppo_cfg.episodes,
            "steps_per_episode": max_steps,
            "num_vehicles": len(problem.vehicles),
            "num_nodes": problem.graph.num_nodes,
            "num_edges": problem.graph.num_edges,
            "trip_counts": {str(v.id): len(v.trips) for v in problem.vehicles},
            "reward_profile": "ppo",
            "learning_reward": {"travel_time_weight": reward_cfg.travel_time_weight, "charging_cost_weight": reward_cfg.charging_cost_weight, "budget_penalty": reward_cfg.budget_penalty},
            "benchmark_reward": {"travel_time_weight": benchmark_reward_cfg.travel_time_weight, "charging_cost_weight": benchmark_reward_cfg.charging_cost_weight, "budget_penalty": benchmark_reward_cfg.budget_penalty},
        }
        (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        from nash_drl.visualization.training import plot_training_history
        plot_training_history(episodes, output_dir / "plots")
        return {
            "output_dir": output_dir,
            "episodes": episodes,
            "steps": all_steps,
            "vehicle_rows": all_vehicles,
            "edge_rows": all_edges,
            "metadata": metadata,
            "dataset_path": dataset_path,
        }

    def _prepare_dataset(self, mode: str, seed: int, output_dir: Path) -> Path:
        if mode == "mock":
            mock = dict(self.config.get("mock_data", {}))
            mock["seed"] = int(mock.get("seed", seed))
            cfg = MockDataConfig(**mock)
            path = Path("datasets/generated") / f"mock_seed_{cfg.seed}" / "dataset.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            MockDatasetGenerator(cfg).save(MockDatasetGenerator(cfg).generate(), path)
            shutil.copy2(path, output_dir / "dataset.json")
            return path
        if mode == "real":
            raw_path = self.config.get("ppo", {}).get("dataset_path") or self.config.get("training", {}).get("dataset_path") or self.config.get("simulation", {}).get("dataset_path")
            if not raw_path:
                raise ValueError("Real PPO training requires training.dataset_path (or simulation.dataset_path)")
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            shutil.copy2(path, output_dir / "dataset.json")
            return path
        raise ValueError("PPO training mode must be 'mock' or 'real'")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
