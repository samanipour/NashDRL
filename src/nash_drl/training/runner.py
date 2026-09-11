from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


from nash_drl.config import load_yaml
from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem
from nash_drl.environment.env import EnvironmentConfig
from nash_drl.environment.reward import RewardConfig, RewardModel
from nash_drl.environment.sumo import SumoConfig
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment, SumoTrainingEnvironmentConfig
from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from nash_drl.routing import DijkstraMapper, BudgetAwareDijkstraMapper
from nash_drl.utils.seeding import seed_everything

from .trainer import NashDRLTrainer, TrainingConfig


class TrainingRunner:
    """Build the dataset, environment, networks, train episodes, and persist reports."""

    def __init__(self, config: dict[str, Any], *, mode: str | None = None) -> None:
        self.config = config
        self.mode_override = mode

    def run(self) -> dict[str, Any]:
        training_cfg_raw = self.config.get("training", {})
        seed = int(training_cfg_raw.get("seed", self.config.get("project", {}).get("seed", 42)))
        seed_everything(seed)
        mode = (self.mode_override or training_cfg_raw.get("mode") or self.config.get("simulation", {}).get("mode") or "mock").lower()
        output_dir = Path(training_cfg_raw.get("output_dir", "outputs/training"))
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = self._prepare_dataset(mode, seed, output_dir)
        problem, payload = load_problem(dataset_path)
        max_steps = max((len(v.trips) for v in problem.vehicles), default=0)
        if max_steps < 1:
            raise ValueError("Training dataset contains no vehicle trips")

        env_cfg = EnvironmentConfig(**self.config.get("environment", {}))
        reward_cfg = RewardConfig(**self.config.get("reward", {}))
        sumo_raw = dict(self.config.get("simulation", {}).get("sumo", {}))
        sumo_raw["seed"] = int(sumo_raw.get("seed", seed))
        sumo_cfg = SumoConfig(**sumo_raw)
        mapper = (
            BudgetAwareDijkstraMapper(
                energy_rate_kwh_per_km=env_cfg.energy_rate_kwh_per_km,
                charging_overhead=env_cfg.charging_overhead,
                charging_fixed_cost=env_cfg.charging_fixed_cost,
                charging_floor_price=env_cfg.charging_floor_price,
                max_repair_passes=int(training_cfg_raw.get("constraint_repair_passes", 2)),
            )
            if bool(training_cfg_raw.get("budget_constraints_enabled", True))
            else DijkstraMapper()
        )
        env = NashSUMOTrainingEnvironment(
            problem,
            SumoTrainingEnvironmentConfig(sumo=sumo_cfg, environment=env_cfg),
            RewardModel(reward_cfg),
            mapper,
            output_root=output_dir / "sumo_runs",
            use_gui=bool(training_cfg_raw.get("visualization", False)),
        )
        state = env.reset(episode_index=0)
        f, e = int(state.agent_features.shape[1]), state.csr_map.num_edges
        net_cfg = self.config.get("network", {})
        actor = ActorNetwork(
            f, e,
            hidden_dim=int(net_cfg.get("hidden_dim", 32)),
            deep_set_dim=int(net_cfg.get("deep_set_dim", 64)),
            hidden_layers=int(net_cfg.get("actor_hidden_layers", 4)),
            interaction_coupling_ratio=float(net_cfg.get("interaction_coupling_ratio", 0.9)),
        )
        critic = CriticNetwork(
            f, e,
            hidden_dim=int(net_cfg.get("hidden_dim", 32)),
            deep_set_dim=int(net_cfg.get("deep_set_dim", 64)),
            hidden_layers=int(net_cfg.get("critic_hidden_layers", 4)),
        )
        target = TargetCriticNetwork(critic)
        allowed = {field.name for field in __import__("dataclasses").fields(TrainingConfig)}
        train_values = {k: v for k, v in training_cfg_raw.items() if k in allowed}
        train_cfg = TrainingConfig(**train_values)
        trainer = NashDRLTrainer(env, actor, critic, target, mapper, train_cfg)

        all_steps: list[dict[str, Any]] = []
        all_vehicles: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        try:
            for episode in range(train_cfg.episodes):
                step_rows, vehicle_rows, edge_rows, episode_row = trainer.train_episode(episode)
                all_steps.extend(step_rows)
                all_vehicles.extend(vehicle_rows)
                all_edges.extend(edge_rows)
                episodes.append(episode_row)
        finally:
            env.close()

        self._write_csv(output_dir / "step_results.csv", all_steps)
        self._write_csv(output_dir / "vehicle_results.csv", all_vehicles)
        self._write_csv(output_dir / "edge_results.csv", all_edges)
        self._write_csv(output_dir / "episode_results.csv", episodes)
        metadata = {
            "mode": mode,
            "seed": seed,
            "dataset_path": str(dataset_path),
            "episodes": train_cfg.episodes,
            "steps_per_episode": max_steps,
            "num_vehicles": len(problem.vehicles),
            "num_nodes": problem.graph.num_nodes,
            "num_edges": problem.graph.num_edges,
            "trip_counts": {str(v.id): len(v.trips) for v in problem.vehicles},
        }
        (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"output_dir": output_dir, "episodes": episodes, "steps": all_steps, "vehicle_rows": all_vehicles, "edge_rows": all_edges, "metadata": metadata, "dataset_path": dataset_path}

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
            raw_path = self.config.get("training", {}).get("dataset_path") or self.config.get("simulation", {}).get("dataset_path")
            if not raw_path:
                raise ValueError("Real training requires training.dataset_path (or simulation.dataset_path)")
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            shutil.copy2(path, output_dir / "dataset.json")
            return path
        raise ValueError("Training mode must be 'mock' or 'real'")

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
