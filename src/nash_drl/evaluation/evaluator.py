from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nash_drl.training.trainer import NashDRLTrainer


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episode: int
    steps: int
    total_reward: float
    budget_violations: int
    total_travel_time_s: float
    total_charging_cost: float


class Evaluator:
    """Run trained policies without exploration and persist test metrics."""

    def __init__(self, trainer: NashDRLTrainer) -> None:
        self.trainer = trainer

    def run_once(self, episode_index: int = 0) -> EvaluationResult:
        state = self.trainer.env.reset(episode_index=episode_index)
        total_reward = 0.0
        violations = 0
        travel = 0.0
        cost = 0.0
        steps = 0
        for step in range(self.trainer.env.max_steps):
            action = self.trainer.select_action(state, training=False)
            next_state, reward, done, _ = self.trainer.env.step(action)
            total_reward += float(reward.total_reward)
            violations += int(reward.budget_violations.sum())
            travel += float(reward.travel_times.sum())
            cost += float(reward.charging_costs.sum())
            steps += 1
            state = next_state
            if done:
                break
        return EvaluationResult(episode_index, steps, total_reward, violations, travel, cost)

    def run(self, episodes: int, output_dir: str | Path) -> list[EvaluationResult]:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        results = [self.run_once(i) for i in range(episodes)]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "test_results.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(EvaluationResult.__dataclass_fields__))
            writer.writeheader()
            for result in results:
                writer.writerow({name: getattr(result, name) for name in EvaluationResult.__dataclass_fields__})
        return results
