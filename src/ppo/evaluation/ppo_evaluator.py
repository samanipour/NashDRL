from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nash_drl.data import Action
from nash_drl.features import StateFeatureExtractor
from ppo.models import PPOActorCritic
from nash_drl.training.checkpoint import load_checkpoint
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment


@dataclass(frozen=True, slots=True)
class PPOEvaluationResult:
    episode: int
    steps: int
    total_reward: float
    budget_violations: int
    total_travel_time_s: float
    total_charging_cost: float


class PPOEvaluator:
    def __init__(self, env: NashSUMOTrainingEnvironment, policy: PPOActorCritic, *, device: str = "cpu") -> None:
        self.env = env
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.extractor = StateFeatureExtractor()

    def run_once(self, episode_index: int = 0) -> PPOEvaluationResult:
        state = self.env.reset(episode_index=episode_index)
        total_reward = 0.0
        violations = 0
        travel = 0.0
        cost = 0.0
        steps = 0
        while steps < self.env.max_steps:
            inputs = self.extractor(state)
            inputs = type(inputs)(inputs.invariant.to(self.device), inputs.non_invariant.to(self.device))
            active = torch.tensor(
                [float(v.current_trip_index < len(v.trips)) for v in self.env.problem.vehicles],
                device=self.device,
            )
            with torch.no_grad():
                action_tensor, _, _, _ = self.policy.sample_action(
                    inputs, active_mask=active, deterministic=True
                )
            state, reward, done, _ = self.env.step(Action(action_tensor.detach()))
            total_reward += float(reward.total_reward.detach().cpu())
            violations += int(reward.budget_violations.sum().item())
            travel += float(reward.travel_times.sum().item())
            cost += float(reward.charging_costs.sum().item())
            steps += 1
            if done:
                break
        return PPOEvaluationResult(episode_index, steps, total_reward, violations, travel, cost)


def evaluate_ppo(
    env: NashSUMOTrainingEnvironment,
    policy: PPOActorCritic,
    *,
    episodes: int,
    output_dir: str | Path,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> list[PPOEvaluationResult]:
    if checkpoint:
        payload = load_checkpoint(checkpoint, map_location=device)
        policy.load_state_dict(payload["policy_state_dict"])
    evaluator = PPOEvaluator(env, policy, device=device)
    results = [evaluator.run_once(i) for i in range(episodes)]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "test_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PPOEvaluationResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in PPOEvaluationResult.__dataclass_fields__})
    return results
