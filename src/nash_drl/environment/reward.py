from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class RewardResult:
    vehicle_rewards: Tensor  # [N]
    total_reward: Tensor  # scalar
    travel_times: Tensor  # [N], hours
    charging_costs: Tensor  # [N]
    budget_violations: Tensor  # [N], bool
    regular_rewards: Tensor  # [N], reward before hard-budget replacement
    budgets: Tensor  # [N], budget used for this decision
    budget_excess: Tensor  # [N], max(0, cost-budget)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    travel_time_weight: float = 1.0
    charging_cost_weight: float = 1.0
    budget_penalty: float = 100.0


class RewardModel:
    """Implement the paper's piecewise reward exactly.

    A hard-budget violation receives -P independent of travel time and cost.
    The experimenter controls P through configuration; it is not silently
    rescaled here.
    """

    def __init__(self, config: RewardConfig, profile_name: str = "unknown") -> None:
        self.config = config
        self.profile_name = profile_name

    def compute(
        self,
        travel_times: Tensor,
        charging_costs: Tensor,
        budgets: Tensor,
    ) -> RewardResult:
        violations = charging_costs > budgets + 1e-9
        budget_excess = (charging_costs - budgets).clamp_min(0.0)
        regular = (
            -self.config.travel_time_weight * travel_times
            - self.config.charging_cost_weight * charging_costs
        )
        rewards = torch.where(
            violations,
            torch.full_like(regular, -self.config.budget_penalty),
            regular,
        )
        return RewardResult(
            vehicle_rewards=rewards,
            total_reward=rewards.sum(),
            travel_times=travel_times,
            charging_costs=charging_costs,
            budget_violations=violations,
            regular_rewards=regular,
            budgets=budgets,
            budget_excess=budget_excess,
        )


def build_reward_config(experiment_config: dict, algorithm: str = "common") -> RewardConfig:
    """Resolve an algorithm-specific reward profile from one experiment file.

    Supported layout:
      reward.common   -> common benchmark/evaluation definition
      reward.nash_drl -> NashDRL learning definition
      reward.ppo      -> PPO learning definition

    A legacy flat ``reward`` mapping is still accepted.
    """
    raw = dict(experiment_config.get("reward", {}))
    profile_keys = {"common", "evaluation", "nash_drl", "ppo"}
    if not profile_keys.intersection(raw):
        return RewardConfig(**raw)

    common = dict(raw.get("common", raw.get("evaluation", {})))
    if algorithm in {"nash_drl", "ppo", "common"}:
        common.update(raw.get(algorithm, {}))
    allowed = {"travel_time_weight", "charging_cost_weight", "budget_penalty"}
    return RewardConfig(**{k: v for k, v in common.items() if k in allowed})
