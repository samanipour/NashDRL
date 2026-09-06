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

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(
        self,
        travel_times: Tensor,
        charging_costs: Tensor,
        budgets: Tensor,
    ) -> RewardResult:
        violations = charging_costs > budgets + 1e-9
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
        )
