from __future__ import annotations

from torch import Tensor


def mean_reward(reward: Tensor) -> float:
    return float(reward.mean().item())


def budget_violation_count(violations: Tensor) -> int:
    return int(violations.sum().item())
