from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from game import LQAdvantage
from models import ActorOutput


@dataclass(frozen=True, slots=True)
class LossResult:
    critic_loss: Tensor
    actor_loss: Tensor
    td_error: Tensor
    td_target: Tensor


def compute_td_target(reward: Tensor, next_value: Tensor, gamma: float, done: Tensor | bool) -> Tensor:
    done_t = torch.as_tensor(done, dtype=reward.dtype, device=reward.device)
    return reward + gamma * (1.0 - done_t) * next_value


def compute_losses(
    value: Tensor,
    target_value: Tensor,
    reward: Tensor,
    actor_output: ActorOutput,
    action: Tensor,
    gamma: float,
    done: Tensor | bool,
) -> LossResult:
    td_target = compute_td_target(reward, target_value, gamma, done)
    advantage = LQAdvantage()(actor_output, action)
    predicted = value + advantage
    td_error = predicted - td_target
    critic_loss = 0.5 * (value - td_target.detach()).square().mean()
    actor_loss = 0.5 * td_error.square().mean()
    return LossResult(critic_loss, actor_loss, td_error, td_target)
