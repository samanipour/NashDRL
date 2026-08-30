from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nash_drl.models import ActorOutput, LQAdvantage


@dataclass(frozen=True, slots=True)
class LossResult:
    critic_loss: Tensor
    actor_loss: Tensor
    td_error: Tensor
    td_target: Tensor
    advantage: Tensor


def compute_td_target(reward: Tensor, next_value: Tensor, gamma: float, done: Tensor | bool) -> Tensor:
    done_t = torch.as_tensor(done, dtype=reward.dtype, device=reward.device)
    return reward + gamma * (1.0 - done_t) * next_value


def compute_training_losses(
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

    # Critic update learns V(x) toward r + gamma V_slow(x').
    critic_loss = 0.5 * (value - td_target.detach()).square().mean()

    # Actor update treats the Critic/target terms as constants and learns through A(x,u).
    predicted_q = value.detach() + advantage
    td_error = predicted_q - td_target.detach()
    actor_loss = 0.5 * td_error.square().mean()
    return LossResult(critic_loss, actor_loss, td_error, td_target, advantage)


# Backward-compatible alias.
def compute_losses(*args, **kwargs) -> LossResult:
    return compute_training_losses(*args, **kwargs)
