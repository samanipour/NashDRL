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
    q_value: Tensor


def compute_td_target(
    reward: Tensor,
    next_value: Tensor,
    gamma: float,
    done: Tensor | bool,
) -> Tensor:
    done_t = torch.as_tensor(done, dtype=reward.dtype, device=reward.device)
    return reward + gamma * (1.0 - done_t) * next_value


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask_t = mask.to(dtype=values.dtype, device=values.device)
    if mask_t.shape != values.shape:
        raise ValueError(
            f"agent_mask must match value shape {tuple(values.shape)}, got {tuple(mask_t.shape)}"
        )
    denom = mask_t.sum().clamp_min(1.0)
    return (values * mask_t).sum() / denom


def compute_training_losses(
    value: Tensor,
    target_value: Tensor,
    reward: Tensor,
    actor_output: ActorOutput,
    action: Tensor,
    gamma: float,
    done: Tensor | bool,
    agent_mask: Tensor | None = None,
) -> LossResult:
    """Compute the Section-4 NashDRL TD/LQ losses.

    ``action`` MUST be detached from the Actor graph.  It represents the
    executed/sample action ``u`` while ``actor_output.mu`` is the trainable
    policy mean used to construct ``z = u - mu``.

    The source design decomposes the local Q estimate as ``Q = V + A`` and
    alternates optimization by detaching the counterpart term:

      critic: V + stop_gradient(A) against stop_gradient(TD target)
      actor:   stop_gradient(V) + A against stop_gradient(TD target)
    """
    if action.requires_grad:
        raise ValueError(
            "action must be detached before loss computation; the executed action "
            "must be treated as fixed while optimizing the Actor."
        )

    td_target = compute_td_target(reward, target_value, gamma, done).detach()
    advantage = LQAdvantage()(actor_output, action)
    q_value = value + advantage
    td_error = q_value - td_target

    critic_residual = value + advantage.detach() - td_target
    critic_loss = 0.5 * _masked_mean(critic_residual.square(), agent_mask)

    actor_residual = value.detach() + advantage - td_target
    actor_loss = 0.5 * _masked_mean(actor_residual.square(), agent_mask)

    return LossResult(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        td_error=td_error,
        td_target=td_target,
        advantage=advantage,
        q_value=q_value,
    )


# Backward-compatible alias.
def compute_losses(*args, **kwargs) -> LossResult:
    return compute_training_losses(*args, **kwargs)
