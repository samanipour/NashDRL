from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nash_drl.data import Action, NetworkInputs


@dataclass(slots=True)
class PPORolloutStep:
    state_inputs: NetworkInputs
    action: Tensor                     # [N,E]
    old_log_prob: Tensor               # scalar
    reward: Tensor                     # scalar total system reward
    value: Tensor                      # scalar V(s)
    done: Tensor                       # scalar 0/1
    active_mask: Tensor                # [N]


@dataclass(slots=True)
class PPORolloutBatch:
    state_inputs: NetworkInputs
    actions: Tensor                    # [B,N,E]
    old_log_probs: Tensor              # [B]
    returns: Tensor                    # [B]
    advantages: Tensor                 # [B]
    active_masks: Tensor               # [B,N]


def stack_network_inputs(items: list[NetworkInputs], device: torch.device) -> NetworkInputs:
    return NetworkInputs(
        torch.stack([x.invariant for x in items], dim=0).to(device),
        torch.stack([x.non_invariant for x in items], dim=0).to(device),
    )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation for scalar total-system rewards."""
    rewards = rewards.reshape(-1)
    values = values.reshape(-1)
    dones = dones.reshape(-1)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    bootstrap = next_value.reshape(()).to(rewards)

    for t in reversed(range(len(rewards))):
        next_v = bootstrap if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * next_v - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def build_rollout_batch(
    steps: list[PPORolloutStep],
    advantages: Tensor,
    returns: Tensor,
    device: torch.device,
    *,
    normalize_advantage: bool = False,
) -> PPORolloutBatch:
    if not steps:
        raise ValueError("Cannot build PPO batch from an empty rollout")
    adv = advantages.to(device)
    if normalize_advantage and len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return PPORolloutBatch(
        state_inputs=stack_network_inputs([s.state_inputs for s in steps], device),
        actions=torch.stack([s.action for s in steps]).to(device),
        old_log_probs=torch.stack([s.old_log_prob for s in steps]).reshape(-1).to(device),
        returns=returns.reshape(-1).to(device),
        advantages=adv.reshape(-1),
        active_masks=torch.stack([s.active_mask for s in steps]).to(device),
    )
