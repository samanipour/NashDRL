from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nash_drl.data import Action, GlobalState, NetworkInputs, Paths


@dataclass(slots=True)
class RolloutStep:
    state: GlobalState
    action: Action
    paths: Paths
    reward: object
    next_state: GlobalState
    done: bool


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    state_inputs: NetworkInputs
    actions: Tensor          # [B,N,E]
    rewards: Tensor          # [B,N]
    next_inputs: NetworkInputs
    done: Tensor             # [B,N]
    agent_mask: Tensor       # [B,N]


class ReplayBuffer:
    """Small CPU replay buffer for fixed-shape multi-agent transitions."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self._items: list[tuple[Tensor, ...]] = []
        self._position = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state_inputs: NetworkInputs,
        action: Tensor,
        reward: Tensor,
        next_inputs: NetworkInputs,
        done: Tensor | bool,
        agent_mask: Tensor,
    ) -> None:
        item = (
            state_inputs.invariant.detach().cpu().clone(),
            state_inputs.non_invariant.detach().cpu().clone(),
            action.detach().cpu().clone(),
            reward.detach().cpu().clone(),
            next_inputs.invariant.detach().cpu().clone(),
            next_inputs.non_invariant.detach().cpu().clone(),
            torch.as_tensor(done, dtype=torch.float32).detach().cpu().clone(),
            agent_mask.detach().cpu().clone(),
        )
        # Keep the tuple intentionally flat for inexpensive stacking.
        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._position] = item
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device | str) -> TrainingBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._items) < batch_size:
            raise ValueError("Not enough replay samples")
        indices = torch.randperm(len(self._items), generator=self.generator)[:batch_size].tolist()
        selected = [self._items[i] for i in indices]

        inv = torch.stack([x[0] for x in selected], dim=0).to(device)
        non = torch.stack([x[1] for x in selected], dim=0).to(device)
        actions = torch.stack([x[2] for x in selected], dim=0).to(device)
        rewards = torch.stack([x[3] for x in selected], dim=0).to(device)
        next_inv = torch.stack([x[4] for x in selected], dim=0).to(device)
        next_non = torch.stack([x[5] for x in selected], dim=0).to(device)
        done = torch.stack([x[6] for x in selected], dim=0).to(device)
        mask = torch.stack([x[7] for x in selected], dim=0).to(device)
        if done.ndim == 1:
            done = done.unsqueeze(-1).expand(-1, rewards.shape[-1])
        return TrainingBatch(
            state_inputs=NetworkInputs(inv, non),
            actions=actions,
            rewards=rewards,
            next_inputs=NetworkInputs(next_inv, next_non),
            done=done,
            agent_mask=mask,
        )
