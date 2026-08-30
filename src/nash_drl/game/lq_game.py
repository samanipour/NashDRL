from __future__ import annotations

import torch
from torch import Tensor

from nash_drl.models.actor import ActorOutput


class LQAdvantage:
    """Computes a per-agent scalar LQ advantage from `[N,E]` actions."""

    def __call__(self, params: ActorOutput, action: Tensor) -> Tensor:
        if action.shape != params.mu.shape:
            raise ValueError(f"Action {tuple(action.shape)} must match mean {tuple(params.mu.shape)}")
        z = action - params.mu  # [N,E]
        ego = -(params.p11 * z.square()).sum(dim=-1)  # [N]
        if z.shape[0] > 1:
            sum_other = z.sum(dim=0, keepdim=True) - z  # [N,E]
            interaction = -(params.p12 * z * sum_other).sum(dim=-1)
            rival = -(params.p22 * sum_other.square()).sum(dim=-1)
        else:
            interaction = torch.zeros_like(ego)
            rival = torch.zeros_like(ego)
        tilt = (params.psi * z).sum(dim=-1)
        return ego + interaction + rival + tilt
