from __future__ import annotations

import torch
from torch import nn


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
