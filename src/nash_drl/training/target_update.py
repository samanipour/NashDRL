from __future__ import annotations

import torch
from torch import nn


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard-copy a source network into a target network.

    TargetCriticNetwork exposes ``hard_update_from`` because it intentionally
    wraps an exact CriticNetwork replica.  Prefer that semantic API when it is
    available.
    """
    updater = getattr(target, "hard_update_from", None)
    if callable(updater):
        updater(source)
        return
    target.load_state_dict(source.state_dict(), strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    with torch.no_grad():
        target_state = target.state_dict()
        source_state = source.state_dict()
        if target_state.keys() != source_state.keys():
            raise ValueError("Soft update requires matching state_dict keys")
        for key in target_state:
            if torch.is_floating_point(target_state[key]):
                target_state[key].mul_(1.0 - tau).add_(source_state[key], alpha=tau)
            else:
                target_state[key].copy_(source_state[key])
    target.eval()
