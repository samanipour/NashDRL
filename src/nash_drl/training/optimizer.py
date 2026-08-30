from __future__ import annotations

from torch import nn


def make_adam(parameters, lr: float) -> nn.Module:
    import torch
    return torch.optim.Adam(parameters, lr=lr)
