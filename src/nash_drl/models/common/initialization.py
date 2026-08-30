from __future__ import annotations

from torch import nn


def initialize_linear_layers(module: nn.Module) -> None:
    """Initialize Linear layers with Xavier weights and zero bias.

    The paper does not prescribe an initialization scheme; this provides a
    deterministic, conventional initialization point for experiments without
    changing the network topology.
    """
    import torch.nn.init as init

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)
