from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """Fully-connected MLP with SiLU hidden activations.

    The main Actor/Critic trunks use four hidden layers of 32
    units with SiLU activations.  This reusable class keeps those dimensions
    configurable while preserving that default architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        hidden_layers: int = 4,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.SiLU()))
            in_dim = hidden_dim
        if output_dim is not None:
            if output_dim <= 0:
                raise ValueError("output_dim must be positive when provided")
            layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
