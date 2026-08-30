from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, hidden_layers: int = 4, output_dim: int | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
