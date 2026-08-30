from __future__ import annotations

from torch import Tensor, nn

from .common.mlp import MLP


class DeepSetEncoder(nn.Module):
    """Shared φ embedding followed by permutation-invariant sum aggregation."""

    def __init__(self, feature_dim: int, embedding_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.phi = MLP(feature_dim, hidden_dim=hidden_dim, hidden_layers=2, output_dim=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [N,N-1,F], got {tuple(x.shape)}")
        n, rivals, feature_dim = x.shape
        embedded = self.phi(x.reshape(n * rivals, feature_dim)).reshape(n, rivals, -1)
        return embedded.sum(dim=1)
