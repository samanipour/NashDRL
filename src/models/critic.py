from __future__ import annotations

from torch import Tensor, nn

from data import NetworkInputs

from .common.mlp import MLP
from .deep_sets import DeepSetEncoder


class CriticNetwork(nn.Module):
    """Permutation-invariant value network producing one scalar per agent [N]."""

    def __init__(self, agent_feature_dim: int, num_edges: int, hidden_dim: int = 32, deep_set_dim: int = 64, hidden_layers: int = 4) -> None:
        super().__init__()
        self.deep_sets = DeepSetEncoder(agent_feature_dim, deep_set_dim, hidden_dim)
        self.trunk = MLP(agent_feature_dim + num_edges + deep_set_dim, hidden_dim, hidden_layers, output_dim=1)

    def forward(self, inputs: NetworkInputs) -> Tensor:
        crowd = self.deep_sets(inputs.invariant)
        x = __import__("torch").cat([inputs.non_invariant, crowd], dim=-1)
        return self.trunk(x).squeeze(-1)
