from __future__ import annotations

import torch
from torch import Tensor, nn

from nash_drl.data import NetworkInputs

from .common.mlp import MLP
from .deep_sets import DeepSetEncoder


class CriticNetwork(nn.Module):
    """per-agent state-value network.

    Inputs:
      * invariant rival stream: ``[N,N-1,F]`` or ``[B,N,N-1,F]``
      * ego/global stream: ``[N,F+E]`` or ``[B,N,F+E]``

    The rival stream is permutation-invariant through Deep Sets.  The
    resulting crowd representation is concatenated with the flattened
    ego/global feature vector for each focal agent and fed to a four-layer,
    32-unit, SiLU value trunk.

    Output:
      * ``[N]`` for one state
      * ``[B,N]`` for a batch of states
    """

    def __init__(
        self,
        agent_feature_dim: int,
        num_edges: int,
        hidden_dim: int = 32,
        deep_set_dim: int = 64,
        hidden_layers: int = 4,
        deep_set_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if agent_feature_dim <= 0:
            raise ValueError("agent_feature_dim must be positive")
        if num_edges <= 0:
            raise ValueError("num_edges must be positive")

        self.agent_feature_dim = agent_feature_dim
        self.num_edges = num_edges
        self.hidden_dim = hidden_dim
        self.deep_set_dim = deep_set_dim
        self.hidden_layers = hidden_layers

        self.deep_sets = DeepSetEncoder(
            feature_dim=agent_feature_dim,
            embedding_dim=deep_set_dim,
            hidden_dim=hidden_dim,
            embedding_hidden_layers=deep_set_hidden_layers,
        )

        # The document's "Flatten" stage represents the focal agent's
        # ego/global feature vector before it is joined with the crowd vector.
        value_input_dim = agent_feature_dim + num_edges + deep_set_dim
        self.trunk = MLP(
            value_input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=1,
        )

    def forward(self, inputs: NetworkInputs) -> Tensor:
        self._validate_inputs(inputs)
        crowd = self.deep_sets(inputs.invariant)

        # Flatten all feature dimensions belonging to one focal agent while
        # preserving batch/agent leading dimensions.
        ego_global = inputs.non_invariant.flatten(start_dim=-1)
        x = torch.cat((ego_global, crowd), dim=-1)
        return self.trunk(x).squeeze(-1)

    def _validate_inputs(self, inputs: NetworkInputs) -> None:
        invariant = inputs.invariant
        non_invariant = inputs.non_invariant

        if invariant.ndim == 3:
            n, rivals, f = invariant.shape
            expected_non = (n, self.agent_feature_dim + self.num_edges)
        elif invariant.ndim == 4:
            b, n, rivals, f = invariant.shape
            expected_non = (b, n, self.agent_feature_dim + self.num_edges)
        else:
            raise ValueError(f"Invalid invariant input rank: {invariant.ndim}")

        if rivals != n - 1:
            raise ValueError(
                f"Invariant input must be [N,N-1,F] or [B,N,N-1,F], got {tuple(invariant.shape)}"
            )
        if f != self.agent_feature_dim:
            raise ValueError(f"Expected F={self.agent_feature_dim}, got {f}")
        if tuple(non_invariant.shape) != expected_non:
            raise ValueError(
                f"Expected non-invariant input {expected_non}, got {tuple(non_invariant.shape)}"
            )
