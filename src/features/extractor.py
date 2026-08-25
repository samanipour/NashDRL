from __future__ import annotations

import torch
from torch import Tensor

from data import GlobalState, NetworkInputs


class StateFeatureExtractor:
    """Builds the exact two-stream representation used by Actor/Critic."""

    def __call__(self, state: GlobalState) -> NetworkInputs:
        return self.extract(state)

    def extract(self, state: GlobalState) -> NetworkInputs:
        state.validate()
        x = state.agent_features
        n, _ = x.shape
        if n < 1:
            raise ValueError("At least one agent is required")

        other = []
        for i in range(n):
            other.append(torch.cat([x[:i], x[i + 1 :]], dim=0))
        invariant = torch.stack(other, dim=0)  # [N,N-1,F]

        edge_flow = state.edge_flow.unsqueeze(0).expand(n, -1)
        non_invariant = torch.cat([x, edge_flow], dim=-1)  # [N,F+E]
        return NetworkInputs(invariant=invariant, non_invariant=non_invariant)
