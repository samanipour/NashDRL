from __future__ import annotations

import torch
from torch import Tensor

from nash_drl.data import GlobalState, NetworkInputs


class StateFeatureExtractor:
    """Build the two network streams with dataset-relative flow normalization."""

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

        # Edge flow is a global feature.  Normalize against the population so
        # a 50-vehicle experiment and a 500-vehicle experiment share a similar
        # input scale.
        edge_flow = state.edge_flow.to(dtype=x.dtype).clamp_min(0) / max(1, n)
        edge_flow = edge_flow.unsqueeze(0).expand(n, -1)
        non_invariant = torch.cat([x, edge_flow], dim=-1)  # [N,F+E]
        return NetworkInputs(invariant=invariant, non_invariant=non_invariant)
