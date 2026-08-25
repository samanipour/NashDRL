from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class Action:
    """Edge weights for all vehicles.

    edge_weights: [N, E]
    """

    edge_weights: Tensor

    def validate(self) -> None:
        if self.edge_weights.ndim != 2:
            raise ValueError(f"Action must be [N,E], got {self.edge_weights.shape}")

    @property
    def num_agents(self) -> int:
        return int(self.edge_weights.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_weights.shape[1])


@dataclass(slots=True)
class Paths:
    """Selected edge-id sequences, one path per agent."""

    edge_ids: list[list[int]]
