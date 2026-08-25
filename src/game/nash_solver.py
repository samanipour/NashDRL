from __future__ import annotations

from torch import Tensor

from models.actor import ActorOutput


class AnalyticalNashPolicy:
    """Interface for analytical Nash/LQ action selection.

    The exact equilibrium derivation can be implemented here without changing
    Actor/Critic network APIs.
    """

    def select(self, params: ActorOutput) -> Tensor:
        return params.mu
