from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from nash_drl.data import NetworkInputs

from .critic import CriticNetwork


class TargetCriticNetwork(nn.Module):
    """Frozen exact architectural copy of :class:`CriticNetwork`.

    The source specification requires the Target Critic to be an exact copy of
    the main Critic and to be updated by hard parameter copies.  A deepcopy is
    therefore preferable to re-inferring architecture dimensions from module
    internals.
    """

    def __init__(self, critic: CriticNetwork) -> None:
        super().__init__()
        self.critic = deepcopy(critic)
        self._freeze()

    def forward(self, inputs: NetworkInputs) -> Tensor:
        return self.critic(inputs)

    @torch.no_grad()
    def hard_update_from(self, critic: CriticNetwork) -> None:
        self.critic.load_state_dict(critic.state_dict(), strict=True)
        self._freeze()

    def _freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

