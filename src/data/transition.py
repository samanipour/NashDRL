from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .action import Action
from .state import GlobalState


@dataclass(slots=True)
class Transition:
    state: GlobalState
    action: Action
    reward: Tensor  # [N]
    next_state: GlobalState
    done: bool
