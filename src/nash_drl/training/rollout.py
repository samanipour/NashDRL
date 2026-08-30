from __future__ import annotations

from dataclasses import dataclass

from nash_drl.data import Action, GlobalState, Paths


@dataclass(slots=True)
class RolloutStep:
    state: GlobalState
    action: Action
    paths: Paths
    reward: object
    next_state: GlobalState
    done: bool
