from __future__ import annotations

from abc import ABC, abstractmethod

from nash_drl.data import Action, GlobalState, Paths


class ActionToPathMapper(ABC):
    @abstractmethod
    def map(self, action: Action, state: GlobalState) -> Paths:
        raise NotImplementedError
