from __future__ import annotations

from abc import ABC, abstractmethod

from data import Action, GlobalState, Paths


class ActionToPathMapper(ABC):
    @abstractmethod
    def map(self, action: Action, state: GlobalState) -> Paths:
        raise NotImplementedError
