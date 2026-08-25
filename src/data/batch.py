from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class NetworkInputs:
    """Network-ready state tensors.

    invariant: [N, N-1, F]
    non_invariant: [N, F+E]
    """

    invariant: Tensor
    non_invariant: Tensor

    def validate(self) -> None:
        if self.invariant.ndim != 3:
            raise ValueError("invariant must have shape [N,N-1,F]")
        if self.non_invariant.ndim != 2:
            raise ValueError("non_invariant must have shape [N,F+E]")
        if self.invariant.shape[0] != self.non_invariant.shape[0]:
            raise ValueError("Agent dimension mismatch between network inputs")
