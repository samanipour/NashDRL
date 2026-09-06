from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class NetworkInputs:
    """Network-ready state tensors.

    Single-state:
      invariant: [N, N-1, F]
      non_invariant: [N, F+E]

    Batched training:
      invariant: [B, N, N-1, F]
      non_invariant: [B, N, F+E]
    """

    invariant: Tensor
    non_invariant: Tensor

    def validate(self) -> None:
        if self.invariant.ndim == 3:
            if self.non_invariant.ndim != 2:
                raise ValueError("single-state non_invariant must be [N,F+E]")
            n, rivals, _ = self.invariant.shape
            if self.non_invariant.shape[0] != n:
                raise ValueError("Agent dimension mismatch between network inputs")
        elif self.invariant.ndim == 4:
            if self.non_invariant.ndim != 3:
                raise ValueError("batched non_invariant must be [B,N,F+E]")
            b, n, rivals, _ = self.invariant.shape
            if self.non_invariant.shape[:2] != (b, n):
                raise ValueError("Batch/agent dimensions mismatch between network inputs")
        else:
            raise ValueError("invariant must be [N,N-1,F] or [B,N,N-1,F]")
        if rivals != n - 1:
            raise ValueError("invariant must contain exactly N-1 rival agents")
