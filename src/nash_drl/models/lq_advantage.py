from __future__ import annotations

import torch
from torch import Tensor, nn

from .actor import ActorOutput


class LQAdvantage(nn.Module):
    """Compute the local LQ advantage ``A_i(x;u)``.

    For edge-level actions ``u_i in R^E`` and ``z_i = u_i - mu_i``, the source
    document gives the per-agent structure

        - ||z_i||^2_{P11,i}
        - sum_{j != i} <z_i, z_j>_{P12,i}
        - sum_{j != i} ||z_j||^2_{P22,i}
        + sum_{j != i} z_j^T Psi_i.

    This implementation evaluates that expression for all agents at once.
    It supports both one state (``[N,E]``) and batches (``[B,N,E]``).
    """

    def forward(self, params: ActorOutput, action: Tensor) -> Tensor:
        self._validate(params, action)
        z = action - params.mu

        # Own deviation penalty.
        ego = -(params.p11 * z.square()).sum(dim=-1)

        # Sum of all rivals' deviations for every focal agent i.
        sum_other = z.sum(dim=-2, keepdim=True) - z

        # Pairwise interaction term.  P12_i is edge-wise, so each edge's
        # interaction is weighted before summation over the edge dimension.
        interaction = -(params.p12 * z * sum_other).sum(dim=-1)

        # Rival quadratic penalty: -sum_{j != i} ||z_j||^2_{P22,i}.
        # This is a sum of per-rival squared norms, NOT ||sum_j z_j||^2.
        sum_other_square = z.square().sum(dim=-2, keepdim=True) - z.square()
        rival = -(params.p22 * sum_other_square).sum(dim=-1)

        # Linear tilt from the rival deviations.
        tilt = (params.psi * sum_other).sum(dim=-1)

        # For N=1 sum_other is exactly zero, so the three interaction/crowd
        # terms naturally vanish.
        return ego + interaction + rival + tilt

    __call__ = nn.Module.__call__

    @staticmethod
    def _validate(params: ActorOutput, action: Tensor) -> None:
        shapes = {
            "mu": params.mu.shape,
            "p11": params.p11.shape,
            "p12": params.p12.shape,
            "p22": params.p22.shape,
            "psi": params.psi.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Actor parameter tensors must share a shape, got {shapes}")
        if action.shape != params.mu.shape:
            raise ValueError(
                f"Action {tuple(action.shape)} must match actor mean {tuple(params.mu.shape)}"
            )
        if action.ndim not in (2, 3):
            raise ValueError(f"Action must be [N,E] or [B,N,E], got {tuple(action.shape)}")
