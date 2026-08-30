from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nash_drl.data import NetworkInputs

from .common.activations import strictly_positive
from .common.mlp import MLP
from .deep_sets import DeepSetEncoder


@dataclass(slots=True)
class ActorOutput:
    """LQ-Advantage parameters produced by the Actor.

    For a single global state each field is ``[N,E]`` and ``as_tensor()`` is
    ``[5,N,E]``.  A leading batch dimension is supported as ``[B,N,E]`` and
    becomes ``[B,5,N,E]``.
    """

    mu: Tensor
    p11: Tensor
    p12: Tensor
    p22: Tensor
    psi: Tensor

    def as_tensor(self) -> Tensor:
        if self.mu.ndim == 2:
            return torch.stack((self.mu, self.p11, self.p12, self.p22, self.psi), dim=0)
        if self.mu.ndim == 3:
            return torch.stack((self.mu, self.p11, self.p12, self.p22, self.psi), dim=1)
        raise ValueError(f"Actor parameters must have shape [N,E] or [B,N,E], got {self.mu.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.mu.shape)


class ActorNetwork(nn.Module):
    """
    Inputs:
      * invariant stream: ``[N,N-1,F]`` or ``[B,N,N-1,F]``
      * non-invariant stream: ``[N,F+E]`` or ``[B,N,F+E]``

    Architecture:
      * Stream 1: shared Deep Sets encoder phi + sum aggregation.
      * Stream 2: ego/global linear projection.
      * Concatenation of the two latent streams.
      * Main trunk: four FC hidden layers, 32 units each, SiLU.
      * Output: five parameter channels ``mu, P11, P12, P22, Psi``, each
        shaped ``[N,E]`` (or ``[B,N,E]``).

    ``P11`` and ``P22`` use a strictly-positive softplus transform, matching
    the paper's requirement that these outputs preserve the concavity-related
    positivity condition.
    """

    PARAMETER_CHANNELS = 5

    def __init__(
        self,
        agent_feature_dim: int,
        num_edges: int,
        hidden_dim: int = 32,
        deep_set_dim: int = 64,
        hidden_layers: int = 4,
        deep_set_hidden_layers: int = 2,
        positivity_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if agent_feature_dim <= 0:
            raise ValueError("agent_feature_dim must be positive")
        if num_edges <= 0:
            raise ValueError("num_edges must be positive")
        if positivity_epsilon <= 0:
            raise ValueError("positivity_epsilon must be positive")

        self.agent_feature_dim = agent_feature_dim
        self.num_edges = num_edges
        self.hidden_dim = hidden_dim
        self.deep_set_dim = deep_set_dim
        self.hidden_layers = hidden_layers
        self.positivity_epsilon = positivity_epsilon

        self.deep_sets = DeepSetEncoder(
            feature_dim=agent_feature_dim,
            embedding_dim=deep_set_dim,
            hidden_dim=hidden_dim,
            embedding_hidden_layers=deep_set_hidden_layers,
        )

        # Stream 2: X_i + X_0, represented here as [F+E] per focal agent.
        self.non_invariant = nn.Sequential(
            nn.Linear(agent_feature_dim + num_edges, deep_set_dim),
            nn.SiLU(),
        )

        # Exactly four hidden FC layers of 32 units by default, as specified
        # in Section 4.2.
        self.trunk = MLP(
            2 * deep_set_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=self.PARAMETER_CHANNELS * num_edges,
        )

    def forward(self, inputs: NetworkInputs) -> ActorOutput:
        self._validate_inputs(inputs)

        n = inputs.non_invariant.shape[-2]
        crowd = self.deep_sets(inputs.invariant)
        ego_global = self.non_invariant(inputs.non_invariant)
        latent = torch.cat((crowd, ego_global), dim=-1)

        raw = self.trunk(latent).reshape(*latent.shape[:-1], self.PARAMETER_CHANNELS, self.num_edges)
        mu, p11_raw, p12, p22_raw, psi = raw.unbind(dim=-2)

        p11 = strictly_positive(p11_raw, self.positivity_epsilon)
        p22 = strictly_positive(p22_raw, self.positivity_epsilon)

        # `n` is retained to make the intended focal-agent dimension explicit
        # for the unbatched API and to guard accidental scalar outputs.
        if mu.shape[-2] != n or mu.shape[-1] != self.num_edges:
            raise RuntimeError(f"Actor output has invalid shape {tuple(mu.shape)}")

        return ActorOutput(mu=mu, p11=p11, p12=p12, p22=p22, psi=psi)

    def _validate_inputs(self, inputs: NetworkInputs) -> None:
        invariant = inputs.invariant
        non_invariant = inputs.non_invariant
        if invariant.ndim == 3:
            n, rivals, f = invariant.shape
            expected_non = (n, self.agent_feature_dim + self.num_edges)
            if rivals != n - 1:
                raise ValueError(
                    f"Invariant input must be [N,N-1,F], got {tuple(invariant.shape)}"
                )
        elif invariant.ndim == 4:
            b, n, rivals, f = invariant.shape
            expected_non = (b, n, self.agent_feature_dim + self.num_edges)
            if rivals != n - 1:
                raise ValueError(
                    f"Invariant input must be [B,N,N-1,F], got {tuple(invariant.shape)}"
                )
        else:
            raise ValueError(f"Invalid invariant input rank: {invariant.ndim}")

        if f != self.agent_feature_dim:
            raise ValueError(f"Expected F={self.agent_feature_dim}, got {f}")
        if tuple(non_invariant.shape) != expected_non:
            raise ValueError(
                f"Expected non-invariant input {expected_non}, got {tuple(non_invariant.shape)}"
            )
