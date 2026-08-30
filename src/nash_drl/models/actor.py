from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from nash_drl.data import NetworkInputs

from .common.mlp import MLP
from .deep_sets import DeepSetEncoder


@dataclass(slots=True)
class ActorOutput:
    mu: Tensor
    p11: Tensor
    p12: Tensor
    p22: Tensor
    psi: Tensor

    def as_tensor(self) -> Tensor:
        return torch_stack([self.mu, self.p11, self.p12, self.p22, self.psi], dim=0)


def torch_stack(xs: list[Tensor], dim: int) -> Tensor:
    import torch
    return torch.stack(xs, dim=dim)


class ActorNetwork(nn.Module):
    """Deep Sets + ego/global stream → five LQ parameter channels [5,N,E]."""

    def __init__(self, agent_feature_dim: int, num_edges: int, hidden_dim: int = 32, deep_set_dim: int = 64, hidden_layers: int = 4) -> None:
        super().__init__()
        self.num_edges = num_edges
        self.deep_sets = DeepSetEncoder(agent_feature_dim, deep_set_dim, hidden_dim)
        self.non_invariant = nn.Sequential(
            nn.Linear(agent_feature_dim + num_edges, deep_set_dim),
            nn.SiLU(),
        )
        self.trunk = MLP(2 * deep_set_dim, hidden_dim, hidden_layers, output_dim=5 * num_edges)

    def forward(self, inputs: NetworkInputs) -> ActorOutput:
        n = inputs.non_invariant.shape[0]
        crowd = self.deep_sets(inputs.invariant)
        ego = self.non_invariant(inputs.non_invariant)
        latent = torch_cat(crowd, ego)
        raw = self.trunk(latent).reshape(n, 5, self.num_edges)
        mu, p11_raw, p12, p22_raw, psi = raw.unbind(dim=1)
        p11 = positive(p11_raw)
        p22 = positive(p22_raw)
        return ActorOutput(mu=mu, p11=p11, p12=p12, p22=p22, psi=psi)


def torch_cat(a: Tensor, b: Tensor) -> Tensor:
    import torch
    return torch.cat([a, b], dim=-1)


def positive(x: Tensor) -> Tensor:
    import torch.nn.functional as F
    return F.softplus(x) + 1e-6
