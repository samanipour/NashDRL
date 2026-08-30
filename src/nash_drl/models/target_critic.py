from __future__ import annotations

from torch import Tensor, nn

from nash_drl.data import NetworkInputs

from .critic import CriticNetwork


class TargetCriticNetwork(nn.Module):
    """Frozen architectural copy of CriticNetwork used for TD targets."""

    def __init__(self, critic: CriticNetwork) -> None:
        super().__init__()
        self.critic = CriticNetwork.__new__(CriticNetwork)
        self.critic.__init__(
            agent_feature_dim=_infer_agent_feature_dim(critic),
            num_edges=_infer_num_edges(critic),
            hidden_dim=_infer_hidden_dim(critic),
            deep_set_dim=_infer_deep_set_dim(critic),
            hidden_layers=_infer_hidden_layers(critic),
        )
        self.load_state_dict(critic.state_dict(), strict=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, inputs: NetworkInputs) -> Tensor:
        return self.critic(inputs)

    def hard_update_from(self, critic: CriticNetwork) -> None:
        self.critic.load_state_dict(critic.state_dict())
        for p in self.parameters():
            p.requires_grad_(False)


def _infer_agent_feature_dim(critic: CriticNetwork) -> int:
    return int(critic.deep_sets.phi.net[0].in_features)


def _infer_hidden_dim(critic: CriticNetwork) -> int:
    return int(critic.deep_sets.phi.net[0].out_features)


def _infer_deep_set_dim(critic: CriticNetwork) -> int:
    return int(critic.deep_sets.phi.net[-1].out_features)


def _infer_num_edges(critic: CriticNetwork) -> int:
    first = critic.trunk.net[0]
    return int(first.in_features - _infer_agent_feature_dim(critic) - _infer_deep_set_dim(critic))


def _infer_hidden_layers(critic: CriticNetwork) -> int:
    linear_count = sum(1 for m in critic.trunk.net if isinstance(m, nn.Linear))
    return max(1, linear_count - 1)
