from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from nash_drl.data import NetworkInputs

from nash_drl.models.deep_sets import DeepSetEncoder
from nash_drl.models.common.mlp import MLP


@dataclass(slots=True)
class PPOOutput:
    """PPO actor-critic outputs.

    ``mean`` and ``log_std`` are [N,E] (or [B,N,E]). ``value`` is one scalar
    global value per state: [] for a single state or [B] for a batch.
    """

    mean: Tensor
    log_std: Tensor
    value: Tensor


class PPOActorCritic(nn.Module):
    """Continuous-action PPO actor-critic baseline.

    It consumes exactly the same state representation used by NashDRL:
      invariant     [N,N-1,F] / [B,N,N-1,F]
      non_invariant [N,F+E]   / [B,N,F+E]

    Unlike the NashDRL Actor/Critic, PPO has:
      * a stochastic Gaussian actor over the [N,E] route-weight matrix;
      * a scalar global value function V(x), because the PPO baseline maximizes
        the total system reward;
      * no LQ parameters, LQ advantage, or Target Critic.

    Note: PPO still uses a *standard policy-gradient advantage estimator*
    (typically GAE) inside its clipped surrogate objective. This is not the
    NashDRL LQ/game-theoretic advantage function.
    """

    def __init__(
        self,
        agent_feature_dim: int,
        num_edges: int,
        hidden_dim: int = 64,
        deep_set_dim: int = 64,
        hidden_layers: int = 2,
        deep_set_hidden_layers: int = 2,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ) -> None:
        super().__init__()
        if agent_feature_dim <= 0 or num_edges <= 0:
            raise ValueError("agent_feature_dim and num_edges must be positive")
        if min_log_std >= max_log_std:
            raise ValueError("min_log_std must be smaller than max_log_std")

        self.agent_feature_dim = agent_feature_dim
        self.num_edges = num_edges
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.deep_sets = DeepSetEncoder(
            feature_dim=agent_feature_dim,
            embedding_dim=deep_set_dim,
            hidden_dim=hidden_dim,
            embedding_hidden_layers=deep_set_hidden_layers,
        )
        self.non_invariant = nn.Sequential(
            nn.Linear(agent_feature_dim + num_edges, deep_set_dim),
            nn.Tanh(),
        )

        latent_dim = 2 * deep_set_dim
        self.actor_trunk = MLP(
            latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=num_edges,
        )
        self.value_trunk = MLP(
            latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=1,
        )
        # A learned, global action scale is closer to the standard diagonal
        # Gaussian actor used in continuous-control PPO than a fixed noise
        # schedule, while avoiding an oversized state-dependent covariance.
        self.log_std = nn.Parameter(torch.full((num_edges,), float(initial_log_std)))

    def _latent(self, inputs: NetworkInputs) -> Tensor:
        inputs.validate()
        crowd = self.deep_sets(inputs.invariant)
        ego_global = self.non_invariant(inputs.non_invariant)
        return torch.cat((crowd, ego_global), dim=-1)

    def forward(self, inputs: NetworkInputs) -> PPOOutput:
        latent = self._latent(inputs)
        mean = self.actor_trunk(latent)
        value_per_agent = self.value_trunk(latent).squeeze(-1)
        # The PPO value is a scalar estimate for the joint/system reward.
        value = value_per_agent.mean(dim=-1)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        log_std = log_std.expand_as(mean)
        return PPOOutput(mean=mean, log_std=log_std, value=value)

    def distribution(self, inputs: NetworkInputs) -> Normal:
        out = self.forward(inputs)
        return Normal(out.mean, out.log_std.exp())

    def sample_action(
        self,
        inputs: NetworkInputs,
        *,
        active_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return joint action, joint log-probability, joint entropy and V(s)."""
        out = self.forward(inputs)
        dist = Normal(out.mean, out.log_std.exp())
        action = out.mean if deterministic else dist.sample()

        per_agent_log_prob = dist.log_prob(action).sum(dim=-1)  # [N] / [B,N]
        per_agent_entropy = dist.entropy().sum(dim=-1)

        if active_mask is not None:
            mask = active_mask.to(dtype=per_agent_log_prob.dtype)
            while mask.ndim < per_agent_log_prob.ndim:
                mask = mask.unsqueeze(0)
            per_agent_log_prob = per_agent_log_prob * mask
            per_agent_entropy = per_agent_entropy * mask

            action_mask = mask
            while action_mask.ndim < action.ndim:
                action_mask = action_mask.unsqueeze(-1)
            action = torch.where(action_mask.bool(), action, torch.zeros_like(action))

        joint_log_prob = per_agent_log_prob.sum(dim=-1)
        joint_entropy = per_agent_entropy.sum(dim=-1)
        return action, joint_log_prob, joint_entropy, out.value

    def evaluate_actions(
        self,
        inputs: NetworkInputs,
        actions: Tensor,
        *,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return joint log-probability, entropy and scalar V(s)."""
        out = self.forward(inputs)
        dist = Normal(out.mean, out.log_std.exp())
        per_agent_log_prob = dist.log_prob(actions).sum(dim=-1)
        per_agent_entropy = dist.entropy().sum(dim=-1)

        if active_mask is not None:
            mask = active_mask.to(dtype=per_agent_log_prob.dtype)
            while mask.ndim < per_agent_log_prob.ndim:
                mask = mask.unsqueeze(0)
            per_agent_log_prob = per_agent_log_prob * mask
            per_agent_entropy = per_agent_entropy * mask

        return per_agent_log_prob.sum(dim=-1), per_agent_entropy.sum(dim=-1), out.value
