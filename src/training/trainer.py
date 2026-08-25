from __future__ import annotations

from dataclasses import dataclass

import torch

from data import Action
from environment import NashEnvironment
from features import StateFeatureExtractor
from game import AnalyticalNashPolicy
from models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from routing import ActionToPathMapper

from .losses import compute_losses
from .target_update import hard_update


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_sigma: float = 0.1
    target_update_interval: int = 100
    device: str = "cpu"


class NashDRLTrainer:
    def __init__(
        self,
        env: NashEnvironment,
        actor: ActorNetwork,
        critic: CriticNetwork,
        target_critic: TargetCriticNetwork,
        mapper: ActionToPathMapper,
        config: TrainingConfig,
    ) -> None:
        self.env = env
        self.actor = actor.to(config.device)
        self.critic = critic.to(config.device)
        self.target_critic = target_critic.to(config.device)
        self.mapper = mapper
        self.config = config
        self.extractor = StateFeatureExtractor()
        self.nash_policy = AnalyticalNashPolicy()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.update_count = 0

    def select_action(self, state, training: bool = True) -> Action:
        inputs = self.extractor(state)
        params = self.actor(inputs)
        action_tensor = self.nash_policy.select(params)
        if training and self.config.exploration_sigma > 0:
            action_tensor = action_tensor + torch.randn_like(action_tensor) * self.config.exploration_sigma
        return Action(action_tensor)

    def train_step(self) -> dict[str, float]:
        state = self.env.reset()
        inputs = self.extractor(state)
        actor_output = self.actor(inputs)
        action_tensor = self.nash_policy.select(actor_output)
        action_tensor = action_tensor + torch.randn_like(action_tensor) * self.config.exploration_sigma
        action = Action(action_tensor)
        paths = self.mapper.map(action, state)
        next_state, reward_result, done, _ = self.env.step(paths)

        next_inputs = self.extractor(next_state)
        value = self.critic(inputs)
        with torch.no_grad():
            target_value = self.target_critic(next_inputs)
        rewards = reward_result.vehicle_rewards.to(value.device)
        done_tensor = torch.tensor(float(done), dtype=value.dtype, device=value.device)
        losses = compute_losses(
            value.detach(),
            target_value,
            rewards,
            actor_output,
            action_tensor,
            self.config.gamma,
            done_tensor,
        )

        critic_value = self.critic(inputs)
        critic_loss = 0.5 * (critic_value - losses.td_target.detach()).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        self.actor_optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            hard_update(self.target_critic, self.critic)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(losses.actor_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
        }
