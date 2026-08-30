from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from nash_drl.data import Action
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment
from nash_drl.features import StateFeatureExtractor
from nash_drl.game import AnalyticalNashPolicy
from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from nash_drl.routing import ActionToPathMapper

from .checkpoint import save_checkpoint
from .losses import compute_training_losses
from .target_update import hard_update


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    episodes: int = 10
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_sigma: float = 0.1
    target_update_interval: int = 100
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs/training"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


class NashDRLTrainer:
    """Online episodic NashDRL training engine.

    The environment defines the episode horizon as the maximum trip-set length.
    Each trainer iteration performs one complete trip leg in SUMO, computes the
    TD target and LQ-advantage error, then alternately updates Critic and Actor.
    """

    def __init__(
        self,
        env: NashSUMOTrainingEnvironment,
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
        """Backward-compatible one-step analytical-environment update for unit tests.

        Production training uses :meth:`train_episode`, which executes SUMO for
        each trip-set step.
        """
        state = self.env.reset()
        inputs = self.extractor(state)
        actor_output = self.actor(inputs)
        action_tensor = self.nash_policy.select(actor_output)
        if self.config.exploration_sigma > 0:
            action_tensor = action_tensor + torch.randn_like(action_tensor) * self.config.exploration_sigma
        action = Action(action_tensor)
        paths = self.mapper.map(action, state)
        next_state, reward_result, done, _ = self.env.step(paths)
        next_inputs = self.extractor(next_state)
        value = self.critic(inputs)
        with torch.no_grad():
            target_value = self.target_critic(next_inputs)
        rewards = reward_result.vehicle_rewards.to(value.device)
        done_tensor = torch.full_like(rewards, float(done))
        losses = compute_training_losses(value, target_value, rewards, actor_output, action_tensor, self.config.gamma, done_tensor)
        self.critic_optimizer.zero_grad(set_to_none=True)
        losses.critic_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        self.actor_optimizer.step()
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            hard_update(self.target_critic, self.critic)
        return {
            "critic_loss": float(losses.critic_loss.detach().cpu()),
            "actor_loss": float(losses.actor_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
        }

    def train_episode(self, episode_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        state = self.env.reset(episode_index=episode_index)
        step_rows: list[dict[str, Any]] = []
        vehicle_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        episode_reward = 0.0
        violation_vehicle_ids: set[int] = set()
        violation_events = 0
        total_travel_time = 0.0
        total_cost = 0.0
        completed_trips = 0

        for step_index in range(self.env.max_steps):
            inputs = self.extractor(state)
            actor_output = self.actor(inputs)
            action_tensor = self.nash_policy.select(actor_output)
            if self.config.exploration_sigma > 0:
                action_tensor = action_tensor + torch.randn_like(action_tensor) * self.config.exploration_sigma
            action = Action(action_tensor)

            next_state, reward_result, done, info = self.env.step(action)
            next_inputs = self.extractor(next_state)

            value = self.critic(inputs)
            with torch.no_grad():
                target_value = self.target_critic(next_inputs)

            rewards = reward_result.vehicle_rewards.to(value.device)
            done_tensor = torch.full_like(rewards, float(done))

            losses = compute_training_losses(
                value=value,
                target_value=target_value,
                reward=rewards,
                actor_output=actor_output,
                action=action_tensor,
                gamma=self.config.gamma,
                done=done_tensor,
            )

            self.critic_optimizer.zero_grad(set_to_none=True)
            losses.critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad(set_to_none=True)
            losses.actor_loss.backward()
            self.actor_optimizer.step()

            self.update_count += 1
            if self.update_count % self.config.target_update_interval == 0:
                hard_update(self.target_critic, self.critic)

            episode_reward += float(reward_result.total_reward.detach().cpu())
            violation_events += int(reward_result.budget_violations.sum().item())
            violation_vehicle_ids.update(i for i, flag in enumerate(reward_result.budget_violations.tolist()) if flag)
            total_travel_time += float(reward_result.travel_times.sum().item())
            total_cost += float(reward_result.charging_costs.sum().item())
            completed_trips = int(info.get("completed_trips", completed_trips))

            for vehicle_metric in info.get("vehicle_metrics", []):
                vehicle_rows.append({"episode": episode_index, "step": step_index, **vehicle_metric})
            for edge_metric in info.get("edge_metrics", []):
                edge_rows.append({"episode": episode_index, "step": step_index, **edge_metric})

            step_rows.append({
                "episode": episode_index,
                "step": step_index,
                "reward": float(reward_result.total_reward.detach().cpu()),
                "mean_vehicle_reward": float(reward_result.vehicle_rewards.mean().detach().cpu()),
                "actor_loss": float(losses.actor_loss.detach().cpu()),
                "critic_loss": float(losses.critic_loss.detach().cpu()),
                "mean_td_error": float(losses.td_error.abs().mean().detach().cpu()),
                "budget_violations": int(reward_result.budget_violations.sum().item()),
                "travel_time_h": float(reward_result.travel_times.sum().item()),
                "travel_time_s": float(reward_result.travel_times.sum().item()) * 3600.0,
                "charging_cost": float(reward_result.charging_costs.sum().item()),
                "completed_trips": completed_trips,
                "sumo_time_s": float(info.get("sumo_time_s", 0.0) or 0.0),
            })

            state = next_state
            if done:
                break

        episode_row = {
            "episode": episode_index,
            "steps": len(step_rows),
            "total_reward": episode_reward,
            "mean_step_reward": episode_reward / max(1, len(step_rows)),
            "actor_loss_mean": sum(r["actor_loss"] for r in step_rows) / max(1, len(step_rows)),
            "critic_loss_mean": sum(r["critic_loss"] for r in step_rows) / max(1, len(step_rows)),
            "budget_violations": len(violation_vehicle_ids),
            "budget_violation_events": violation_events,
            "total_travel_time_h": total_travel_time,
            "total_travel_time_s": total_travel_time * 3600.0,
            "total_charging_cost": total_cost,
            "completed_trips": completed_trips,
            "target_update_count": self.update_count // max(1, self.config.target_update_interval),
        }
        if self.config.save_checkpoints and (
            (episode_index + 1) % self.config.checkpoint_interval == 0 or episode_index + 1 == self.config.episodes
        ):
            save_checkpoint(
                f"{self.config.output_dir}/checkpoints/episode_{episode_index + 1:05d}.pt",
                actor_state_dict=self.actor.state_dict(),
                critic_state_dict=self.critic.state_dict(),
                target_critic_state_dict=self.target_critic.state_dict(),
                actor_optimizer_state_dict=self.actor_optimizer.state_dict(),
                critic_optimizer_state_dict=self.critic_optimizer.state_dict(),
                episode=episode_index + 1,
                update_count=self.update_count,
                config=asdict(self.config),
            )
        return step_rows, vehicle_rows, edge_rows, episode_row
