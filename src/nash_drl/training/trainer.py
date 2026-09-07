from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from nash_drl.data import Action, NetworkInputs
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment
from nash_drl.features import StateFeatureExtractor
from nash_drl.game import AnalyticalNashPolicy
from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork
from nash_drl.routing import ActionToPathMapper

from .checkpoint import save_checkpoint
from .losses import LossResult, compute_training_losses
from .rollout import ReplayBuffer, TrainingBatch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    episodes: int = 10
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_sigma: float = 0.10
    exploration_sigma_final: float = 0.02
    exploration_decay_episodes: int = 1000
    reward_scale: float = 1.0
    max_grad_norm: float = 5.0
    replay_enabled: bool = True
    replay_capacity: int = 10000
    replay_batch_size: int = 32
    replay_warmup: int = 32
    updates_per_step: int = 1
    target_update_interval: int = 100
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs/training"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


class NashDRLTrainer:
    """Online episodic NashDRL trainer with a small replay buffer.

    The optimizer follows the Section-4 decomposition exactly:
      Q(x,u) = V(x) + A(x,u)
      TD target = r + gamma * V_slow(x')

    Critic update: A is detached.
    Actor update: V and the target are detached.

    The executed action is always detached from the Actor graph so that
    ``u`` remains fixed while ``mu`` is learned through ``z = u - mu``.
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
        self.device = torch.device(config.device)
        self.extractor = StateFeatureExtractor()
        self.nash_policy = AnalyticalNashPolicy()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.update_count = 0
        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed) if config.replay_enabled else None
        self.noise_generator = torch.Generator(device=self.device)
        self.noise_generator.manual_seed(config.seed)

    def current_exploration_sigma(self, episode_index: int) -> float:
        if self.config.exploration_decay_episodes <= 0:
            return self.config.exploration_sigma_final
        progress = min(1.0, max(0.0, episode_index / self.config.exploration_decay_episodes))
        return float(
            self.config.exploration_sigma
            + (self.config.exploration_sigma_final - self.config.exploration_sigma) * progress
        )

    def select_action(self, state, training: bool = True, episode_index: int = 0) -> Action:
        inputs = self.extractor(state)
        params = self.actor(inputs)
        # The environment executes a sampled action, not a differentiable node
        # in the Actor graph.  This is essential for the LQ advantage update.
        action_tensor = self.nash_policy.select(params).detach()
        if training:
            sigma = self.current_exploration_sigma(episode_index)
            if sigma > 0:
                noise = torch.randn(
                    action_tensor.shape,
                    device=action_tensor.device,
                    dtype=action_tensor.dtype,
                    generator=self.noise_generator,
                ) * sigma
                action_tensor = action_tensor + noise
        return Action(action_tensor.detach())

    def _optimize_batch(self, batch: TrainingBatch) -> tuple[LossResult, float, float]:
        actor_output = self.actor(batch.state_inputs)
        value = self.critic(batch.state_inputs)
        with torch.no_grad():
            target_value = self.target_critic(batch.next_inputs)

        rewards = batch.rewards * self.config.reward_scale
        losses = compute_training_losses(
            value=value,
            target_value=target_value,
            reward=rewards,
            actor_output=actor_output,
            action=batch.actions.detach(),
            gamma=self.config.gamma,
            done=batch.done,
            agent_mask=batch.agent_mask,
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        losses.critic_loss.backward()
        critic_grad = float(
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.max_grad_norm
            ).detach().cpu()
        )
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        actor_grad = float(
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            ).detach().cpu()
        )
        self.actor_optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_critic.hard_update_from(self.critic)
        return losses, actor_grad, critic_grad

    def _single_transition_batch(
        self,
        inputs: NetworkInputs,
        action: Tensor,
        rewards: Tensor,
        next_inputs: NetworkInputs,
        done: Tensor,
        agent_mask: Tensor,
    ) -> TrainingBatch:
        return TrainingBatch(
            state_inputs=NetworkInputs(
                inputs.invariant.unsqueeze(0), inputs.non_invariant.unsqueeze(0)
            ),
            actions=action.detach().unsqueeze(0),
            rewards=rewards.detach().unsqueeze(0),
            next_inputs=NetworkInputs(
                next_inputs.invariant.unsqueeze(0), next_inputs.non_invariant.unsqueeze(0)
            ),
            done=done.detach().unsqueeze(0),
            agent_mask=agent_mask.detach().unsqueeze(0),
        )

    def train_step(self) -> dict[str, float]:
        """Backward-compatible one-step analytical-environment update for tests."""
        state = self.env.reset()
        inputs = self.extractor(state)
        actor_output = self.actor(inputs)
        action_tensor = self.nash_policy.select(actor_output).detach()
        if self.config.exploration_sigma > 0:
            action_tensor = action_tensor + torch.randn_like(action_tensor) * self.config.exploration_sigma
        action = Action(action_tensor.detach())
        paths = self.mapper.map(action, state)
        next_state, reward_result, done, _ = self.env.step(paths)
        next_inputs = self.extractor(next_state)
        n = action_tensor.shape[0]
        rewards = reward_result.vehicle_rewards.to(self.device)
        done_tensor = torch.full_like(rewards, float(done))
        mask = torch.ones_like(rewards)
        batch = self._single_transition_batch(
            self._transition_to_device(inputs),
            action_tensor.to(self.device),
            rewards,
            self._transition_to_device(next_inputs),
            done_tensor,
            mask,
        )
        losses, actor_grad, critic_grad = self._optimize_batch(batch)
        return {
            "critic_loss": float(losses.critic_loss.detach().cpu()),
            "actor_loss": float(losses.actor_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "actor_gradient_norm": actor_grad,
            "critic_gradient_norm": critic_grad,
        }

    def _transition_to_device(self, x: NetworkInputs) -> NetworkInputs:
        return NetworkInputs(x.invariant.to(self.device), x.non_invariant.to(self.device))

    def _run_updates(self, fallback_batch: TrainingBatch | None = None) -> tuple[LossResult | None, float, float]:
        if self.replay is None:
            if fallback_batch is None:
                return None, 0.0, 0.0
            return self._optimize_batch(fallback_batch)
        if len(self.replay) < self.config.replay_warmup:
            return None, 0.0, 0.0

        last_losses: LossResult | None = None
        actor_grad = critic_grad = 0.0
        for _ in range(max(1, self.config.updates_per_step)):
            batch = self.replay.sample(self.config.replay_batch_size, self.device)
            last_losses, actor_grad, critic_grad = self._optimize_batch(batch)
        return last_losses, actor_grad, critic_grad

    def train_episode(
        self, episode_index: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        td_losses: list[float] = []

        for step_index in range(self.env.max_steps):
            inputs = self.extractor(state)
            # Active mask belongs to the state BEFORE this action is executed.
            active_mask = torch.tensor(
                [
                    float(v.current_trip_index < len(v.trips))
                    for v in self.env.problem.vehicles
                ],
                dtype=torch.float32,
                device=self.device,
            )

            actor_output = self.actor(inputs)
            action_tensor = self.nash_policy.select(actor_output).detach()
            sigma = self.current_exploration_sigma(episode_index)
            if sigma > 0:
                action_tensor = action_tensor + torch.randn(
                    action_tensor.shape,
                    device=action_tensor.device,
                    dtype=action_tensor.dtype,
                    generator=self.noise_generator,
                ) * sigma
            action_tensor = action_tensor.detach()
            action = Action(action_tensor)

            next_state, reward_result, done, info = self.env.step(action)
            next_inputs = self.extractor(next_state)
            rewards = reward_result.vehicle_rewards.to(self.device)
            agent_done = torch.as_tensor(
                info.get("agent_done", [done] * self.env.num_agents),
                dtype=torch.float32,
                device=self.device,
            )

            if self.replay is not None:
                self.replay.add(
                    self._transition_to_device(inputs),
                    action_tensor,
                    rewards,
                    self._transition_to_device(next_inputs),
                    agent_done,
                    active_mask,
                )
                losses, actor_grad, critic_grad = self._run_updates()
            else:
                fallback = self._single_transition_batch(
                    self._transition_to_device(inputs),
                    action_tensor,
                    rewards,
                    self._transition_to_device(next_inputs),
                    agent_done,
                    active_mask,
                )
                losses, actor_grad, critic_grad = self._run_updates(fallback)

            raw_reward = float(reward_result.total_reward.detach().cpu())
            episode_reward += raw_reward
            violation_events += int(reward_result.budget_violations.sum().item())
            violation_vehicle_ids.update(
                i for i, flag in enumerate(reward_result.budget_violations.tolist()) if flag
            )
            total_travel_time += float(reward_result.travel_times.sum().item())
            total_cost += float(reward_result.charging_costs.sum().item())
            completed_trips = int(info.get("completed_trips", completed_trips))

            for vehicle_metric in info.get("vehicle_metrics", []):
                vehicle_rows.append({"episode": episode_index, "step": step_index, **vehicle_metric})
            for edge_metric in info.get("edge_metrics", []):
                edge_rows.append({"episode": episode_index, "step": step_index, **edge_metric})

            if losses is not None:
                actor_loss_value = float(losses.actor_loss.detach().cpu())
                critic_loss_value = float(losses.critic_loss.detach().cpu())
                actor_losses.append(actor_loss_value)
                critic_losses.append(critic_loss_value)
                td_losses.append(critic_loss_value)
                td_error_value = float(losses.td_error.abs().mean().detach().cpu())
                mean_advantage = float(losses.advantage.mean().detach().cpu())
            else:
                actor_loss_value = 0.0
                critic_loss_value = 0.0
                td_error_value = 0.0
                mean_advantage = 0.0

            step_rows.append(
                {
                    "episode": episode_index,
                    "step": step_index,
                    "reward": raw_reward,
                    "learning_reward_mean": float((rewards * self.config.reward_scale).mean().cpu()),
                    "mean_vehicle_reward": float(reward_result.vehicle_rewards.mean().cpu()),
                    # In the exact Section-4 squared TD formulation, the
                    # Actor and Critic objectives have the same scalar value;
                    # only gradient paths differ because of stop-gradient.
                    "actor_loss": actor_loss_value,
                    "critic_loss": critic_loss_value,
                    "td_loss": critic_loss_value,
                    "mean_td_error": td_error_value,
                    "mean_advantage": mean_advantage,
                    "actor_gradient_norm": actor_grad,
                    "critic_gradient_norm": critic_grad,
                    "budget_violations": int(reward_result.budget_violations.sum().item()),
                    "budget_violation_rate": float(reward_result.budget_violations.float().sum().item()) / max(1, int(active_mask.sum().item())),
                    "travel_time_h": float(reward_result.travel_times.sum().item()),
                    "travel_time_s": float(reward_result.travel_times.sum().item()) * 3600.0,
                    "charging_cost": float(reward_result.charging_costs.sum().item()),
                    "completed_trips": completed_trips,
                    "active_agents": int(active_mask.sum().item()),
                    "replay_size": len(self.replay) if self.replay is not None else 0,
                    "updates": self.update_count,
                    "exploration_sigma": sigma,
                    "sumo_time_s": float(info.get("sumo_time_s", 0.0) or 0.0),
                }
            )

            state = next_state
            if done:
                break

        episode_row = {
            "episode": episode_index,
            "steps": len(step_rows),
            "total_reward": episode_reward,
            "mean_step_reward": episode_reward / max(1, len(step_rows)),
            "actor_loss_mean": sum(actor_losses) / max(1, len(actor_losses)),
            "critic_loss_mean": sum(critic_losses) / max(1, len(critic_losses)),
            "td_loss_mean": sum(td_losses) / max(1, len(td_losses)),
            "actor_gradient_norm_mean": sum(r["actor_gradient_norm"] for r in step_rows) / max(1, len(step_rows)),
            "critic_gradient_norm_mean": sum(r["critic_gradient_norm"] for r in step_rows) / max(1, len(step_rows)),
            "budget_violations": len(violation_vehicle_ids),
            "budget_violation_rate": len(violation_vehicle_ids) / max(1, len(self.env.problem.vehicles)),
            "budget_violation_events": violation_events,
            "total_travel_time_h": total_travel_time,
            "total_travel_time_s": total_travel_time * 3600.0,
            "total_charging_cost": total_cost,
            "completed_trips": completed_trips,
            "target_update_count": self.update_count // max(1, self.config.target_update_interval),
            "updates": self.update_count,
            "replay_size": len(self.replay) if self.replay is not None else 0,
            "exploration_sigma": self.current_exploration_sigma(episode_index),
        }

        if self.config.save_checkpoints and (
            (episode_index + 1) % self.config.checkpoint_interval == 0
            or episode_index + 1 == self.config.episodes
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
