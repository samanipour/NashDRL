from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nash_drl.data import Action
from nash_drl.environment.sumo_training import NashSUMOTrainingEnvironment
from nash_drl.features import StateFeatureExtractor
from ppo.models.ppo import PPOActorCritic
from nash_drl.routing import ActionToPathMapper

from nash_drl.training.checkpoint import save_checkpoint
from ppo.training.ppo_rollout import PPORolloutStep, compute_gae, build_rollout_batch


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    episodes: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.00
    ppo_epochs: int = 10
    minibatch_size: int = 16
    normalize_advantage: bool = False
    max_grad_norm: float = 0.50
    reward_scale: float = 1.0
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs/ppo_training"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


class PPOTrainer:
    """On-policy PPO trainer using total system reward as its only objective."""

    def __init__(
        self,
        env: NashSUMOTrainingEnvironment,
        policy: PPOActorCritic,
        mapper: ActionToPathMapper,
        config: PPOTrainingConfig,
    ) -> None:
        self.env = env
        self.policy = policy.to(config.device)
        self.mapper = mapper
        self.config = config
        self.device = torch.device(config.device)
        self.extractor = StateFeatureExtractor()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def _active_mask(self) -> Tensor:
        return torch.tensor(
            [float(v.current_trip_index < len(v.trips)) for v in self.env.problem.vehicles],
            dtype=torch.float32,
            device=self.device,
        )

    def _state_to_device(self, state):
        inputs = self.extractor(state)
        return type(inputs)(inputs.invariant.to(self.device), inputs.non_invariant.to(self.device))

    def collect_episode(self, episode_index: int) -> tuple[list[PPORolloutStep], dict[str, Any]]:
        state = self.env.reset(episode_index=episode_index)
        steps: list[PPORolloutStep] = []
        total_reward = 0.0
        benchmark_total_reward = 0.0
        violations = 0
        travel_time = 0.0
        charging_cost = 0.0
        violation_vehicle_ids: set[int] = set()
        violation_events = 0
        vehicle_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []

        for step_index in range(self.env.max_steps):
            active_mask = self._active_mask()
            inputs = self._state_to_device(state)
            with torch.no_grad():
                action_tensor, log_prob, _, value = self.policy.sample_action(
                    inputs, active_mask=active_mask, deterministic=False
                )
            action = Action(action_tensor.detach())
            next_state, reward_result, done, info = self.env.step(action)

            reward_scalar = reward_result.total_reward.to(self.device).reshape(()) * self.config.reward_scale
            steps.append(
                PPORolloutStep(
                    state_inputs=inputs,
                    action=action_tensor.detach(),
                    old_log_prob=log_prob.detach().reshape(()),
                    reward=reward_scalar.detach(),
                    value=value.detach().reshape(()),
                    done=torch.tensor(float(done), device=self.device),
                    active_mask=active_mask.detach(),
                )
            )

            total_reward += float(reward_result.total_reward.detach().cpu())
            benchmark_value = info.get("benchmark_reward_total")
            if benchmark_value is not None:
                benchmark_total_reward += float(benchmark_value)
            step_violations = int(reward_result.budget_violations.sum().item())
            violations += step_violations
            violation_events += step_violations
            violation_vehicle_ids.update(
                i for i, flag in enumerate(reward_result.budget_violations.tolist()) if flag
            )
            travel_time += float(reward_result.travel_times.sum().item())
            charging_cost += float(reward_result.charging_costs.sum().item())
            for vehicle_metric in info.get("vehicle_metrics", []):
                vehicle_rows.append({"episode": episode_index, "step": step_index, **vehicle_metric})
            for edge_metric in info.get("edge_metrics", []):
                edge_rows.append({"episode": episode_index, "step": step_index, **edge_metric})
            state = next_state
            if done:
                break

        # Bootstrap with V(s_T) only when the rollout is non-terminal.
        if steps and steps[-1].done.item() < 0.5:
            next_inputs = self._state_to_device(state)
            next_mask = self._active_mask()
            with torch.no_grad():
                next_value = self.policy(next_inputs).value
        else:
            next_value = torch.zeros((), device=self.device)

        rewards = torch.stack([s.reward for s in steps])
        values = torch.stack([s.value for s in steps])
        dones = torch.stack([s.done for s in steps])
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            next_value,
            self.config.gamma,
            self.config.gae_lambda,
        )
        batch = build_rollout_batch(
            steps,
            advantages,
            returns,
            self.device,
            normalize_advantage=self.config.normalize_advantage,
        )
        stats = self._update(batch)
        stats.update(
            {
                "total_reward": total_reward,
                "learning_total_reward": total_reward,
                "benchmark_total_reward": benchmark_total_reward,
                "mean_step_reward": total_reward / max(1, len(steps)),
                "budget_violations": len(violation_vehicle_ids),
                "budget_violation_events": violation_events,
                "total_travel_time_h": travel_time,
                "total_charging_cost": charging_cost,
                "steps": len(steps),
                "mean_advantage": float(advantages.mean().detach().cpu()),
                "vehicle_rows": vehicle_rows,
                "edge_rows": edge_rows,
            }
        )
        return steps, stats

    def _slice_inputs(self, batch, idx: Tensor):
        from nash_drl.data import NetworkInputs
        return NetworkInputs(
            batch.state_inputs.invariant[idx],
            batch.state_inputs.non_invariant[idx],
        )

    def _update(self, batch) -> dict[str, float]:
        size = batch.actions.shape[0]
        if size == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "actor_gradient_norm": 0.0, "critic_gradient_norm": 0.0}

        last_policy_loss = last_value_loss = last_entropy = last_kl = 0.0
        last_grad = 0.0
        for _ in range(max(1, self.config.ppo_epochs)):
            permutation = torch.randperm(size, device=self.device)
            for start in range(0, size, max(1, self.config.minibatch_size)):
                idx = permutation[start : start + self.config.minibatch_size]
                inputs = self._slice_inputs(batch, idx)
                actions = batch.actions[idx]
                old_log_probs = batch.old_log_probs[idx]
                returns = batch.returns[idx]
                advantages = batch.advantages[idx]
                masks = batch.active_masks[idx]

                log_probs, entropy, values = self.policy.evaluate_actions(
                    inputs, actions, active_mask=masks
                )
                ratios = torch.exp(log_probs - old_log_probs)
                unclipped = ratios * advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (values - returns).square().mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    ).detach().cpu()
                )
                self.optimizer.step()

                approx_kl = 0.5 * ((log_probs - old_log_probs) ** 2).mean()
                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_entropy = float(entropy_mean.detach().cpu())
                last_kl = float(approx_kl.detach().cpu())
                last_grad = grad_norm

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
            "actor_gradient_norm": last_grad,
            "critic_gradient_norm": last_grad,
        }

    def train_episode(self, episode_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        steps, stats = self.collect_episode(episode_index)
        step_rows: list[dict[str, Any]] = []
        vehicle_rows: list[dict[str, Any]] = list(stats.pop("vehicle_rows", []))
        edge_rows: list[dict[str, Any]] = list(stats.pop("edge_rows", []))

        for idx, step in enumerate(steps):
            step_rows.append(
                {
                    "episode": episode_index,
                    "step": idx,
                    "algorithm": "ppo",
                    "reward": float(step.reward.cpu()),
                    "old_log_prob": float(step.old_log_prob.cpu()),
                    "value": float(step.value.cpu()),
                }
            )

        episode_row = {
            "episode": episode_index,
            "algorithm": "ppo",
            "total_reward": stats["total_reward"],
            "learning_total_reward": stats.get("learning_total_reward", stats["total_reward"]),
            "benchmark_total_reward": stats.get("benchmark_total_reward", stats["total_reward"]),
            "mean_step_reward": stats["mean_step_reward"],
            "budget_violations": stats["budget_violations"],
            "budget_violation_rate": stats["budget_violations"] / max(1, len(self.env.problem.vehicles)),
            "budget_violation_events": stats["budget_violation_events"],
            "total_travel_time_h": stats["total_travel_time_h"],
            "total_charging_cost": stats["total_charging_cost"],
            "policy_loss_mean": stats["policy_loss"],
            "value_loss_mean": stats["value_loss"],
            "td_loss_mean": stats["value_loss"],
            "actor_loss_mean": stats["policy_loss"],
            "critic_loss_mean": stats["value_loss"],
            "entropy_mean": stats["entropy"],
            "approx_kl": stats["approx_kl"],
            "actor_gradient_norm_mean": stats["actor_gradient_norm"],
            "critic_gradient_norm_mean": stats["critic_gradient_norm"],
            "mean_advantage": stats["mean_advantage"],
            "steps": stats["steps"],
        }
        return step_rows, vehicle_rows, edge_rows, episode_row

    def save_checkpoint(self, path: str) -> None:
        save_checkpoint(
            path,
            algorithm="ppo",
            policy_state_dict=self.policy.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            config=self.config,
        )
