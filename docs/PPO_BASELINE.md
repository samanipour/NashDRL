# PPO Baseline for NashDRL Comparison

## Purpose

This module provides a conventional Proximal Policy Optimization (PPO) actor-critic baseline that uses the same world, state, route representation, SUMO/TraCI execution, reward equations, trip sets, and reporting format as NashDRL.

The original PPO paper describes alternating data collection and stochastic optimization of a clipped surrogate objective. PPO is an actor-critic policy-gradient method; for continuous actions the actor defines a probability distribution and the critic estimates the state value. The clipped objective limits the policy ratio update. The project follows that baseline structure rather than reusing the NashDRL LQ objective.

## Important terminology

The statement that PPO has "no advantage" is not literally correct for the standard PPO algorithm. PPO normally uses an estimate of the policy advantage in its clipped policy objective; this project uses Generalized Advantage Estimation (GAE). What PPO does **not** use is the NashDRL **LQ/game-theoretic advantage function** and its parameters `P11`, `P12`, `P22`, and `Psi`.

Therefore:

- NashDRL advantage: analytical LQ/game-theoretic `A_theta(x,u)` used to represent local equilibrium behavior.
- PPO advantage: standard policy-gradient estimator used only as a scalar weighting term in the clipped PPO surrogate objective.

## Common environment and state

Both algorithms consume the same state feature representation:

- invariant: `[N,N-1,F]`
- non-invariant: `[N,F+E]`

Both operate on the same `[N,E]` edge-weight action representation and the same deterministic action-to-path mapper. The same SUMO traffic-flow observations are used in each state. The road topology remains a routing/environment object and is not directly encoded into the neural-network input.

Both use the same reward model from NashDRL-Model-V12, including travel time, charging cost, and the hard budget penalty.

## PPO architecture

```text
state [N,N-1,F] + [N,F+E]
             |
        Deep Sets + ego/global encoder
             |
      +---------------------+
      |                     |
      v                     v
  Policy actor          Value critic
      |                     |
 Gaussian mean/std       scalar V(s)
      |
 action [N,E]
      |
 SUMO route execution
      |
 total system reward
```

The PPO critic is scalar because the baseline objective is explicitly the total system reward across all vehicles. NashDRL retains the per-agent value decomposition required by its model.

## PPO objective

For collected transitions, PPO computes:

`delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

and the GAE recursion to obtain a standard policy-gradient advantage estimate.

The clipped surrogate is:

`L_clip = mean(min(ratio*A, clip(ratio, 1-epsilon, 1+epsilon)*A))`

where `ratio = exp(log_pi_new - log_pi_old)`.

The optimization loss used by the code is:

`policy_loss + value_coef * value_loss - entropy_coef * entropy`

with multiple minibatch epochs over the on-policy rollout.

## Files

- `src/ppo/models/ppo.py`: PPO Gaussian actor + scalar critic.
- `src/ppo/training/ppo_rollout.py`: on-policy rollout storage and GAE.
- `src/ppo/training/ppo_trainer.py`: PPO collection and clipped-objective updates.
- `src/ppo/training/ppo_runner.py`: dataset/environment construction and training reports.
- `src/ppo/evaluation/ppo_evaluator.py`: deterministic PPO evaluation.
- `src/ppo/evaluation/ppo_runner.py`: PPO evaluation orchestration.
- `scripts/train_ppo.py`: dedicated PPO command-line entry point.
- `scripts/compare_algorithms.py`: episode-by-episode NashDRL/PPO comparison CSV.

## Reproducibility

`configs/experiments/medium.yaml` and `configs/experiments/medium.yaml` intentionally share the same `mock_data` block and seed. Running both therefore reconstructs the same mock graph, vehicles, and fixed trip sets. Keep these sections identical when designing paired experiments.

## Interpretation of budget violations

PPO is deliberately not given a constraint-specific objective. It maximizes the total system return supplied by the common reward model. It is therefore expected that PPO may discover policies that increase aggregate reward while causing some vehicles to violate their budget, particularly in scenarios where the reward trade-off makes such a solution attractive.
