# Reward regimes for the PPO-vs-NashDRL experiment

The same map, SUMO dynamics, traffic-flow state, trip sets, travel-time model, charging model, and episode horizon are used by both algorithms. The experiment permits algorithm-specific reward hyperparameters so that the two objectives are not artificially forced to behave identically.

## Piecewise vehicle reward

For vehicle i:

`R_regular = -(w_T*T_i + w_C*C_i)`

`R_violation = -P`

Therefore a violation is individually preferred by a total-reward optimizer when:

`P < w_T*T_i + w_C*C_i`.

This is a necessary vehicle-level condition. A system-level PPO advantage additionally requires that the aggregate improvement for the sacrificed/benefited vehicles outweighs the losses imposed on the rest of the population.

## Why the previous medium result had the wrong reward ordering

The previous experiment used `P=500` for both algorithms. The observed vehicle charging costs were mostly far below 500, so a violating vehicle received a much worse reward (`-500`) than its regular outcome (typically around `-(charging cost + travel time)`). Consequently, the reward function strongly discouraged sacrifice even for PPO. PPO still produced violations because policy search is not a hard feasibility guarantee, but those violations were expensive in the objective instead of reward-preferred.

## Current medium experiment

The medium experiment uses:

- `reward.common`: P=100, algorithm-neutral benchmark.
- `reward.nash_drl`: P=500, constraint-dominant Nash learning.
- `reward.ppo`: P=10, sacrifice-permitting PPO learning.

The PPO value is deliberately below the regular-loss scale of a meaningful subset of routes. This places PPO in the regime where it can rationally sacrifice some vehicles when that improves aggregate system reward. NashDRL uses a much larger penalty and a budget-aware mapper, so its learning process strongly favors budget-feasible behavior.

The condition should be checked empirically for every generated dataset; use `scripts/analyze_reward_regime.py` after a run. The reported `violation_break_even_penalty` in `vehicle_results.csv` is the empirical threshold `w_T*T_i + w_C*C_i` for each realized vehicle-trip outcome.

## Reporting

`total_reward` / `learning_total_reward` show the objective actually optimized by each algorithm. `benchmark_total_reward` applies `reward.common` to both policies and should be used for an algorithm-neutral final comparison. Physical metrics such as violations, travel time, and charging cost remain directly comparable in all cases.
