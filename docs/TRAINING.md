# NashDRL Training and Testing

## Training semantics

A NashDRL dataset is generated once in `mock` mode or loaded once in `real` mode. The vehicle trip-sets are therefore fixed across all learning episodes. The episode horizon is derived from the dataset:

`steps_per_episode = max(len(vehicle.trips) for vehicle in vehicles)`.

At each step, the Actor processes the current state and the deterministic route mapper produces an action route for every active vehicle. SUMO/TraCI executes the current trip leg, the environment computes the NashDRL model reward, advances vehicle trip indices/budgets, and returns the next state.

The loss calculation follows the Section-4 decomposition:

`Q(x,u) = V(x) + A(x,u)`

`TD target = r + gamma * V_slow(x')`

`TD error = (V(x) + A(x,u)) - TD target`

The executed action `u` is detached before loss computation. This is important: the sampled action is treated as fixed while the Actor learns the mean `mu` through `z = u - mu`.

For the Critic update, `A(x,u)` is detached. For the Actor update, `V(x)` and the TD target are detached. The Target Critic is never updated by backpropagation and is hard-copied from the Main Critic every `target_update_interval` optimization updates.

## Replay and optimization

Training uses a small replay buffer by default because the supplied reference implementation computes the Nash loss over batches of transitions. The buffer stores the two neural input streams, executed action, per-agent reward, per-agent terminal flag, and active-agent mask.

`replay_warmup` controls how many transitions must exist before updates begin. `replay_batch_size` controls the number of transitions per optimization update and `updates_per_step` controls how many batches are learned after each environment step.

The loss is normalized over active agents only. Vehicles whose trip-set has already finished do not contribute artificial zero-reward learning targets to the Actor/Critic updates.

## Observation normalization

The six Section-3.2.1 per-agent features are normalized using dataset-level scales:

- node identifiers / `(N_nodes - 1)`;
- remaining trips / maximum trip count;
- remaining budget / maximum initial budget;
- free-flow speed / maximum free-flow speed.

Global edge flows are normalized by the number of vehicles. This keeps feature magnitudes comparable across small and large experiments.

## Exploration

Exploration is Gaussian noise added to the detached Actor mean action. `exploration_sigma` is linearly decayed toward `exploration_sigma_final` over `exploration_decay_episodes`.

## Route mapping

The Actor emits real-valued edge weights. Dijkstra converts them into positive traversal costs using a smooth `softplus(-weight)` transformation. This avoids the previous `1/max(weight, eps)` saturation in which every negative Actor output effectively became the same huge cost.

The route mapper remains deterministic and non-trainable.

## Reward and hard constraints

The model reward remains:

`R_i = -w_T T_i - w_C C_i`, when `C_i <= B_i`

`R_i = -P`, when `C_i > B_i`.

The large training configuration intentionally uses a stronger `P` than the illustrative paper example. This is a training configuration choice, not a change to the mathematical definition. For strict reproduction of the paper's illustrative setting, use `budget_penalty: 100.0`.

SUMO travel-time telemetry is retained for diagnostics. By default training reward uses the analytical congestion/travel-time equation from the model rather than raw SUMO travel time. Set `environment.use_model_travel_time: false` to use SUMO-reported travel time for an alternative experiment.

## Mock dataset generation

Mock numeric attributes are sampled from reproducible clipped normal distributions. Trip-set length is sampled from a normal distribution and clipped to the configured bounds.

Vehicle budgets are generated after trip-sets and the graph are known. The generator first computes a deterministic shortest-path reference assignment and its model charging cost, then samples each budget from a normal distribution conditioned to be at least that reference cost. Consequently, zero hard-constraint violations are feasible for every generated vehicle under at least one known reference routing policy.

The same dataset is reused in every learning episode.

## Training outputs

The training run writes:

- `episode_results.csv`: one row per episode;
- `step_results.csv`: one row per learning/environment step;
- `vehicle_results.csv`: trip-level vehicle telemetry and model quantities;
- `edge_results.csv`: edge telemetry and model parameters;
- `training_metadata.json`: experiment and dataset provenance;
- `checkpoints/*.pt`: Actor/Critic/Target Critic and optimizer states;
- `plots/*.png`: raw and rolling-mean learning curves.

Important diagnostics now include `actor_gradient_norm`, `critic_gradient_norm`, `mean_advantage`, `mean_td_error`, `replay_size`, `updates`, and `exploration_sigma`.

## Run training

```bash
python scripts/train.py --config configs/experiments/small.yaml --mode mock --episodes 100
```

For the larger experiment:

```bash
python scripts/train.py --config configs/experiments/large.yaml --mode mock --episodes 1000
```

## Test a checkpoint

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --mode mock \
    --episodes 5 \
    --checkpoint outputs/training_large/checkpoints/episode_01000.pt
```
