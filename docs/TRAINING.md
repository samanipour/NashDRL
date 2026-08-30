# NashDRL Training and Testing

## Training semantics

A NashDRL episode uses a fixed vehicle trip-set dataset. The dataset is generated or loaded once before training and is not changed between episodes. The episode horizon is derived from the dataset:

`steps_per_episode = max(len(vehicle.trips) for vehicle in vehicles)`.

At each step, the Actor receives the current global state and produces LQ parameters. The current mean action is optionally perturbed with Gaussian exploration noise and mapped deterministically to one route per active vehicle. The SUMO environment executes that trip leg to completion. The environment then computes the model reward, updates each vehicle's current stop, remaining trip count and remaining budget, and exposes the next state.

Vehicles with fewer trips become inactive after their trip-set is exhausted. Their state remains represented so the fixed `[N,F]`, `[N,N-1,F]` and `[N,F+E]` tensor contracts remain valid.

## Episode outputs

Each training run writes:

- `episode_results.csv`: one row per episode with total reward, mean losses, hard-constraint violations, travel time, charging cost and number of completed trips.
- `step_results.csv`: one row per environment/learning step.
- `vehicle_results.csv`: per-vehicle trip-level telemetry, reward, budget and route data.
- `edge_results.csv`: SUMO edge telemetry enriched with graph and charging/congestion parameters.
- `training_metadata.json`: dataset identity, seed, dimensions and derived episode horizon.
- `plots/*.png`: learning curves for reward, actor/critic losses, violations, travel time and charging cost.
- `checkpoints/*.pt`: Actor, Critic, Target Critic and optimizer state when checkpointing is enabled.

## Mock dataset reproducibility

Mock data are generated once per configured seed using clipped normal distributions for road length, capacity, speed, budget and trip-count selection. The graph contains a bidirectional ring plus reproducibly sampled extra edges so every generated origin/destination pair has a directed path.

The generated dataset is saved under `datasets/generated/mock_seed_<seed>/dataset.json` and copied into the training output directory for provenance.

## Run training

```bash
python scripts/train.py --config configs/experiments/small.yaml --mode mock --episodes 3
```

For a real dataset:

```bash
python scripts/train.py --config configs/default.yaml --mode real
```

Set `training.dataset_path` to the JSON dataset in real-data mode.

## Test a checkpoint

Set `evaluation.checkpoint` to a saved checkpoint or pass the checkpoint through the CLI. Testing runs the same fixed trip sets with exploration disabled.

```bash
python scripts/evaluate.py --config configs/default.yaml --mode mock --episodes 5
```

## Learning equations implemented here

For each step, the implementation uses:

`TD Target = r + gamma * V_slow(x')`

`TD Error = (V(x) + A(x,u)) - TD Target`

The Critic is optimized against the detached TD target, while the Actor is optimized through the LQ advantage term with Critic/target terms detached. The Target Critic is hard-synchronized from the Critic every configured number of optimization updates.
