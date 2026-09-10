# NashDRL

NashDRL is a research implementation of a multi-agent electric autonomous vehicle (EAV) routing problem formulated around hard budget constraints, shared road resources, congestion, demand-dependent charging prices, and a Nash/game-theoretic deep reinforcement learning architecture.

This README is the project-level technical guide. It maps the repository implementation to our theoretical models and describes how the current simulation, environment, neural networks, training, testing, mock-data generation, reporting, and visualization components work together.


> The current repository is an implementation of the specified architecture, but some research components remain explicit extension points. In particular, `game/nash_solver.py` currently exposes the analytical-policy interface and returns `mu`; a future implementation can replace that policy selection with the fully derived analytical equilibrium solution without changing the Actor/Critic interfaces.

---

# 1. Project directories organization and their applications in the project

The repository follows a layered architecture in which the mathematical domain model is separated from runtime tensor structures, simulation, routing, feature construction, neural networks, game-theoretic computation, training, evaluation, and visualization.

```text
NashDRL/
│
├── pyproject.toml
├── README.md│
├── checkpoints/
├── configs/
│   ├── default.yaml
│   ├── environment.yaml
│   ├── network.yaml
│   ├── training.yaml
│   └── experiments/
│       ├── small.yaml
│       ├── medium.yaml
│       └── large.yaml
│
├── datasets/
│   ├── generated/
│   ├── maps/
│   └── scenarios/
│
├── docs/
│   ├── architecture.md
│   ├── CONFIGURATION.md
│   ├── DATA_MODEL.md
│   ├── development.md
│   ├── FILE_MANIFEST.txt
│   ├── module-guide.md
│   ├── NETWORK_DESIGN.md
│   ├── PROJECT_STRUCTURE.md
│   ├── README.md
│   ├── SIMULATION.md
│   ├── tensor-contracts.md
│   ├── TESTING.md
│   ├── TRAINING.md
│   └── TRAINING_DIAGNOSTICS.md
│
├── logs/
├── outputs/
├── scripts/
│   ├── evaluate.py
│   ├── infer.py
│   ├── simulate.py
│   ├── train.py
│   └── visualize.py
│
├── src/
│   └── nash_drl/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       │
│       ├── domain/
│       ├── data/
│       ├── environment/
│       ├── routing/
│       ├── features/
│       ├── models/
│       ├── game/
│       ├── training/
│       ├── evaluation/
│       ├── visualization/
│       └── utils/
│
└── tests/
    ├── fixtures/
    ├── integration/
    └── unit/
```

## Architectural dependency flow

The intended dependency direction is:

```text
                 ┌────────────────────┐
                 │      CLI/scripts    │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Training / Testing │
                 └──────┬─────┬───────┘
                        │     │
               ┌────────▼┐   │
               │ Models  │   │
               └────┬─────┘   │
                    │         │
              ┌─────▼──────┐  │
              │  Features  │  │
              └─────┬──────┘  │
                    │         │
              ┌─────▼─────────▼─────┐
              │ Runtime Data / State │
              └─────┬─────────┬─────┘
                    │         │
             ┌──────▼───┐ ┌──▼─────────┐
             │ Routing  │ │ Environment│
             └──────┬───┘ └──┬─────────┘
                    │         │
                    └────┬────┘
                         │
                   ┌─────▼─────┐
                   │   Domain  │
                   └───────────┘
```

`domain/` contains mathematical entities and definitions and does not depend on PyTorch. `data/` translates those entities into compact runtime structures and tensor contracts. `features/` converts state into the neural-network streams. `models/` performs inference. `routing/` converts the continuous edge-weight action into valid routes. `environment/` executes those routes and evaluates travel time, charging cost, congestion, and reward. `training/` orchestrates the Actor/Critic optimization.

---

## 1.1 Project source files and their purpose

### Root files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, Python version constraint, runtime/dev dependencies, executable entry points, pytest and Ruff configuration. |
| `README.md` | This project-level architecture, formulation, execution, configuration, and usage guide. |

### `configs/`

| File | Purpose |
|---|---|
| `configs/default.yaml` | Complete baseline configuration containing project, simulation, mock-data, network, environment, reward, training, and evaluation settings. |
| `configs/environment.yaml` | Focused environment/SUMO configuration for model-development and environment experiments. |
| `configs/network.yaml` | Focused neural-network architecture configuration. |
| `configs/training.yaml` | Focused training configuration. |
| `configs/experiments/small.yaml` | Small reproducible training/simulation benchmark for smoke tests and development. |
| `configs/experiments/medium.yaml` | Medium benchmark deliberately constructed to produce a non-trivial budget-constraint learning signal. |
| `configs/experiments/large.yaml` | Larger benchmark for extended training and scalability experiments. |

### `scripts/`

| File | Purpose |
|---|---|
| `scripts/simulate.py` | Runs the simulation-only workflow through the CLI. |
| `scripts/train.py` | Starts episodic Actor/Critic training. |
| `scripts/evaluate.py` | Runs testing/evaluation of a policy/checkpoint. |
| `scripts/infer.py` | Inference entry point; currently delegates to evaluation behavior. |
| `scripts/visualize.py` | Starts simulation with visualization enabled. |

### `src/nash_drl/` core

| File | Purpose | Important interfaces/functions |
|---|---|---|
| `src/nash_drl/__init__.py` | Package marker and package namespace. | Package import. |
| `src/nash_drl/config.py` | YAML configuration loading. | `load_yaml()` |
| `src/nash_drl/cli.py` | Command-line orchestration. Converts CLI arguments into config changes and starts simulation/training/evaluation. | `_bool()`, `simulate()`, `train()`, `evaluate()`, `infer()`, `visualize()` |

### `src/nash_drl/domain/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `domain/__init__.py` | Exposes domain classes as package API. | Imports `Edge`, `DirectedGraph`, `Trip`, `Vehicle`, `ChargingStationModel`, `ProblemDefinition`. |
| `domain/graph.py` | Directed road-network model. | `Edge`, `DirectedGraph`; `DirectedGraph.num_edges`. |
| `domain/vehicle.py` | Heterogeneous EAV model and trip progress. | `Vehicle`, `next_destination`, `final_destination`, `remaining_trip_count`. |
| `domain/trip.py` | One origin-to-destination trip leg. | `Trip(origin, destination)`. |
| `domain/charging.py` | Demand-dependent charging-station price model. | `ChargingStationModel.unit_price()`. |
| `domain/problem.py` | Aggregates the graph and vehicle population. | `ProblemDefinition(graph, vehicles)`. |

### `src/nash_drl/data/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `data/__init__.py` | Public runtime-data API. | Exposes CSR/state/action/batch/transition classes. |
| `data/csr_map.py` | CSR graph representation plus parallel edge properties. | `CSRMap.from_graph()`, `outgoing_edge_ids()`. |
| `data/state.py` | Explicit environment state and tensor contract. | `GlobalState.validate()`. |
| `data/action.py` | Continuous edge-weight action and mapped paths. | `Action`, `Action.validate()`, `Paths`. |
| `data/batch.py` | Network-input container for batched training/evaluation. | `NetworkInputs`, `validate()`. |
| `data/transition.py` | Transition representation used by learning infrastructure. | `Transition`. |
| `data/mock_dataset.py` | Reproducible stochastic mock-world generator and JSON persistence. | `MockDataConfig`, `MockDatasetGenerator.generate()`, `save()`, `load_problem()`. |

### `src/nash_drl/environment/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `environment/__init__.py` | Environment package API. | Public imports. |
| `environment/env.py` | Analytical environment independent of SUMO. | `EnvironmentConfig`, `NashEnvironment.reset()`, `step()`. |
| `environment/simulator.py` | Lightweight non-SUMO road simulator/dataflow utility. | `SimulationState`, `RoadSimulator.apply_paths()`. |
| `environment/congestion.py` | Implements the congestion multiplier. | `congestion_multiplier()`. |
| `environment/travel_time.py` | Implements the paper travel-time equation. | `edge_travel_time()`. |
| `environment/energy_cost.py` | Implements charging price and path energy/charging cost. | `charging_price()`, `edge_energy_cost()`. |
| `environment/reward.py` | Implements the paper piecewise reward and violation outcome. | `RewardConfig`, `RewardModel.compute()`. |
| `environment/simulation.py` | Simulation-only orchestration, dataset preparation, SUMO execution and analytical reports. | `SimulationRunner.run()`. |
| `environment/sumo.py` | SUMO network/route/config construction and TraCI execution. | `SumoScenarioBuilder`, `run_sumo()`, `run_sumo_trip()`. |
| `environment/sumo_training.py` | Episode/step training environment over SUMO. One environment step corresponds to one active trip leg. | `NashSUMOTrainingEnvironment.reset()`, `step()`. |

### `src/nash_drl/routing/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `routing/__init__.py` | Public routing API. | `ActionToPathMapper`, `DijkstraMapper`, `GreedyMapper`. |
| `routing/mapper.py` | Abstract interface for deterministic action-to-path conversion. | `ActionToPathMapper.map()`. |
| `routing/dijkstra.py` | Deterministic shortest-path conversion from edge weights to valid paths. | `DijkstraMapper.map()`, `_edge_cost()`, `_shortest_path()`. |
| `routing/greedy.py` | Alternative greedy route mapper. | `GreedyMapper.map()`. |
| `routing/constraints.py` | Generic path shape/edge-count validation. | `validate_paths()`. |

### `src/nash_drl/features/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `features/__init__.py` | Public feature API. | Public imports. |
| `features/agent_features.py` | Constructs the six semantic per-agent features and applies normalization. | `encode_agent_features()`. |
| `features/edge_features.py` | Converts edge flows into model input representation. | `edge_flow_features()`. |
| `features/deep_set_input.py` | Validates permutation-invariant tensor shape. | `validate_deep_set_input()`. |
| `features/extractor.py` | Builds the two neural streams from `GlobalState`. | `StateFeatureExtractor.extract()`. |

### `src/nash_drl/models/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `models/__init__.py` | Public neural-network API. | Actor/Critic/Target Critic/Deep Set/LQ Advantage exports. |
| `models/deep_sets.py` | Shared Deep Sets rival embedding and invariant aggregation. | `DeepSetEncoder.forward()`. |
| `models/actor.py` | LQ parameterizing Actor. | `ActorNetwork.forward()`, `ActorOutput`. |
| `models/critic.py` | Per-agent value network. | `CriticNetwork.forward()`. |
| `models/target_critic.py` | Frozen copy of Critic used for stable targets. | `TargetCriticNetwork.forward()`, `hard_update_from()`. |
| `models/lq_advantage.py` | Computes the LQ local advantage from Actor parameters and executed actions. | `LQAdvantage.forward()`. |
| `models/common/mlp.py` | Reusable fully connected MLP. | `MLP.forward()`. |
| `models/common/activations.py` | Positive-definiteness transformation for P11/P22. | `strictly_positive()`. |
| `models/common/initialization.py` | Linear-layer initialization. | `initialize_linear_layers()`. |
| `models/common/__init__.py` | Common-model package API. | Public imports. |

### `src/nash_drl/game/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `game/__init__.py` | Game-theoretic package API. | Public exports. |
| `game/action_distribution.py` | Exploration noise over continuous edge-weight actions. | `add_gaussian_exploration()`. |
| `game/lq_game.py` | Reserved module for higher-level LQ game utilities; current implementation is intentionally minimal. | Extension point. |
| `game/nash_solver.py` | Analytical Nash/LQ policy-selection interface. | `AnalyticalNashPolicy.select()`. |

### `src/nash_drl/training/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `training/__init__.py` | Training package API. | Public exports. |
| `training/experiment.py` | Training entry orchestration plus learning-curve generation. | `run_training()`. |
| `training/runner.py` | Builds dataset, environment, networks, trainer and persists training outputs. | `TrainingRunner.run()`, `_prepare_dataset()`. |
| `training/trainer.py` | Episode/step learning loop and Actor/Critic optimization. | `NashDRLTrainer.train_episode()`, `_optimize_batch()`, `_run_updates()`. |
| `training/rollout.py` | Replay buffer and batched transition structures. | `ReplayBuffer.add()`, `sample()`. |
| `training/losses.py` | TD target and NashDRL actor/critic loss calculation. | `compute_td_target()`, `compute_training_losses()`. |
| `training/optimizer.py` | Optimizer construction utility. | `make_adam()`. |
| `training/target_update.py` | Generic hard/soft target-update utility. | `hard_update()`, `soft_update()`. |
| `training/checkpoint.py` | Model/training-state serialization. | `save_checkpoint()`, `load_checkpoint()`. |

### `src/nash_drl/evaluation/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `evaluation/__init__.py` | Evaluation package API. | Public exports. |
| `evaluation/evaluator.py` | Runs one or multiple test episodes and summarizes results. | `Evaluator.run_once()`, `run()`. |
| `evaluation/metrics.py` | Basic evaluation metrics. | `mean_reward()`, `budget_violation_count()`. |
| `evaluation/reports.py` | Human-readable and aggregate evaluation summaries. | `format_result()`, `summarize()`. |
| `evaluation/runner.py` | Builds an evaluation environment, optionally restores a checkpoint, runs tests, writes output, and plots history. | `EvaluationRunner.run()`. |
| `evaluation/test_runner.py` | Test-runner support/placeholder for evaluation integration. | Extension point. |

### `src/nash_drl/visualization/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `visualization/__init__.py` | Visualization package API. | Public exports. |
| `visualization/network.py` | Converts CSR graph to NetworkX representation. | `to_networkx()`. |
| `visualization/routes.py` | Route-length analysis helper. | `route_lengths()`. |
| `visualization/training.py` | Generates learning curves and rolling means. | `plot_training_history()`, `load_episode_results()`. |
| `visualization/salabim_replay.py` | Optional salabim trajectory replay/animation from SUMO traces. | `SalabimReplay.from_trace()`, `show()`. |

### `src/nash_drl/utils/`

| File | Purpose | Important classes/functions |
|---|---|---|
| `utils/__init__.py` | Utility package API. | Public imports. |
| `utils/device.py` | Resolve CPU/CUDA device selection. | `resolve_device()`. |
| `utils/logging.py` | Standard project logger factory. | `get_logger()`. |
| `utils/seeding.py` | Reproducibility helper across random generators. | `seed_everything()`. |
| `utils/serialization.py` | JSON persistence. | `save_json()`. |

### `tests/`

| File/group | Purpose |
|---|---|
| `tests/fixtures/small_graph.yaml` | Deterministic small graph fixture for tests. |
| `tests/unit/test_csr_map.py` | CSR representation correctness. |
| `tests/unit/test_deep_sets.py` | Deep Sets shape and permutation invariance. |
| `tests/unit/test_actor.py` | Actor output/positivity contracts. |
| `tests/unit/test_critic.py` | Critic output contract. |
| `tests/unit/test_lq_advantage.py` | LQ advantage mathematical behavior. |
| `tests/unit/test_models.py` | Cross-network model checks. |
| `tests/unit/test_features.py` | Feature extraction. |
| `tests/unit/test_feature_normalization.py` | Feature scaling behavior. |
| `tests/unit/test_mapper.py` | Deterministic route mapping. |
| `tests/unit/test_mock_dataset.py` | Reproducibility and feasibility-aware data generation. |
| `tests/unit/test_charging.py` | Charging price model. |
| `tests/unit/test_congestion.py` | Congestion equation. |
| `tests/unit/test_reward.py` | Reward and budget-violation logic. |
| `tests/unit/test_sumo_scenario.py` | SUMO network/config/route generation. |
| `tests/unit/test_training_losses.py` | TD/LQ training losses. |
| `tests/unit/test_training_horizon.py` | Maximum-trip-count episode horizon. |
| `tests/integration/test_environment_step.py` | Environment step integration. |
| `tests/integration/test_actor_environment.py` | Actor → mapper → environment integration. |
| `tests/integration/test_training_step.py` | One-step training integration. |
| `tests/integration/test_training_runner.py` | End-to-end runner orchestration. |

---

# Configuration reference

The project uses YAML configuration. CLI arguments override selected values at runtime; the Python runners then pass the relevant sections into typed configuration dataclasses.

**Range convention used below:**

- `> 0` means strictly positive.
- `>= 0` means zero or greater.
- `0..1` means normalized probability/fraction.
- `integer >= 1` means positive integer.
- `enum` lists the supported symbolic values.
- Where the code has an explicit validation rule, that rule is stated. Otherwise the listed range is the mathematically meaningful/project-supported range rather than a hard YAML parser constraint.

## `configs/default.yaml`

### `project`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `project.name` | `nash-drl` | Project identifier used for configuration/metadata. | Non-empty string. |
| `project.seed` | `42` | Global reproducibility seed. | Integer; any deterministic integer supported by NumPy/PyTorch. |

### `simulation`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `simulation.mode` | `mock` | Selects whether the simulation generates a mock dataset or loads real data. | `mock`  `real`. |
| `simulation.visualization` | `false` | Selects SUMO-GUI vs headless SUMO for simulation. | Boolean. |
| `simulation.generate_analytic_report` | `true` | Writes consolidated CSV analytical report. | Boolean. |
| `simulation.output_dir` | `outputs/demo` | Simulation artifact root. | Valid path. |
| `simulation.dataset_path` | `datasets/generated/mock_seed_42/dataset.json` | Existing dataset path used in real-data mode; in mock mode the generator path is used. | Existing JSON file for real mode. |
| `simulation.backend` | `traci` | Python simulation control backend. | Currently `traci`. |

### `simulation.sumo`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `binary` | `auto` | SUMO executable selection. | `auto`, `sumo`, `sumo-gui`, or absolute path. |
| `step_length_s` | `1.0` | SUMO simulation time increment. | `> 0`. |
| `end_time_s` | `300` | Maximum SUMO simulation time. | `> 0`. |
| `teleport_time_s` | `120` | SUMO time-to-teleport for stuck vehicles. | `>= 0`. |
| `seed` | `42` | SUMO-specific random seed. | Integer. |
| `default_edge_speed_kmh` | `50.0` | Default synthetic SUMO edge speed. | `> 0`. |
| `vehicle_departure_gap_s` | `2.0` | Delay between generated vehicle departures. | `>= 0`. |
| `allow_turnarounds` | `true` | Whether generated SUMO connections may include turnaround movements required by some multi-trip routes. | Boolean. |

### `simulation.salabim`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `enabled` | `true` | Enables optional salabim replay/animation layer. | Boolean. |
| `replay_speed` | `1.0` | Playback speed multiplier. | `> 0`. |

### `mock_data`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `seed` | `42` | Seed dedicated to mock-world generation. | Integer. |
| `num_vehicles` | `12` | Number `N` of EAVs. | `>= 1`. |
| `min_trips_per_vehicle` | `1` | Minimum trip-set length. | `>= 1`. |
| `max_trips_per_vehicle` | `3` | Maximum trip-set length; determines a possible episode horizon. | `>= min_trips_per_vehicle`. |
| `graph_nodes` | `12` | Number `V` of graph nodes. | Code requires `>= 3`. |
| `graph_extra_edges` | `10` | Number of extra directed edges beyond the guaranteed bidirectional ring. | `>= 0`. |
| `road_min_length_km` | `1.0` | Lower bound for stochastic road length. | `> 0`. |
| `road_max_length_km` | `8.0` | Upper bound for road length. | `>= road_min_length_km`. |
| `road_length_std_km` | `1.5` | Standard deviation of normal road-length sampling before clipping. | `>= 0`. |
| `capacity_min_vph` | `20` | Lower bound on edge capacity. | Positive integer. |
| `capacity_max_vph` | `80` | Upper bound on edge capacity. | `>= capacity_min_vph`. |
| `capacity_std_vph` | `15.0` | Normal-distribution standard deviation for capacity. | `>= 0`. |
| `speed_min_kmh` | `30.0` | Lower free-flow vehicle speed. | `> 0`. |
| `speed_max_kmh` | `70.0` | Upper free-flow vehicle speed. | `>= speed_min_kmh`. |
| `speed_std_kmh` | `10.0` | Speed sampling standard deviation. | `>= 0`. |
| `budget_min` | `75.0` | Lower generated budget bound. | `> 0`. |
| `budget_max` | `300.0` | Upper generated budget bound. | `>= budget_min`. |
| `budget_std` | `45.0` | Budget normal-distribution standard deviation. | `>= 0`. |
| `budget_feasibility_multiplier` | `1.20` | Multiplies an optimistic shortest-path cost to center the budget distribution above a feasible reference cost. | Code requires `>= 1.0`. |
| `budget_resample_limit` | `100` | Maximum feasibility-aware normal resampling attempts. | `>= 1`. |
| `trip_count_std` | `0.9` | Standard deviation of stochastic trip-count sampling. | `>= 0`. |
| `energy_rate_kwh_per_km` | `1.0` | Vehicle energy consumption coefficient `η`. | `> 0`. |
| `charging_overhead` | `6.0` | Charging-station shared overhead `O`. | `> 0`. |
| `charging_fixed_cost` | `1.0` | Charging-station fixed unit cost `C`. | `>= 0`. |
| `charging_floor_price` | `0.0` | Minimum charging price `p_e^min`. | `>= 0`. |
| `congestion_alpha` | `0.15` | Congestion coefficient `α`. | `>= 0`. |
| `congestion_beta` | `4.0` | Congestion exponent `β`. | `>= 1` according to the model. |
| `vehicle_departure_gap_s` | `2.0` | Synthetic SUMO departure spacing. | `>= 0`. |

### `network`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `hidden_dim` | `32` | Width of fully connected hidden layers. | Positive integer. |
| `deep_set_dim` | `64` | Dimension of the aggregated crowd vector. | Positive integer. |
| `actor_hidden_layers` | `4` | Number of Actor hidden FC layers. | Positive integer; V12 specifies 4. |
| `critic_hidden_layers` | `4` | Number of Critic hidden FC layers. | Positive integer; V12 specifies 4. |

### `environment`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `energy_rate_kwh_per_km` | `1.0` | `η` used by cost model. | `> 0`. |
| `charging_overhead` | `6.0` | `O` in charging price. | `> 0`. |
| `charging_fixed_cost` | `1.0` | `C` in charging price. | `>= 0`. |
| `charging_floor_price` | `0.0` | `p_e^min`. | `>= 0`. |
| `congestion_alpha` | `0.15` | `α`. | `>= 0`. |
| `congestion_beta` | `4.0` | `β`. | `>= 1`. |
| `max_steps` | `300` | Environment maximum-step safety limit. | `>= 1`. |
| `use_model_travel_time` | `true` | If true, reward uses the mathematical congestion/travel-time model; if false, SUMO observed trip travel time is used. | Boolean. |

### `reward`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `travel_time_weight` | `1.0` | `w_T` in regular reward. | `>= 0`. |
| `charging_cost_weight` | `1.0` | `w_C` in regular reward. | `>= 0`. |
| `budget_penalty` | `100.0` | `P`; reward assigned to a vehicle exceeding its budget. | `>= 0`; experiment-dependent. |

### `training`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `mode` | `mock` | Dataset source for training. | `mock` or `real`. |
| `episodes` | `20` | Number of training episodes. | Integer `>= 1`. |
| `gamma` | `0.99` | Discount factor `γ`. | Conventionally `0..1`; use `<1` for discounted learning. |
| `actor_lr` | `0.0003` | Actor Adam learning rate. | `> 0`. |
| `critic_lr` | `0.0003` | Critic Adam learning rate. | `> 0`. |
| `exploration_sigma` | `0.1` | Initial Gaussian action-noise standard deviation. | `>= 0`. |
| `exploration_sigma_final` | `0.02` | Final exploration-noise standard deviation. | `>= 0`. |
| `exploration_decay_episodes` | `1000` | Number of episodes over which noise linearly decays. | `>= 0`. |
| `reward_scale` | `0.01` | Multiplicative scale applied to rewards when constructing training targets; reporting retains raw reward. | Finite positive value; `0` is generally not useful. |
| `max_grad_norm` | `5.0` | Gradient clipping threshold. | `> 0`. |
| `replay_enabled` | `true` | Enables replay-buffer sampling. | Boolean. |
| `replay_capacity` | `10000` | Maximum number of stored transitions. | Integer `>= 1`. |
| `replay_batch_size` | `32` | Number of transitions per optimizer update. | Integer `>= 1`. |
| `replay_warmup` | `32` | Minimum replay size before updates begin. | Integer `>= 1`. |
| `updates_per_step` | `1` | Number of replay updates performed after each environment step. | Integer `>= 1`. |
| `target_update_interval` | `100` | Number of optimizer updates between Target Critic hard copies. | Integer `>= 1`. |
| `device` | `cpu` | PyTorch execution device. | `cpu`, valid CUDA device string, or project-supported device value. |
| `seed` | `42` | Training/replay/random-noise seed. | Integer. |
| `output_dir` | `outputs/training` | Training artifact root. | Valid path. |
| `visualization` | `false` | Enables SUMO-GUI during training. | Boolean. |
| `save_checkpoints` | `true` | Enables checkpoint writing. | Boolean. |
| `checkpoint_interval` | `5` | Episodes between checkpoints. | Integer `>= 1`. |

### `evaluation`

| Parameter | Default | Meaning / definition | Range / constraint |
|---|---:|---|---|
| `evaluation.mode` | `mock` | Dataset source for testing. | `mock` or `real`. |
| `evaluation.episodes` | `5` | Number of evaluation episodes. | Integer `>= 1`. |
| `evaluation.output_dir` | `outputs/evaluation` | Evaluation output root. | Valid path. |
| `evaluation.visualization` | `false` | Evaluation visualization selection. | Boolean. |

---

## `configs/environment.yaml`

This file is intended for environment-focused work. It uses the same mathematical model parameters but omits the network and most training settings.

| Parameter | Value | Meaning | Range / constraint |
|---|---:|---|---|
| `environment.energy_rate_kwh_per_km` | `1.0` | `η`, vehicle energy consumption. | `> 0` |
| `environment.charging_overhead` | `6.0` | `O`, station overhead. | `> 0` |
| `environment.charging_fixed_cost` | `1.0` | `C`, fixed price component. | `>= 0` |
| `environment.charging_floor_price` | `0.0` | Minimum charging unit price. | `>= 0` |
| `environment.congestion_alpha` | `0.15` | Congestion coefficient `α`. | `>= 0` |
| `environment.congestion_beta` | `4.0` | Congestion exponent `β`. | `>= 1` |
| `environment.max_steps` | `300` | Environment safety-step limit. | `>= 1` |
| `simulation.backend` | `traci` | TraCI simulation backend. | `traci` |
| `simulation.sumo.binary` | `auto` | Automatic SUMO binary resolution. | `auto`, `sumo`, `sumo-gui`, absolute path |
| `simulation.sumo.step_length_s` | `1.0` | SUMO timestep. | `> 0` |
| `simulation.sumo.end_time_s` | `300` | Maximum SUMO time. | `> 0` |
| `simulation.sumo.teleport_time_s` | `120` | SUMO teleport timeout. | `>= 0` |

---

## `configs/network.yaml`

| Parameter | Value | Meaning | Range / constraint |
|---|---:|---|---|
| `network.hidden_dim` | `32` | Fully connected hidden-layer width. V12 specifies 32 nodes per hidden layer. | Positive integer |
| `network.deep_set_dim` | `64` | Crowd embedding output dimension. | Positive integer |
| `network.deep_set_hidden_layers` | `2` | Hidden layers inside the shared Deep Sets embedding `φ`. | Integer `>= 1` |
| `network.actor_hidden_layers` | `4` | Actor trunk hidden layers. | `4` in V12; positive integer implementation range |
| `network.critic_hidden_layers` | `4` | Critic trunk hidden layers. | `4` in V12; positive integer implementation range |
| `network.positivity_epsilon` | `1e-6` | Numerical epsilon added by the P11/P22 positive transform. | `> 0` |

---

## `configs/training.yaml`

| Parameter | Value | Meaning | Range / constraint |
|---|---:|---|---|
| `training.mode` | `mock` | Dataset source. | `mock` / `real` |
| `training.episodes` | `20` | Training episode count. | `>= 1` |
| `training.gamma` | `0.99` | Discount factor. | `0..1` conventionally |
| `training.actor_lr` | `0.0003` | Actor learning rate. | `> 0` |
| `training.critic_lr` | `0.0003` | Critic learning rate. | `> 0` |
| `training.exploration_sigma` | `0.10` | Initial action noise. | `>= 0` |
| `training.exploration_sigma_final` | `0.02` | Final action noise. | `>= 0` |
| `training.exploration_decay_episodes` | `1000` | Linear noise-decay horizon. | `>= 0` |
| `training.reward_scale` | `1.0` | Training reward scaling in this focused config. | Positive finite value |
| `training.max_grad_norm` | `5.0` | Gradient clipping. | `> 0` |
| `training.replay_enabled` | `true` | Replay buffer toggle. | Boolean |
| `training.replay_capacity` | `10000` | Replay capacity. | `>= 1` |
| `training.replay_batch_size` | `32` | Sample size. | `>= 1` |
| `training.replay_warmup` | `32` | Minimum transitions before learning. | `>= 1` |
| `training.updates_per_step` | `1` | Replay updates per environment step. | `>= 1` |
| `training.target_update_interval` | `100` | Target Critic copy interval. | `>= 1` |
| `training.device` | `cpu` | Torch device. | Valid PyTorch device |
| `training.seed` | `42` | Training seed. | Integer |
| `training.output_dir` | `outputs/training` | Output root. | Valid path |
| `training.visualization` | `false` | SUMO GUI during training. | Boolean |
| `training.save_checkpoints` | `true` | Checkpoint toggle. | Boolean |
| `training.checkpoint_interval` | `5` | Checkpoint episode interval. | `>= 1` |

For real-data training, the configuration can additionally specify `training.dataset_path` pointing to an existing dataset JSON file.

---

# Experiment configuration reference

The three experiment files are standalone runnable configurations. Their values intentionally differ because each benchmark has a different scale and learning objective.

## `configs/experiments/small.yaml`

### Scenario definition

| Parameter | Value | Purpose |
|---|---:|---|
| Seed | `42` | Fully reproducible small benchmark. |
| Vehicles | `6` | Small multi-agent population. |
| Graph nodes | `9` | Small road graph. |
| Extra edges | `5` | Provides route alternatives beyond the bidirectional ring. |
| Trips/vehicle | `1..2` | Short trip-sets; horizon is at most 2. |
| Road length | `1..4 km` | Compact synthetic network. |
| Capacity | `30..80 veh/h` | Moderate edge capacities. |
| Speed | `35..60 km/h` | Moderate heterogeneous free-flow speeds. |
| Budget | `80..240` with feasibility multiplier `1.20` | Reference-feasible budget generation. |
| Congestion | `α=0.15`, `β=4` | Same mathematical congestion model as V12. |

### Learning definition

| Parameter | Value | Purpose |
|---|---:|---|
| Episodes | `3` | Fast smoke-test default. |
| Gamma | `0.99` | Standard discounting. |
| Actor/Critic LR | `3e-4 / 3e-4` | Baseline optimizer. |
| Exploration | `0.10 → 0.02` | Noise decay over 100 episodes. |
| Replay | `1000` capacity, `16` batch/warmup | Lightweight replay. |
| Target interval | `20` updates | Frequent target synchronization for small runs. |
| Reward penalty | `500` | Stronger hard-budget signal than the base config. |

**Purpose:** use this benchmark for package validation, route/SUMO debugging, tensor-contract checks, and short end-to-end learning smoke tests. It is too small to be treated as a definitive research benchmark.

---

## `configs/experiments/medium.yaml`

The medium benchmark is intended to answer a practical diagnostic question: **does the hard-budget violation rate decrease as Actor/Critic training progresses?**

| Parameter | Value | Meaning |
|---|---:|---|
| Seed | `2027` | Reproducible medium benchmark. |
| Vehicles | `20` | Enough agents to create interaction effects. |
| Graph nodes | `14` | Moderate graph. |
| Extra edges | `14` | Provides route alternatives. |
| Trips/vehicle | `1..3` | Variable trip-set lengths; horizon determined from the dataset. |
| Road length | `1..6 km` | Moderate route lengths. |
| Capacity | `20..100 veh/h` | More visible congestion/resource interaction. |
| Speed | `35..70 km/h` | Heterogeneous vehicles. |
| Budget | `15..250`, std `4` | Relatively tight budget distribution. |
| Budget feasibility multiplier | `1.0` | Centers budgets close to the optimistic feasible reference cost. |
| Budget penalty | `500` | Makes violations visible in reward comparisons while retaining the piecewise reward definition. |
| Gamma | `0.97` | Shorter effective horizon discounting. |
| Actor LR | `1e-4` | Conservative Actor updates. |
| Critic LR | `3e-4` | Faster value-function adaptation. |
| Exploration | `0.25 → 0.03` | Strong initial exploration, gradual decay. |
| Replay | capacity `10000`, batch `32`, warmup `32` | Stable sample reuse. |
| Updates/step | `2` | More learning per simulation step. |
| Target interval | `50` | More frequent target refresh. |
| Episodes | `300` | Long enough for a learning diagnostic. |

**Purpose:** training-development benchmark, especially for checking that budget violations are initially non-zero and have an opportunity to decrease over time.

Important interpretation: convergence to zero violations is **not guaranteed** by configuration alone. The correct result is a downward trend or lower stable plateau supported by repeatable runs and statistical analysis.

---

## `configs/experiments/large.yaml`

| Parameter | Value | Meaning |
|---|---:|---|
| Seed | `123` | Reproducible larger benchmark. |
| Vehicles | `50` | Larger multi-agent population. |
| Graph nodes | `40` | Larger synthetic network. |
| Extra edges | `55` | Richer route alternatives. |
| Trips/vehicle | `1..4` | Up to four trip legs; episode horizon is derived from the dataset. |
| Road length | `0.5..10 km` | Broad road-length distribution. |
| Capacity | `15..150 veh/h` | Wide capacity range. |
| Speed | `25..80 km/h` | Heterogeneous EAVs. |
| Budget | `50..500`, std `70` | Broad budget distribution; feasibility-aware generation with multiplier `1.20`. |
| Gamma | `0.99` | High discount factor. |
| Actor/Critic LR | `3e-4 / 3e-4` | Baseline learning rates. |
| Exploration | `0.10 → 0.02` | Slow decay over 1000 episodes. |
| Replay | capacity `10000`, batch/warmup `32` | Standard replay configuration. |
| Target interval | `100` | Target hard-copy interval. |
| Budget penalty | `1000` | Large hard-budget penalty for larger-scale diagnostic. |
| SUMO max time | `600 s` | More simulation time per trip-level SUMO run. |
| Episodes | `20` in file | Override from CLI for extended experiments, e.g. `--episodes 1000`. |

**Purpose:** larger-scale stress and research experiments after the implementation has been validated on the small and medium configurations.

---

# 2. Match important Network Designs with source code

The network design in **NashDRL-Model-V12 Section 4** contains three neural modules: Actor, Critic, and Target Critic, with Deep Sets used to enforce permutation invariance. Section 4.1 explicitly defines the two input streams as `[N,N-1,F]` and `[N,F+E]`; the Actor produces `[5,N,E]`; the Critic produces per-agent values; and the Target Critic is a frozen copy updated by hard synchronization. 

## 2.1 State-to-network input mapping

The model defines:

```text
Per-agent features       X_agents : [N,F]
Edge-flow/global state             [E]

Invariant rival stream             [N,N-1,F]
Non-invariant stream                [N,F+E]
```

Implementation:

```text
GlobalState
    │
    ├── agent_features [N,F]
    ├── edge_flow      [E]
    ├── current_nodes  [N]
    ├── next_destinations [N]
    └── final_destinations [N]
             │
             ▼
StateFeatureExtractor.extract()
             │
       ┌─────┴─────────┐
       ▼               ▼
[N,N-1,F]          [N,F+E]
Invariant          Non-invariant
```

Source mapping:

| V12 component | Source code |
|---|---|
| `[N,F]` agent state | `data/state.py::GlobalState.agent_features`; `features/agent_features.py::encode_agent_features()` |
| `[E]` edge flow | `data/state.py::GlobalState.edge_flow`; `features/edge_features.py::edge_flow_features()` |
| `[N,N-1,F]` rival stream | `features/extractor.py::StateFeatureExtractor.extract()` and `features/deep_set_input.py` |
| `[N,F+E]` ego/global stream | `features/extractor.py::StateFeatureExtractor.extract()` |

The six semantic agent features are the current stop, next destination, final destination, remaining trips, remaining budget, and free-flow speed; global edge flows are concatenated to the focal-agent stream. This follows the source specification. 

---

## Deep Sets implementation

The paper requires each rival feature vector to pass through the **same shared embedding function `φ`**, followed by element-wise summation to create a crowd representation independent of rival ordering.

```text
X^-i [N,N-1,F]
       │
       ▼
 shared φ
       │
       ▼
[N,N-1,D]
       │
     SUM
       │
       ▼
 [N,D]
```

Source:

```text
src/nash_drl/models/deep_sets.py
    DeepSetEncoder.forward()
```

The older reference implementation supplied with the project used a mean-moment summary. NashDRL v12 was implemented according to the newer specification: shared embedding + summation. This distinction is intentional.

---

## Actor Network mapping

V12 Section 4.2 specifies two streams, concatenation, a four-hidden-layer fully connected trunk with 32 units/SiLU, and five outputs for every vehicle/edge pair.

```text
                  ┌─────────────────────────┐
Invariant         │ DeepSetEncoder          │
[N,N-1,F] ───────►│ shared φ + sum          │
                  └───────────┬─────────────┘
                              │ [N,D]
                              │
Non-invariant                 │
[N,F+E] ──────────────────────┤
                              ▼
                         concatenate
                              │
                   ┌──────────▼──────────┐
                   │ FC(32) + SiLU       │
                   │ FC(32) + SiLU       │
                   │ FC(32) + SiLU       │
                   │ FC(32) + SiLU       │
                   └──────────┬──────────┘
                              ▼
                         [5,N,E]
                              │
             ┌────────┬───────┼────────┬────────┐
             ▼        ▼       ▼        ▼        ▼
             μ       P11     P12      P22       Ψ
```

Source mapping:

| V12 Actor component | Implementation |
|---|---|
| Shared rival encoder | `models/deep_sets.py::DeepSetEncoder` |
| Crowd aggregation | `DeepSetEncoder.forward()` using `sum(dim=-2)` |
| Ego/global stream | `models/actor.py::ActorNetwork.non_invariant` |
| Concatenation | `ActorNetwork.forward()` |
| Four FC + SiLU layers | `models/common/mlp.py::MLP`, instantiated by `ActorNetwork` |
| Raw five-channel output | `ActorNetwork.forward()` reshaping to `[... ,5,E]` per focal-agent dimension |
| `μ` | `ActorOutput.mu` |
| `P11` | `ActorOutput.p11` |
| `P12` | `ActorOutput.p12` |
| `P22` | `ActorOutput.p22` |
| `Ψ` | `ActorOutput.psi` |
| Strict positivity for `P11/P22` | `models/common/activations.py::strictly_positive()` |

The source document states that P11 and P22 are transformed to preserve positive-definiteness/concavity requirements. 

---

## Critic Network mapping

The Critic mirrors the permutation-invariant design but outputs a single value per agent.

```text
Invariant [N,N-1,F]
      │
      ▼
   Deep Sets
      │
      ▼
    [N,D]

Non-invariant [N,F+E]
      │
   flatten
      │
      ├───────────────┐
      │               │
      └──── concat ───┘
              │
      FC(32)+SiLU × 4
              │
             [N]
```

Source mapping:

| V12 Critic component | Implementation |
|---|---|
| Rival/permutation stream | `models/critic.py::CriticNetwork.deep_sets` |
| Non-invariant stream | `models/critic.py::CriticNetwork.forward()` |
| Flatten/concatenation | `CriticNetwork.forward()` |
| Four-layer value trunk | `models/common/mlp.py::MLP` instantiated in `CriticNetwork` |
| Per-agent `V_i(x)` | `CriticNetwork.forward()` output `[N]` or `[B,N]` |

The source states that the Critic predicts a scalar baseline value for each agent and that the same four 32-unit SiLU layers are used. 

---

## Target Critic mapping

V12 requires the Target Critic to be an exact architectural copy of the Critic, frozen during normal optimization, and synchronized by hard copy.

Source:

```text
src/nash_drl/models/target_critic.py

TargetCriticNetwork.__init__()
TargetCriticNetwork.forward()
TargetCriticNetwork.hard_update_from()
TargetCriticNetwork._freeze()
```

Relationship:

```text
CriticNetwork
      │
   deepcopy
      ▼
TargetCriticNetwork
      │
   frozen
      │
      └── hard_update_from(critic)
```

Training integration occurs in:

```text
src/nash_drl/training/trainer.py
    NashDRLTrainer._optimize_batch()
```

where the Target Critic is evaluated under `torch.no_grad()` and hard synchronized after the configured update interval.

---

## LQ Advantage mapping

The source defines:

\[
z_i=u_i-\mu_i
\]

and the local advantage structure:

\[
A_i(x;u)=
-\|z_i\|^2_{P_{11,i}}
-\sum_{j\ne i}\langle z_i,z_j\rangle_{P_{12,i}}
-\sum_{j\ne i}\|z_j\|^2_{P_{22,i}}
+\sum_{j\ne i}z_j^T\Psi_i(x).
\]

Source:

```text
src/nash_drl/models/lq_advantage.py
    LQAdvantage.forward()
```

The implementation computes all agents simultaneously for `[N,E]` or `[B,N,E]` actions.

---

# 2.1 Match problem formulations with source code and exact functions

NashDRL-Model-V12 Section 2 defines a directed graph, heterogeneous vehicles, flow-dependent charging price, congestion-dependent travel time, vehicle reward, and system reward.

## 2.1.1 Road network model

Mathematical formulation:

\[
G=(V,E)
\]

Each edge has:

- length `l_e`
- capacity `c_e`

Implementation:

```text
src/nash_drl/domain/graph.py
    Edge
    DirectedGraph
```

Runtime/CSR representation:

```text
src/nash_drl/data/csr_map.py
    CSRMap.from_graph()
```

The implementation stores:

```text
row_ptr [V+1]
col_idx [E]
edge_ids [E]
edge_len [E]
edge_cap [E]
edge_flow [E]
edge_from [E]
```

The CSR structure follows the document's Section 3 representation.

---

## 2.1.2 Vehicle model

For each vehicle:

- origin `O_i`
- destination `D_i`
- free-flow speed `v_i`
- budget `B_i`

Implementation:

```text
src/nash_drl/domain/vehicle.py
    Vehicle
```

The multi-trip extension is represented by:

```text
src/nash_drl/domain/trip.py
    Trip
```

and a vehicle stores:

```text
Vehicle.trips
Vehicle.current_trip_index
Vehicle.current_node
Vehicle.remaining_budget
```

This supports variable trip-set lengths across vehicles.

---

## 2.1.3 Dynamic traffic flow as a state variable

The current implementation treats road topology and traffic flow differently. The CSR map remains available to routing/environment code, but the static graph topology is not passed to the Actor or Critic. The dynamic edge-flow vector is part of the neural-network state.

For the SUMO training environment, one persistent TraCI session is maintained for the entire episode. At the beginning of `t=0`, SUMO is queried for an `[E]` edge-flow snapshot. After the selected trip legs are executed, SUMO is queried again and the resulting snapshot becomes `GlobalState.edge_flow` for `t+1`.

Relevant implementation:

```text
src/nash_drl/environment/sumo.py
    SumoTrafficSession.start()
    SumoTrafficSession.snapshot_edge_flow()
    SumoTrafficSession.execute_routes()

src/nash_drl/environment/sumo_training.py
    NashSUMOTrainingEnvironment.reset()
    NashSUMOTrainingEnvironment.step()
```

This prevents the previous behavior where each RL step restarted SUMO and consequently began with a zero-flow map. The detailed timing contract is documented in `docs/TRAFFIC_FLOW_STATE.md`.

---

## 2.1.3 Edge flow

The mathematical model defines:

\[
f_e=\sum_i x_{i,e}.
\]

In the implementation, edge flow is maintained as the dynamic state vector:

```text
GlobalState.edge_flow
CSRMap.edge_flow
```

and updated by the analytical/SUMO environment according to selected paths.

Relevant functions:

```text
environment/env.py
    NashEnvironment.step()

environment/sumo_training.py
    NashSUMOTrainingEnvironment.step()
```

The current SUMO training environment obtains microscopic telemetry and keeps model-level edge flow/cost computation separate from SUMO diagnostic metrics.

---

## 2.1.4 Charging price model

The model defines:

\[
p_e(f_e)=\max\left(p_e^{min}, \frac{O}{f_e}+C\right)
\]

for an active edge.

Implementation:

```text
src/nash_drl/domain/charging.py
    ChargingStationModel.unit_price()
```

and tensor form:

```text
src/nash_drl/environment/energy_cost.py
    charging_price()
```

---

## 2.1.5 Vehicle charging/energy cost

The model defines:

\[
C_i=\eta\sum_e x_{i,e}l_e p_e(f_e).
\]

Implementation:

```text
src/nash_drl/environment/energy_cost.py
    edge_energy_cost()
```

The SUMO training environment calls this function when evaluating the selected edge path for an active vehicle.

---

## 2.1.6 Congestion model

The model defines the edge travel-time multiplier:

\[
1+\alpha\left(\frac{f_e}{c_e}\right)^\beta.
\]

Implementation:

```text
src/nash_drl/environment/congestion.py
    congestion_multiplier()
```

---

## 2.1.7 Travel-time model

The model defines:

\[
\tau_{i,e}(f_e)=
\frac{l_e}{v_i}\left[1+\alpha\left(\frac{f_e}{c_e}\right)^\beta\right].
\]

Implementation:

```text
src/nash_drl/environment/travel_time.py
    edge_travel_time()
```

For the SUMO training environment, `environment.use_model_travel_time=true` makes the mathematical equation the learning-reward travel-time source while measured SUMO travel time remains available as a diagnostic metric.

---

## 2.1.8 Vehicle reward

The model defines:

\[
R_i=
\begin{cases}
-w_TT_i-w_CC_i, & C_i\le B_i,\\
-P, & C_i>B_i.
\end{cases}
\]

Implementation:

```text
src/nash_drl/environment/reward.py
    RewardModel.compute()
```

The function returns:

```text
vehicle_rewards [N]
total_reward    scalar
ttravel_times   [N]
charging_costs  [N]
budget_violations [N]
```

---

## 2.1.9 System total reward

The model defines:

\[
TR=\sum_iR_i.
\]

Implementation:

```text
RewardModel.compute()
    total_reward = rewards.sum()
```

Training reporting then accumulates per-step system reward into the episode result in:

```text
src/nash_drl/training/trainer.py
    NashDRLTrainer.train_episode()
```

---

# 3. Illustrative execution sequences

## Overall execution architecture

The complete application has three related paths:

```text
A. Simulation only

Dataset → SUMO scenario → TraCI → trace/report → visualization


B. Learning

Dataset → Episode reset → State → Actor → Action → Route mapper
         → SUMO trip-leg execution → Reward/Next State
         → Critic/Target Critic → TD/LQ loss → parameter updates
         → next step → next episode


C. Testing

Dataset + checkpoint → deterministic policy → episode execution
                    → evaluation CSV → evaluation plots
```

---

# 3.1 Learning and testing sequences

## Learning sequence

The high-level call chain for:

```powershell
python scripts/train.py --config configs/experiments/medium.yaml --mode mock --episodes 300
```

is:

```text
scripts/train.py
    ↓
nash_drl.cli.train()
    ↓
run_training()
    ↓
TrainingRunner.run()
    ↓
_prepare_dataset()
    ↓
MockDatasetGenerator.generate()/save()
    ↓
load_problem()
    ↓
NashSUMOTrainingEnvironment(...)
    ↓
ActorNetwork / CriticNetwork / TargetCriticNetwork
    ↓
NashDRLTrainer
    ↓
for episode in range(training.episodes)
    ↓
trainer.train_episode(episode)
```

### Episode start

`NashDRLTrainer.train_episode()` calls:

```text
NashSUMOTrainingEnvironment.reset(episode_index=episode)
```

The reset operation restores the vehicles to the beginning of their fixed trip sets. The trip sets themselves are **not regenerated between episodes** when using the same generated dataset.

The number of steps is derived from the loaded dataset:

```text
max_steps = max(len(vehicle.trips))
```

Thus the learning horizon is data-dependent rather than independently specified.

### Step `t`

For each step:

```text
1. GlobalState from environment
        ↓
2. StateFeatureExtractor.extract()
        ↓
3. ActorNetwork.forward()
        ↓
4. ActorOutput [μ,P11,P12,P22,Ψ]
        ↓
5. AnalyticalNashPolicy.select()
        ↓
6. exploration noise during training
        ↓
7. Action [N,E]
        ↓
8. ActionToPathMapper.map()
        ↓
9. SUMO/TraCI trip-leg execution
        ↓
10. mathematical charging/travel-time/reward model
        ↓
11. next GlobalState
        ↓
12. ReplayBuffer.add()
        ↓
13. sample replay batch when warm-up is satisfied
        ↓
14. Critic + Target Critic evaluation
        ↓
15. LQAdvantage.forward()
        ↓
16. compute_training_losses()
        ↓
17. Critic optimization
        ↓
18. Actor optimization
        ↓
19. periodic TargetCritic.hard_update_from()
        ↓
20. save step metrics
```

### Actor/Critic loss dependency

The current implementation follows the V12 training decomposition:

```text
Q(x,u) = V(x) + A(x,u)

TD target = r + γ Vslow(x')

Critic path:
    V(x) + detach(A(x,u))

Actor path:
    detach(V(x)) + A(x,u)
```

The executed action is detached from the Actor graph before learning. This keeps `u` fixed while `mu` is optimized through `z=u-mu`.

Relevant functions:

```text
src/nash_drl/training/trainer.py
    _optimize_batch()
    _run_updates()

src/nash_drl/training/losses.py
    compute_td_target()
    compute_training_losses()

src/nash_drl/models/lq_advantage.py
    LQAdvantage.forward()
```

### End of episode

After the maximum dataset-derived number of trip steps:

```text
step_rows
vehicle_rows
edge_rows
episode_row
```

are returned from `train_episode()`.

`TrainingRunner.run()` aggregates all episodes and writes:

```text
step_results.csv
episode_results.csv
vehicle_results.csv
edge_results.csv
training_metadata.json
```

`training/experiment.py::run_training()` then calls `plot_training_history()` to create learning curves.

---

## Testing/evaluation sequence

A typical command is:

```powershell
python scripts/evaluate.py --config configs/experiments/medium.yaml --mode mock --episodes 10 --checkpoint <checkpoint.pt>
```

Execution:

```text
scripts/evaluate.py
    ↓
nash_drl.cli.evaluate()
    ↓
EvaluationRunner.run()
    ↓
load/generate dataset
    ↓
construct Actor/Critic/Target Critic
    ↓
load checkpoint (if provided)
    ↓
Evaluator.run()
    ↓
execute evaluation episodes
    ↓
calculate metrics
    ↓
write CSV/metadata/plots
```

Evaluation does not use exploration as the training policy does. The objective is to measure the learned policy on the configured fixed dataset.

---

# 3.2 Creating mock or loading real world data for SUMO learning/testing

## Mock-data workflow

Mock mode uses a two-stage design:

```text
YAML mock_data configuration
       ↓
MockDataConfig
       ↓
MockDatasetGenerator
       ↓
reproducible stochastic world
       ↓
dataset.json
       ↓
load_problem()
       ↓
ProblemDefinition
       ↓
SUMO ScenarioBuilder
       ↓
SUMO net + route + config
```

### Mock dataset structure

The generated JSON contains:

```json
{
  "schema_version": 3,
  "generator": "MockDatasetGenerator",
  "seed": 42,
  "config": { ... },
  "graph": {
    "num_nodes": 12,
    "edges": [
      {
        "id": 0,
        "source": 0,
        "destination": 1,
        "length_km": 3.1,
        "capacity_vph": 40.0
      }
    ]
  },
  "vehicles": [
    {
      "id": 0,
      "free_flow_speed_kmh": 52.0,
      "budget": 135.0,
      "trips": [
        {"origin": 0, "destination": 4},
        {"origin": 4, "destination": 2}
      ]
    }
  ]
}
```

The graph is generated with a bidirectional ring to ensure strong connectivity, plus configurable extra directed edges. Numeric road/capacity/speed/trip-count values are generated from normal distributions and clipped to configured limits. Vehicle budgets are additionally conditioned on an optimistic shortest-path cost to ensure that the mock world has at least one reference-feasible routing assignment when configuration permits it.

### Trip-set rules

A vehicle contains an ordered trip-set:

```text
Trip 0: O0 → D0
Trip 1: D0 → D1
Trip 2: D1 → D2
...
```

In the current generator, the trip endpoints are generated as a sequence of nodes; the training horizon is:

```text
max number of trips among all vehicles
```

Vehicles with fewer trips become inactive in later steps while remaining present in the fixed-size `[N,F]` neural state representation.

### Real-data mode

Real mode does **not** generate a dataset. It expects an existing dataset JSON path:

```yaml
training:
  mode: real
  dataset_path: datasets/scenarios/real_dataset.json
```

or, for simulation-only runs:

```yaml
simulation:
  mode: real
  dataset_path: datasets/scenarios/real_dataset.json
```

The loader converts that JSON into the same:

```text
ProblemDefinition
  ├── DirectedGraph
  └── Vehicle[]
        └── Trip[]
```

representation used by mock data.

### Dataset compatibility requirements

A dataset used for training/testing should preserve this logical schema:

```text
graph.num_nodes

for each edge:
    id
    source
    destination
    length_km
    capacity_vph

for each vehicle:
    id
    free_flow_speed_kmh
    budget
    trips[]
        origin
        destination
```

The loader currently consumes exactly these fields. Additional dataset metadata may be preserved in JSON, but the core fields above must be present.

### SUMO world-data conversion

For simulation, the graph and paths are translated to:

```text
network.nod.xml
network.edg.xml
network.net.xml
routes.rou.xml
simulation.sumocfg
```

`environment/sumo.py::SumoScenarioBuilder` creates these files. `run_sumo()` and `run_sumo_trip()` then start SUMO/SUMO-GUI through TraCI.

The SUMO scenario is a microscopic execution layer. The mathematical NashDRL model remains responsible for its own energy/charging/congestion/reward computations.

---

# 3.3 Currently available mock scenarios

## Small

```text
N = 6 vehicles
V = 9 nodes
extra edges = 5
trips/vehicle = 1..2
```

Designed for:

- installation validation
- SUMO route debugging
- unit/integration development
- tensor-contract checking
- rapid smoke tests

The small scenario is intentionally inexpensive. It is not intended as the principal scientific benchmark.

## Medium

```text
N = 20 vehicles
V = 14 nodes
extra edges = 14
trips/vehicle = 1..3
```

Designed for:

- observing interaction between vehicles
- observing congestion and charging-price coupling
- testing hard budget constraints
- monitoring whether violation rates decrease during training
- validating reward/TD/gradient diagnostics

The budget distribution is deliberately tighter than the large benchmark and is centered around a feasible reference cost so that budget violations can exist for an untrained policy without making zero violations structurally impossible.

## Large

```text
N = 50 vehicles
V = 40 nodes
extra edges = 55
trips/vehicle = 1..4
```

Designed for:

- larger-scale training
- higher agent interaction
- more complex graph routing
- extended SUMO execution
- scalability experiments

Because SUMO is executed at the trip-step level, the computational cost grows with both vehicle population and the number of episode trip steps.

---

# 4. How to run the project

## 4.1 Prerequisites and installation on Windows 11

### Python

The project requires:

```text
Python >= 3.11 and < 3.14
```

Check:

```powershell
python --version
```

Python 3.11 or 3.12 is a conservative choice for the current dependency range.

### SUMO

The project targets:

```text
SUMO 1.27.1
TraCI 1.27.1
sumolib 1.27.1
```

Install SUMO 1.27.1 on Windows and ensure the following are discoverable:

```text
sumo
sumo-gui
netconvert
```

You can either add SUMO's `bin` directory to `PATH` or set:

```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

Verify:

```powershell
sumo --version
sumo-gui --version
netconvert --version
```

The Python project installs matching TraCI/SUMO libraries from `pyproject.toml`:

```text
traci==1.27.1
sumolib==1.27.1
```

### Virtual environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

then activate again.

### Install project

```powershell
python -m pip install --upgrade pip
pip install -e .
```

For development/testing:

```powershell
pip install -e ".[dev]"
```

The development extras include pytest, coverage, Ruff and mypy.

### Verify Python dependencies

```powershell
python -c "import torch, numpy, yaml, traci, sumolib, salabim; print('dependencies OK')"
```

### Run tests

```powershell
pytest
```

---

## 4.2 Instructions to run the project

### A. Simulation only — mock data, headless SUMO

```powershell
python scripts/simulate.py --config configs/experiments/small.yaml --mode mock --visualization false --report true
```

Execution:

```text
small.yaml
   ↓
MockDatasetGenerator
   ↓
datasets/generated/mock_seed_42/dataset.json
   ↓
SUMO network/route generation
   ↓
SUMO + TraCI
   ↓
outputs/.../simulation trace
   ↓
optional analytical CSV report
```

### B. Simulation only — mock data with SUMO-GUI

```powershell
python scripts/simulate.py --config configs/experiments/small.yaml --mode mock --visualization true --report true
```

The program adds SUMO-GUI startup options so the simulation begins automatically rather than waiting for the Run button.

### C. Simulation only — real dataset

Prepare:

```text
datasets/scenarios/my_scenario.json
```

then configure:

```yaml
simulation:
  mode: real
  dataset_path: datasets/scenarios/my_scenario.json
```

and run:

```powershell
python scripts/simulate.py --config configs/default.yaml --mode real --visualization false --report true
```

### D. Training — small mock scenario

```powershell
python scripts/train.py --config configs/experiments/small.yaml --mode mock --episodes 100
```

### E. Training — medium mock scenario

Recommended diagnostic run:

```powershell
python scripts/train.py --config configs/experiments/medium.yaml --mode mock --episodes 300
```

This is the preferred starting point for examining:

```text
budget violation rate
reward
TD loss
Actor gradient norm
Critic gradient norm
travel time
charging cost
```

### F. Training — large scenario

```powershell
python scripts/train.py --config configs/experiments/large.yaml --mode mock --episodes 1000
```

The file itself defaults to a smaller episode count, so the CLI override `--episodes 1000` is useful for extended experiments.

### G. Training with a real dataset

Set:

```yaml
training:
  mode: real
  dataset_path: datasets/scenarios/my_scenario.json
```

then:

```powershell
python scripts/train.py --config configs/training.yaml --mode real --episodes 300
```

The trip-sets in the dataset remain fixed through training episodes. Only the learned policy/action and resulting environment/SUMO outcomes change.

### H. Evaluate a checkpoint

```powershell
python scripts/evaluate.py `
  --config configs/experiments/medium.yaml `
  --mode mock `
  --episodes 10 `
  --checkpoint outputs/training_medium/checkpoints/episode_00025.pt
```

### I. Run visualization helper

```powershell
python scripts/visualize.py --config configs/experiments/medium.yaml --mode mock
```

### J. Installed console commands

After `pip install -e .`, the following aliases are available:

```powershell
nash-drl-simulate
nash-drl-train
nash-drl-evaluate
nash-drl-infer
nash-drl-visualize
```

For example:

```powershell
nash-drl-train --config configs/experiments/medium.yaml --mode mock --episodes 300
```

---

# Output organization

## Simulation outputs

Typical simulation artifacts include:

```text
outputs/demo/
├── dataset.json
├── simulation_trace.csv
├── analytical_report.csv
└── sumo_scenario/
    ├── network.nod.xml
    ├── network.edg.xml
    ├── network.net.xml
    ├── routes.rou.xml
    └── simulation.sumocfg
```

The exact directory depends on the configured `output_dir` and simulation runner.

## Training outputs

A training run normally creates:

```text
outputs/training_medium/
├── dataset.json
├── training_metadata.json
├── episode_results.csv
├── step_results.csv
├── vehicle_results.csv
├── edge_results.csv
├── checkpoints/
│   └── episode_*.pt
├── plots/
│   ├── episode_reward.png
│   ├── td_loss.png
│   ├── actor_gradient_norm.png
│   ├── critic_gradient_norm.png
│   ├── budget_violations.png
│   ├── budget_violation_rate.png
│   ├── mean_step_reward.png
│   ├── travel_time.png
│   ├── charging_cost.png
│   ├── actor_loss.png
│   └── critic_loss.png
└── sumo_runs/
    └── episode_*/
        ├── network/
        └── step_*/
```

`actor_loss.csv`/`critic_loss.csv` style scalar traces may look numerically similar because the V12 formulation uses the same squared TD residual while detaching different terms for the alternating Actor and Critic updates. For diagnosing whether the two networks actually learn differently, use:

```text
actor_gradient_norm
critic_gradient_norm
mean_td_error
mean_advantage
```

rather than interpreting the shared scalar residual as two independent objectives.

---

# Important implementation notes

## 1. The environment and neural model are separate

SUMO provides microscopic vehicle movement and telemetry. The NashDRL environment retains the mathematical model for charging cost, congestion travel time, and hard-budget reward. This permits SUMO to function as the execution layer without replacing the mathematical problem definition.

## 2. The trip-set is fixed across episodes

Mock-data generation occurs before training and produces a single dataset. `TrainingRunner` loads that dataset once, and each episode resets vehicle progress against the same trips. Therefore the learning problem is not changing its task definition from episode to episode.

## 3. Different trip-set sizes are supported

The model uses a fixed `N`-vehicle tensor representation. Vehicles that have completed all trips remain represented but are marked inactive/terminal. This preserves tensor shape while accommodating different trip-set lengths.

## 4. The action is continuous, the executed route is discrete

The Actor predicts continuous edge weights:

```text
[N,E]
```

The deterministic mapper converts them into edge sequences. The current production mapper is Dijkstra-based.

## 5. Hard budget violations are explicit learning signals

The reward model does not remove a violating vehicle from the system. Instead it assigns the configured `-P` penalty to that vehicle for the current trip-step.

Therefore experiments must distinguish:

```text
reward optimization
```

from:

```text
hard-constraint satisfaction
```

A sufficiently high penalty may be necessary for a constraint-dominant experiment, while a paper-faithful experiment should retain the specified penalty value.

## 6. The current analytical Nash policy is an explicit extension point

`game/nash_solver.py::AnalyticalNashPolicy.select()` currently returns `params.mu`. The network already produces all five LQ parameter tensors. The intended architecture allows the exact analytical local Nash solution to be implemented there without modifying Actor, Critic, environment, or replay-buffer interfaces.

---

# Recommended development order

When extending the research implementation, use this sequence:

```text
1. Validate domain/data with unit tests
        ↓
2. Validate equations independently of SUMO
        ↓
3. Validate deterministic route mapping
        ↓
4. Validate StateFeatureExtractor tensor contracts
        ↓
5. Validate Actor/Critic/Target Critic
        ↓
6. Validate one environment trip-step
        ↓
7. Validate one training update
        ↓
8. Run small benchmark
        ↓
9. Run medium constraint-learning benchmark
        ↓
10. Run large/scalability experiment
```

This order minimizes the risk of interpreting a SUMO, data-generation, reward, or tensor bug as an RL convergence problem.

---

# Reference to NashDRL-Model-V12

The repository implementation directly reflects the principal structures in NashDRL-Model-V12:

```text
Section 2
Problem formulation
    G=(V,E)
    vehicle model
    charging price
    energy cost
    congestion/travel time
    reward

Section 3
State/action model
    CSR map
    [N,F] agent features
    [E] edge flows
    [N,N-1,F] invariant network input
    [N,F+E] non-invariant network input
    [N,E] edge-weight action

Section 4
Network design
    Deep Sets
    Actor → μ,P11,P12,P22,Ψ
    Critic → V(x)
    Target Critic → Vslow(x')
    LQ advantage
    TD target / TD error
    alternating Actor/Critic optimization
```

The current implementation therefore has a clean separation between the **research formulation**, the **SUMO simulation platform**, and the **learning infrastructure**, allowing subsequent research work to focus on analytical Nash solving, policy improvement, experiment design, and statistical evaluation rather than rewriting the system architecture.


## PPO Baseline

The project includes a PPO baseline for direct comparison with NashDRL. PPO uses exactly the same dataset generator, SUMO/TraCI environment, state feature extractor, action-to-path mapper, reward model, trip-set horizon, and CSV report schema as NashDRL. The algorithmic difference is the learning objective.

The NashDRL implementation learns the LQ/game-theoretic advantage parameters (`mu`, `P11`, `P12`, `P22`, `Psi`) and uses its analytical Nash decomposition. PPO instead uses a stochastic Gaussian actor and a scalar global value critic and optimizes the standard clipped PPO policy objective against the total system reward. The original PPO algorithm is described by Schulman et al. (2017). Note that PPO itself still uses a standard policy-gradient advantage estimator (this project uses GAE); this must not be confused with the NashDRL LQ advantage function. See `docs/PPO_BASELINE.md` for the distinction.

Run NashDRL and PPO on the same medium benchmark:

```powershell
python scripts/train.py --config configs/experiments/medium.yaml --mode mock --episodes 300 --algorithm nash_drl
python scripts/train.py --config configs/experiments/medium.yaml --mode mock --episodes 300 --algorithm ppo
python scripts/compare_algorithms.py
```

The comparison report is written to `outputs/comparison/nash_vs_ppo.csv` and uses the common metrics `total_reward`, `budget_violations`, `budget_violation_rate`, `total_travel_time_h`, and `total_charging_cost`.

The PPO training report is written using the same filenames as NashDRL, including `episode_results.csv`, `step_results.csv`, `vehicle_results.csv`, and `edge_results.csv`, plus PPO-specific diagnostics such as `policy_loss_mean`, `value_loss_mean`, `entropy_mean`, and `approx_kl`.


### Paired experiments

NashDRL and PPO use the same `configs/experiments/small.yaml`, `medium.yaml`, and `large.yaml`; shared dataset/environment/reward/simulation sections are common, while algorithm-specific settings live under `training` and `ppo`/`ppo_network`.
