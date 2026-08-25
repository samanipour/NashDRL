# Module and Class Guide

This document maps the architecture to concrete Python modules and expected responsibilities.

## `domain/`

### `domain/graph.py`
**Primary types:** `Edge`, `DirectedGraph`

Responsibilities:

- represent directed graph topology;
- expose node/edge counts;
- provide structural graph queries.

Must not:

- calculate rewards;
- run neural networks;
- perform optimization.

### `domain/trip.py`
**Primary type:** `Trip`

Represents one origin/destination leg.

### `domain/vehicle.py`
**Primary type:** `Vehicle`

Stores vehicle identity, trip sequence, free-flow speed, budget, and current progress fields needed by the scenario model.

### `domain/problem.py`
**Primary type:** `ProblemDefinition`

Groups the graph, vehicles, and scenario-level configuration into a problem definition.

## `data/`

### `data/csr_map.py`
**Primary type:** `CSRMap`

Owns the CSR arrays and edge-property tensors.

### `data/state.py`
**Primary type:** `GlobalState`

Represents the evolving state consumed by feature extraction.

### `data/action.py`
**Primary types:** `Action`, `Paths`

`Action` owns `[N,E]` edge weights. `Paths` owns variable-length mapped routes.

### `data/batch.py`
**Primary type:** `NetworkInputs`

Stores:

```text
invariant     [N,N-1,F]
non_invariant [N,F+E]
```

### `data/transition.py`
**Primary type:** `Transition`

Represents one training transition and should remain independent of a specific optimizer.

## `environment/`

### `environment/congestion.py`
Pure congestion calculation functions.

### `environment/travel_time.py`
Pure edge travel-time calculation.

### `environment/energy_cost.py`
Charging-price and energy/charging-cost functions.

### `environment/reward.py`
**Primary types:** `RewardConfig`, `RewardResult`, `RewardModel`

Owns the piecewise vehicle reward and hard budget penalty.

### `environment/simulator.py`
**Primary types:** `SimulationState`, `RoadSimulator`

Low-level route/flow state progression.

### `environment/env.py`
**Primary types:** `EnvironmentConfig`, `NashEnvironment`

Public environment API: reset, step, and scenario lifecycle.

## `features/`

### `features/agent_features.py`
Builds per-agent `[N,F]` features.

### `features/edge_features.py`
Builds global `[E]` edge features.

### `features/deep_set_input.py`
Validates invariant tensors and related shape assumptions.

### `features/extractor.py`
**Primary type:** `StateFeatureExtractor`

Combines agent/global features into the two neural input streams.

## `models/`

### `models/common/mlp.py`
Reusable MLP block.

### `models/deep_sets.py`
**Primary type:** `DeepSetEncoder`

Shared rival embedding and permutation-invariant aggregation.

### `models/actor.py`
**Primary types:** `ActorOutput`, `ActorNetwork`

Produces the five LQ parameter tensors.

### `models/critic.py`
**Primary type:** `CriticNetwork`

Produces `[N]` baseline values.

### `models/target_critic.py`
**Primary type:** `TargetCriticNetwork`

Replica of the Critic used for delayed/frozen target estimates.

## `game/`

### `game/lq_game.py`
**Primary type:** `LQAdvantage`

Implements the LQ advantage expression driven by Actor parameters and selected actions.

### `game/nash_solver.py`
**Primary type:** `AnalyticalNashPolicy`

Interface for deterministic/analytical Nash action calculation. This is the main research extension point for the final derivation.

### `game/action_distribution.py`
Exploration/noise helpers applied to the Actor's mean action.

## `routing/`

### `routing/mapper.py`
**Primary type:** `ActionToPathMapper`

Abstract deterministic mapper interface.

### `routing/dijkstra.py`
**Primary type:** `DijkstraMapper`

Shortest-path interpretation of edge weights.

### `routing/greedy.py`
**Primary type:** `GreedyMapper`

Greedy outgoing-edge interpretation.

### `routing/constraints.py`
Route validity checks.

## `training/`

### `training/rollout.py`
Collects environment/model interaction steps.

### `training/losses.py`
TD target, TD error, and actor/critic loss calculations.

### `training/optimizer.py`
Optimizer construction.

### `training/target_update.py`
Hard/soft synchronization between Critic and Target Critic.

### `training/checkpoint.py`
Save/load of experiment state.

### `training/trainer.py`
**Primary type:** `NashDRLTrainer`

Top-level orchestration only. Keep mathematical equations elsewhere.

## `evaluation/`

### `evaluation/metrics.py`
Pure metric functions.

### `evaluation/evaluator.py`
**Primary type:** `Evaluator`

Runs scenarios and aggregates metrics.

### `evaluation/reports.py`
Human-readable report formatting.

## `visualization/`

Visualization-only modules. They should not be required for training or inference.

## `utils/`

Cross-cutting helpers for devices, seeding, logging, and serialization.

## Public interfaces

The following interfaces should be considered stable first-class extension points:

```python
class ActionToPathMapper(ABC):
    @abstractmethod
    def map(self, action: Action, state: GlobalState) -> Paths: ...

class StateFeatureExtractor:
    def __call__(self, state: GlobalState) -> NetworkInputs: ...

class ActorNetwork(nn.Module):
    def forward(self, inputs: NetworkInputs) -> ActorOutput: ...

class CriticNetwork(nn.Module):
    def forward(self, inputs: NetworkInputs) -> Tensor: ...

class LQAdvantage:
    def __call__(self, actor_output: ActorOutput, action: Tensor) -> Tensor: ...

class NashEnvironment:
    def reset(self) -> GlobalState: ...
    def step(self, paths: Paths): ...
```

The exact method signatures in source may evolve while the semantic contracts remain unchanged.
