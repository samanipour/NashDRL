# System Architecture


## 1. System boundaries

The system has five primary boundaries:

1. **Problem/domain boundary** — road graph, edges, vehicles, trips, charging parameters.
2. **Environment boundary** — current state, route execution, edge-flow updates, travel time, charging cost, reward.
3. **Representation boundary** — conversion of environment state into neural-network tensors.
4. **Learning boundary** — Actor, Critic, Target Critic, LQ advantage/game logic and parameter updates.
5. **Experiment boundary** — configuration, evaluation, reporting, visualization and checkpointing.

## Simulation-first architecture

The current implementation adds a fully independent simulation pipeline before DRL learning:

```text
YAML configuration
       │
       ▼
MockDatasetGenerator ──────► dataset.json
       │                           │
       │ real mode ────────────────┘
       ▼
ProblemDefinition
       │
       ▼
SumoScenarioBuilder
       │
       ├── network.nod.xml
       ├── network.edg.xml
       ├── network.net.xml
       ├── routes.rou.xml
       └── simulation.sumocfg
       │
       ▼
SUMO / TraCI
       │
       ├── vehicle telemetry
       ├── edge telemetry
       └── system telemetry
       │
       ├── simulation_trace.csv
       └── analytical_report.csv
       │
       ▼
Salabim replay (optional)
```

The non-learning demo uses deterministic shortest-path routing from the generated graph. This is explicitly a simulation baseline, not the final Nash policy.

## Future DRL boundary

```text
Domain objects
    │
    ▼
Environment state
    │
    ▼
StateFeatureExtractor
    │
    ├──────────────► invariant [N,N-1,F]
    │
    └──────────────► non-invariant [N,F+E]
                        │
                        ▼
               Deep Sets + FC trunk
                        │
                        ▼
                 ActorOutput [5,N,E]
                        │
                        ▼
                    action [N,E]
                        │
                        ▼
               ActionToPathMapper
                        │
                        ▼
                    paths
                        │
                        ▼
                  Environment
                    /       \\
                 reward   next_state
                    │          │
                    │          ▼
                    │     Target Critic
                    │          │
                    ▼          ▼
                      TD / LQ loss
                            │
                            ▼
                      optimizers
```

## 3. Environment boundary

`environment/env.py` exposes the lifecycle:

```python
reset() -> GlobalState
step(paths) -> EnvironmentStep
```

`EnvironmentStep` contains at minimum:

- next state
- reward
- termination status
- diagnostic information

The environment is responsible for physical/economic state transition, not learning.

## 4. Routing boundary

The model emits edge weights, not paths. The routing layer translates these weights into valid routes. This is intentionally non-trainable and deterministic.

```python
mapper.map(action, state) -> Paths
```

Routing algorithms can therefore evolve independently of the neural model.

## 5. Learning boundary

The training engine consumes model inputs and orchestrates:

```text
Actor → action → mapper → environment
Critic → V(x)
Target Critic → Vslow(x')
LQ advantage → A(x,u)
Losses → optimizer updates
```

The training package owns the sequencing, while each model owns only its own computation.

## 6. Design constraints

### No circular dependencies

The following is forbidden conceptually:

```text
Actor → Environment → Actor
```

Instead:

```text
Trainer → Actor
Trainer → Environment
Trainer → Mapper
```

### Explicit tensor contracts

Tensor shapes must be documented at module boundaries and tested.

### Deterministic components stay deterministic

The route mapper should not receive gradients or optimizer state.

### Research equations are isolated

Any equation from the research model should be implemented in a small function/class with a direct unit test.
