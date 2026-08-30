# Nash-DRL

Nash-DRL is a research implementation for multi-agent electric autonomous vehicle routing. This repository currently provides a complete **simulation, environment, reproducible mock-data, SUMO/TraCI integration, and visualization layer**. The DRL learning loop is intentionally not required for the simulation demo stage.

## Simulation objective

The simulation layer translates the problem model into a microscopic traffic experiment:
## 1. Research scope

The project models a heterogeneous set of autonomous electric vehicles traveling on a directed road network. Each vehicle has an origin/destination sequence, free-flow speed, and budget. Vehicles select routes jointly; shared edge flow affects both congestion/travel time and flow-dependent charging prices. The research objective is to study policies that improve system-level payoff while avoiding hard budget-constraint violations.

The implementation is based on **Nash DRL, Document Version 12 (2026-08-22)**, especially Sections 2–4:

- Section 2 defines the road-network, vehicle, charging-cost, congestion/travel-time, and reward models.
- Section 3 defines the CSR map representation, the global neural-network state, and the `[N,E]` action-weight representation.
- Section 4 defines the Actor, Critic, Target Critic, Deep Sets aggregation, LQ advantage representation, and training data flow.

The source document uses a directed graph `G=(V,E)`, edge length/capacity, heterogeneous vehicles, flow-dependent charging price, congestion-dependent travel time, and a piecewise reward that imposes a hard penalty when cost exceeds a vehicle budget. See the source document for the authoritative mathematical definitions.

## 2. Core system flow

```text
Problem Definition
       │
       ▼
CSR Map + Vehicles + Trips
       │
       ▼
Environment State
       │
       ▼
State Feature Extraction
       │
       ├───────────────┐
       ▼               ▼
Invariant Stream    Non-Invariant Stream
[N,N-1,F]           [N,F+E]
       │               │
       └──────┬────────┘
              ▼
       Actor / Critic
              │
       ActorOutput [5,N,E]
       (μ,P11,P12,P22,Ψ)
              │
              ▼
       Action [N,E]
              │
              ▼
       Action-to-Path Mapper
              │
              ▼
       Valid paths for vehicles
              │
              ▼
          Environment
              │
       ┌──────┴──────┐
       ▼             ▼
     Reward       Next State
       │             │
       └──────┬──────┘
              ▼
       Critic + Target Critic
              │
              ▼
       TD / Advantage losses
              │
              ▼
        Parameter updates
```

## 3. Architectural principles

### 3.1 Domain and learning logic are separated

`domain/` contains the mathematical entities. It should not depend on PyTorch.

### 3.2 Runtime tensors have explicit contracts

The implementation treats tensor shapes as part of the public interface:

| Tensor | Shape | Meaning |
|---|---:|---|
| Agent features | `[N,F]` | Per-vehicle state features |
| Edge flow | `[E]` | Current flow per edge |
| Invariant input | `[N,N-1,F]` | Other-agent features for each focal vehicle |
| Non-invariant input | `[N,F+E]` | Focal/self features plus global edge-flow features |
| Actor output | `[5,N,E]` | `μ`, `P11`, `P12`, `P22`, `Ψ` |
| Action | `[N,E]` | Raw edge weights for each vehicle |
| Critic value | `[N]` | Per-agent value estimate |

### 3.3 The route mapper is deterministic

The Actor does not directly emit a variable-length route. It emits raw edge weights. A non-trainable `ActionToPathMapper` converts those weights into valid paths subject to routing constraints.

### 3.4 The Target Critic is a replica, not a second architecture

`target_critic.py` should wrap or reuse the same Critic architecture and provide synchronization utilities. The target network is frozen between synchronization operations.

### 3.5 Mathematical equations live in focused modules

Charging price, congestion, travel time, reward, LQ advantage, and route construction should each be independently testable.

## 4. Repository structure

```text
nash-drl/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── configs/
│   ├── default.yaml
│   ├── network.yaml
│   ├── environment.yaml
│   ├── training.yaml
│   └── experiments/
│       ├── small.yaml
│       └── large.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── visualize.py
│
├── src/
│   └── nash_drl/
│
│       ├── __init__.py
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   ├── vehicle.py
│       │   ├── trip.py
│       │   ├── charging.py
│       │   └── problem.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── csr_map.py
│       │   ├── state.py
│       │   ├── action.py
│       │   ├── transition.py
│       │   └── batch.py
│       │
│       ├── environment/
│       │   ├── __init__.py
│       │   ├── env.py
│       │   ├── simulator.py
│       │   ├── congestion.py
│       │   ├── travel_time.py
│       │   ├── energy_cost.py
│       │   └── reward.py
│       │
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── mapper.py
│       │   ├── dijkstra.py
│       │   ├── greedy.py
│       │   └── constraints.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── extractor.py
│       │   ├── agent_features.py
│       │   ├── edge_features.py
│       │   └── deep_set_input.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── common/
│       │   │   ├── mlp.py
│       │   │   ├── activations.py
│       │   │   └── initialization.py
│       │   │
│       │   ├── deep_sets.py
│       │   ├── actor.py
│       │   ├── critic.py
│       │   ├── target_critic.py
│       │   └── lq_advantage.py
│       │
│       ├── game/
│       │   ├── __init__.py
│       │   ├── lq_game.py
│       │   ├── nash_solver.py
│       │   └── action_distribution.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── rollout.py
│       │   ├── losses.py
│       │   ├── optimizer.py
│       │   ├── target_update.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── evaluator.py
│       │   └── reports.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── network.py
│       │   ├── routes.py
│       │   └── training.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── seeding.py
│           ├── device.py
│           └── serialization.py
│
├── tests/
│   ├── unit/
│   │   ├── test_csr_map.py
│   │   ├── test_congestion.py
│   │   ├── test_charging.py
│   │   ├── test_reward.py
│   │   ├── test_mapper.py
│   │   ├── test_features.py
│   │   ├── test_actor.py
│   │   ├── test_critic.py
│   │   └── test_lq_advantage.py
│   │
│   ├── integration/
│   │   ├── test_environment_step.py
│   │   ├── test_actor_environment.py
│   │   └── test_training_step.py
│   │
│   └── fixtures/
│       └── small_graph.yaml
│
├── datasets/
│   ├── maps/
│   ├── scenarios/
│   └── generated/
│
├── checkpoints/
├── logs/
└── outputs/
```

## 5. Layer responsibilities

| Package | Responsibility | Should know about |
|---|---|---|
| `domain` | Mathematical problem entities | Domain only |
| `data` | Runtime tensor/data containers | Domain + tensors |
| `environment` | Transition and physical/economic models | Domain + data |
| `routing` | Convert edge weights to valid paths | Domain + data |
| `features` | Convert state to model inputs | Data |
| `models` | Actor/Critic/Deep Sets/LQ parameterization | Tensor interfaces |
| `game` | LQ game and Nash calculations | Model outputs + actions |
| `training` | Rollout, losses, optimization, checkpoints | Environment + models + game |
| `evaluation` | Metrics and reports | Environment outputs |
| `visualization` | Graph/routes/training plots | Domain + evaluation outputs |
| `utils` | Cross-cutting utilities | Minimal dependencies |

## 6. Dependency direction

The intended dependency direction is:

```text
domain
  ↑
data
  ↑
features ───────► models
  ↑                  ↑
  └──────────── training ◄──── game
                    ↑
             environment
                    ↑
                 routing
```

The actual Python rule is simpler:

- `domain` never imports from `models`, `training`, or `environment`.
- `models` never call the environment directly.
- `routing` does not perform learning.
- `environment` does not know the Actor/Critic implementation.
- `training` is the orchestration layer.


```text
Mock/Real Dataset
      │
      ▼
ProblemDefinition
      │
      ▼
SUMO Scenario Builder
      │
      ├── network.nod.xml
      ├── network.edg.xml
      ├── network.net.xml
      ├── routes.rou.xml
      └── simulation.sumocfg
      │
      ▼
SUMO + TraCI
      │
      ├── vehicle observations
      ├── edge observations
      └── system observations
      │
      ├── simulation_trace.csv
      └── analytical_report.csv
```

SUMO 1.27.1 is the pinned simulator/client target. TraCI is used as the Python control interface. The current stable release is SUMO 1.27.1 (25 June 2026), and the matching `traci` and `sumolib` Python packages are available as version 1.27.1. The salabim dependency is used for optional trajectory replay/animation. See the project documentation for installation details.

## 7. Installation
Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\\Scripts\\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -e .
```

Development dependencies:

Install SUMO 1.27.1 separately and make sure `sumo`, `sumo-gui`, and `netconvert` are discoverable. Alternatively set `SUMO_HOME` to the SUMO installation directory.
```bash
pip install -e ".[dev]"
```

Run the reproducible mock simulation:

```bash
python scripts/simulate.py --config configs/experiments/small.yaml --mode mock --visualization false --report true
```

or:

```bash
nash-drl-simulate --config configs/default.yaml --mode mock --visualization true --report true
```

The run first creates a deterministic dataset under `datasets/generated/`, then converts that dataset into a SUMO scenario, runs SUMO through TraCI, and finally writes trace/report artifacts under the configured output directory.

## Runtime options

`simulation.mode` has two values:

- `mock`: generate a new reproducible dataset from `mock_data` and use that dataset as the simulator input.
- `real`: load an existing Nash-DRL dataset JSON from `simulation.dataset_path`.

`simulation.visualization` controls whether `sumo-gui` is used instead of headless `sumo`.

`simulation.generate_analytic_report` controls creation of the consolidated CSV analytical report. The report contains vehicle, edge, trip, and system rows and includes both static entity parameters and dynamic observations collected during the SUMO run.

## Mock dataset generation

The mock generator is deterministic with respect to its configured seed. It creates:

- a strongly connected directed road network;
- edge length and capacity;
- heterogeneous vehicles;
- one to many trips per vehicle;
- speed and budget parameters;
- simulation-model parameters.

The generated JSON contains the complete dataset and generator configuration so the experiment is reproducible.

## Output artifacts

A simulation run writes approximately:

```text
outputs/<run>/
├── dataset.json
├── simulation_trace.csv
├── analytical_report.csv
├── run_metadata.json
└── sumo/
    ├── network.nod.xml
    ├── network.edg.xml
    ├── network.net.xml
    ├── routes.rou.xml
    └── simulation.sumocfg
```

`simulation_trace.csv` contains dynamic observations collected through TraCI. `analytical_report.csv` combines vehicle-, edge-, and system-level observations with relevant static entity parameters.

## Architecture

The simulation code is deliberately separated from the DRL code:

```text
src/nash_drl/
├── domain/          mathematical problem entities
├── data/            tensors + dataset generation
├── environment/     mathematical environment + SUMO execution
├── routing/         deterministic route mapping
├── features/        state feature construction
├── models/          Actor/Critic/Deep Sets/LQ models
├── game/            game-theoretic mathematics
├── training/        learning orchestration
├── evaluation/      metrics and reports
├── visualization/   visualization and salabim replay
└── utils/           cross-cutting utilities
```

The current simulation demo uses `domain`, `data`, `environment`, `routing`, `evaluation`, `visualization`, and `utils`. It does not invoke the learning components.

## Research model correspondence

The road network is represented as a directed graph, vehicles are heterogeneous, edge flow drives both charging price and congestion, and reward depends on travel time, charging cost, and the hard budget constraint. The neural-network state/action structures remain available for the later DRL stage: CSR map, `[N,F]` agent features, `[N,N-1,F]` permutation-invariant input, `[N,F+E]` non-invariant input, and `[N,E]` action weights.

## Real-data mode

A real-data JSON uses the same schema as a generated dataset. A minimal dataset has:

```json
{
  "schema_version": 1,
  "graph": {"num_nodes": 4, "edges": [...]},
  "vehicles": [
    {"id": 0, "free_flow_speed_kmh": 60, "budget": 150,
     "trips": [{"origin": 0, "destination": 3}]}
  ]
}
```

The simulator converts the graph into SUMO node/edge files, builds deterministic routes for all trips, and runs those routes in SUMO. This mode is intended for later replacement of the generated data with actual scenario datasets without changing the simulation runtime.

## Verification

The repository includes unit and integration tests for the mathematical environment, routing, feature extraction, and network components. SUMO-dependent integration tests should be executed on a machine with SUMO 1.27.1 installed.

## Current implementation boundary

No learning or DRL optimization is executed by `scripts/simulate.py`. The route policy used for the demo is deterministic shortest-path routing. The existing Actor/Critic/game/training packages remain available for the subsequent learning stage.

## 8. Basic commands

```bash
nash-drl-train --config configs/default.yaml
nash-drl-evaluate --config configs/default.yaml
nash-drl-infer --checkpoint checkpoints/latest.pt
nash-drl-visualize --input outputs/
```

The scripts in `scripts/` are also usable directly:

```bash
python scripts/train.py --config configs/default.yaml
```

## 9. Testing

```bash
pytest
```

Run focused tests:

```bash
pytest tests/unit/test_csr_map.py
pytest tests/unit/test_features.py
pytest tests/integration/test_training_step.py
```

## 10. Configuration

Configuration is deliberately split into:

- `default.yaml`: composition and shared experiment defaults.
- `network.yaml`: feature sizes, hidden dimensions, activation, Actor/Critic settings.
- `environment.yaml`: map, charging, congestion, travel-time, reward settings.
- `training.yaml`: optimizer, learning rates, discount, rollout, target update, checkpointing.
- `configs/experiments/*.yaml`: experiment-specific overrides.

See `docs/CONFIGURATION.md`.

## 11. Data and tensor contracts

See `docs/DATA_MODEL.md` for the canonical definition of:

- CSR arrays
- agent features
- edge-flow features
- invariant/non-invariant model inputs
- Actor output channels
- action representation
- transition and batch objects

## 12. Neural-network architecture

See `docs/NETWORK_DESIGN.md` for the implementation contract of:

- Deep Sets
- Actor
- Critic
- Target Critic
- LQ advantage
- tensor dimensions and information flow

## 13. Implementation status

### Implemented scaffolding

- Package and build configuration
- Domain models
- CSR representation
- State/action/transition structures
- State feature extraction interfaces
- Deep Sets and feed-forward building blocks
- Actor/Critic interfaces
- Target-Critic synchronization interface
- Routing interfaces and reference mappers
- Environment component interfaces
- Training orchestration interfaces
- Evaluation and visualization interfaces
- Tests and fixtures

### Explicit research implementation points

The repository intentionally does not claim that every research equation is fully finalized. In particular, the exact LQ/Nash derivation, the full multi-trip episode transition semantics, complete route-constraint policy, and any training details not specified by the source document must be implemented and validated as part of the research process.

## 14. Development workflow

Recommended order:

1. Validate the domain and CSR map.
2. Validate charging, congestion, travel-time, and reward equations independently.
3. Build a deterministic environment scenario and reproduce the document's illustrative comparison.
4. Validate route mapping and path constraints.
5. Validate feature extraction and permutation invariance.
6. Unit-test Actor/Critic tensor shapes and positive-definiteness constraints.
7. Implement and test the LQ game/advantage calculation.
8. Connect environment, Actor, Critic, Target Critic, and training loop.
9. Add experiment configurations and evaluation protocols.
10. Add large-scale datasets and profiling.

## 15. Research reproducibility

Every experiment should save:

```text
outputs/<experiment>/
├── config.yaml
├── metrics.csv
├── summary.json
├── plots/
└── checkpoints/
```

Set random seeds through `nash_drl.utils.seeding` and record the exact configuration used for each run.

## 16. Source of truth

The authoritative mathematical specification for the current project is the research document **Nash DRL, Document Version 12, last modified 2026-08-22**. This repository documents an implementation architecture for that specification; it should not silently reinterpret unspecified equations or algorithms.
