# Project Structure and Module Organization

## Top-level directories

### `configs/`
Experiment configuration. No Python implementation belongs here.

### `scripts/`
Thin CLI entry points. Scripts should call package APIs instead of containing research logic.

### `src/nash_drl/`
Installable application/library code.

### `tests/`
Unit, integration and fixture data.

### `datasets/`
Maps, scenarios and generated data. Large datasets should not be committed blindly.

### `checkpoints/`, `logs/`, `outputs/`
Runtime artifacts.

## Package responsibilities

| Module | Primary responsibility |
|---|---|
| `domain.graph` | Graph/edge entities and graph-level validation |
| `domain.vehicle` | Vehicle entity and runtime vehicle metadata |
| `domain.trip` | Ordered trips and trip progression |
| `domain.charging` | Charging-station parameters |
| `domain.problem` | Problem-level composition/configuration |
| `data.csr_map` | Compact CSR graph tensors |
| `data.state` | Environment and model-facing state structures |
| `data.action` | `[N,E]` actions and route collections |
| `data.transition` | Single-step transition |
| `data.batch` | Batched training data |
| `environment.env` | Environment orchestration |
| `environment.simulator` | Scenario/episode execution helpers |
| `environment.congestion` | Congestion equation |
| `environment.travel_time` | Travel-time equation |
| `environment.energy_cost` | Charging/energy cost calculation |
| `environment.reward` | Vehicle/system reward calculation |
| `routing.mapper` | Mapper interface |
| `routing.dijkstra` | Dijkstra-based mapper |
| `routing.greedy` | Greedy mapper |
| `routing.constraints` | Route validity rules |
| `features.extractor` | Composite state-to-model conversion |
| `features.agent_features` | Per-agent feature extraction |
| `features.edge_features` | Edge/global feature extraction |
| `features.deep_set_input` | Construction of invariant tensors |
| `models.common` | Shared neural utilities |
| `models.deep_sets` | Shared permutation-invariant encoder |
| `models.actor` | Actor/LQ parameter network |
| `models.critic` | Value network |
| `models.target_critic` | Frozen critic + synchronization |
| `models.lq_advantage` | LQ advantage calculation |
| `game.lq_game` | LQ game representation |
| `game.nash_solver` | Nash/equilibrium calculation interface |
| `game.action_distribution` | Exploration/action construction |
| `training.trainer` | End-to-end learning orchestration |
| `training.rollout` | Environment interaction |
| `training.losses` | TD/actor/critic loss calculations |
| `training.optimizer` | Optimizer setup and stepping |
| `training.target_update` | Target network update policy |
| `training.checkpoint` | Save/restore training state |
| `evaluation.metrics` | Research metrics |
| `evaluation.evaluator` | Evaluation orchestration |
| `evaluation.reports` | Tables/JSON/CSV reports |
| `visualization.network` | Graph visualization |
| `visualization.routes` | Path/flow visualization |
| `visualization.training` | Training curve visualization |
| `utils.*` | Small cross-cutting helpers |

## File size rule

Prefer focused modules. A file should generally have one major reason to change. Equation modules should stay small and independently testable.
