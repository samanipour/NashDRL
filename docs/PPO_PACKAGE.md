# PPO Package Architecture

PPO is intentionally isolated from `nash_drl` so that algorithm-specific implementation does not enter the NashDRL package.

```text
src/
├── nash_drl/                 # NashDRL algorithm + shared domain/runtime infrastructure
└── ppo/                      # Independent PPO baseline
    ├── models/
    │   └── ppo.py
    ├── training/
    │   ├── ppo_rollout.py
    │   ├── ppo_trainer.py
    │   └── ppo_runner.py
    └── evaluation/
        ├── ppo_evaluator.py
        └── ppo_runner.py
```

The PPO package may import shared simulation, state, routing, feature-extraction, configuration, checkpoint, and visualization services from `nash_drl`. The inverse dependency is avoided: `nash_drl` core packages do not import PPO implementation modules.

## Responsibilities

| Package | Responsibility |
|---|---|
| `ppo.models` | PPO actor-critic neural network and Gaussian policy |
| `ppo.training` | rollout storage, GAE, clipped PPO updates, training orchestration |
| `ppo.evaluation` | PPO checkpoint loading and evaluation |
| `nash_drl.domain` | shared graph, vehicle, trip and problem model |
| `nash_drl.data` | shared state/action/tensor containers |
| `nash_drl.environment` | SUMO/TraCI simulation and reward model |
| `nash_drl.features` | shared state-to-network feature extraction |
| `nash_drl.routing` | shared deterministic action-to-path mapping |

This separation makes the experimental comparison explicit: both algorithms consume the same environment, state representation, route mapper and reward model, while their policy/value-learning implementations remain independent.
