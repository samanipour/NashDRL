# Testing Strategy

## Unit tests

Unit tests verify local mathematical and data contracts.

```text
test_csr_map.py
    CSR indices, edge lookup, flow storage

test_congestion.py
    congestion/time equation

test_charging.py
    flow-dependent pricing

test_reward.py
    budget branch and total reward

test_mapper.py
    valid deterministic path construction

test_features.py
    tensor shapes and permutation invariance

test_actor.py
    Actor output shape and parameter constraints

test_critic.py
    per-agent value shape

test_lq_advantage.py
    analytical advantage shape and numerical checks
```

## Integration tests

Integration tests verify the system boundaries:

- `test_environment_step.py` — route selection → environment transition.
- `test_actor_environment.py` — model output → mapper → environment.
- `test_training_step.py` — one complete optimization step.

## Invariants worth testing

1. CSR row pointer length is `V+1`.
2. Edge arrays are length `E`.
3. `edge_flow` remains non-negative.
4. Mapper output follows graph connectivity.
5. Rival-agent permutation does not change the Deep Sets representation.
6. Actor output has exactly five channels.
7. Critic output has one value per vehicle.
8. Target Critic does not receive normal optimizer gradients.
9. A budget violation activates the hard penalty branch.
10. Training does not mutate the environment through the model layer.
