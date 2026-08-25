# Development Guide

## 1. Implementation order

### Stage 1 — Domain and data

Implement and test:

- graph entities
- vehicle/trip entities
- CSR map construction and lookup
- state/action/transition containers

### Stage 2 — Environment equations

Implement and test:

- charging price
- energy/charging cost
- congestion
- travel time
- budget-based reward

### Stage 3 — Routing

Implement:

- route constraints
- deterministic Dijkstra mapper
- greedy mapper

### Stage 4 — Features

Implement:

- `[N,F]` agent feature extraction
- `[E]` edge-flow features
- `[N,N-1,F]` invariant construction
- `[N,F+E]` non-invariant construction

### Stage 5 — Networks

Implement and test:

- Deep Sets
- Actor
- Critic
- Target Critic
- LQ advantage

### Stage 6 — Game logic

Implement and validate the LQ game/Nash calculations supported by the research specification.

### Stage 7 — Training

Connect:

```text
feature extraction
→ Actor
→ action construction
→ mapper
→ environment
→ Critic / Target Critic
→ losses
→ optimizer
```

### Stage 8 — Research experiments

Add reproducible scenarios, baseline methods, evaluation metrics and reports.

## 2. Testing policy

Every mathematical component gets at least one deterministic unit test.

Integration tests should verify component boundaries rather than exact learned outcomes.

## 3. Debugging strategy

When a training result looks wrong, test in this order:

1. CSR adjacency and edge IDs.
2. Route validity.
3. Edge flow after route selection.
4. Charging price.
5. Travel time.
6. Reward and budget violation logic.
7. Feature tensor shapes.
8. Deep Sets permutation invariance.
9. Actor output shapes and constraints.
10. Critic/target synchronization.
11. Losses and optimizer updates.

## 4. Coding standards

- Python type hints for public interfaces.
- Small modules with explicit contracts.
- No hidden global mutable state.
- No environment calls from model classes.
- No optimizer logic inside model classes.
- Keep experimental alternatives behind interfaces or configuration.
