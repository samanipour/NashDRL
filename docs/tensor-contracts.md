# Tensor Contracts and Data Shapes

## 1. Purpose

This document defines the tensor-level API between data preparation, feature extraction, neural networks, routing, and training.

Treat these shapes as public contracts. A shape change should be accompanied by tests and documentation updates.

## 2. Symbols

| Symbol | Meaning |
|---|---|
| `N` | number of vehicles/agents |
| `F` | number of scalar features per vehicle |
| `E` | number of directed graph edges |
| `V` | number of graph nodes |
| `D` | Deep Sets latent dimension |

## 3. CSR map tensors

| Field | Shape | Type | Description |
|---|---:|---|---|
| `row_ptr` | `[V+1]` | integer | CSR row offsets |
| `col_idx` | `[E]` | integer | destination node IDs in CSR order |
| `edge_ids` | `[E]` | integer | edge ID corresponding to each CSR adjacency entry |
| `edge_len` | `[E]` | floating point | edge length in km |
| `edge_cap` | `[E]` | floating point | edge capacity |
| `edge_flow` | `[E]` | floating point | dynamic current flow |
| `edge_from` | `[E]` | integer | source node for every edge ID |

The structural arrays (`row_ptr`, `col_idx`, `edge_ids`, `edge_len`, `edge_cap`, `edge_from`) describe the scenario graph. `edge_flow` changes during simulation.

## 4. Global state

The state exposed to the feature extractor includes:

```text
agent_features : [N,F]
edge_flow      : [E]
```

The exact feature ordering is a configuration-level contract. The source specification requires the per-agent representation to include:

1. current stop ID;
2. next destination node ID;
3. final destination ID;
4. remaining destination/trip count;
5. remaining budget;
6. free-flow speed.

Implementations may encode node IDs as normalized scalar values or one-hot/embedded values, but the choice must be explicit and consistent across experiments.

## 5. Invariant stream

For every focal agent `i`, the feature extractor creates the feature matrix of the other agents:

```text
other[i] = {agent j | j != i}
```

Stacking all focal agents produces:

```text
invariant = [N, N-1, F]
```

The ordering of the `N-1` rival dimension is semantically irrelevant. Deep Sets must therefore be invariant to permutations along that dimension.

### Example

For `N=4` and `F=6`:

```text
invariant.shape == [4,3,6]
```

## 6. Non-invariant stream

The focal agent's own features are concatenated with global edge-flow features:

```text
self features : [N,F]
edge flow     : [E]
```

Broadcast/concatenation yields:

```text
non_invariant : [N,F+E]
```

For `N=4`, `F=6`, and `E=20`:

```text
non_invariant.shape == [4,26]
```

## 7. Deep Sets output

The shared embedding maps every rival vector to a latent vector:

```text
[N,N-1,F] → [N,N-1,D]
```

Summation over rivals yields:

```text
crowd_embedding : [N,D]
```

No information about rival ordering should remain in `crowd_embedding` except information contained in the multiset of rival features.

## 8. Actor output

The Actor returns five `[N,E]` tensors:

```text
mu   [N,E]
P11  [N,E]
P12  [N,E]
P22  [N,E]
psi  [N,E]
```

These may also be represented as one tensor:

```text
actor_raw.shape == [5,N,E]
```

The preferred Python API is the named `ActorOutput` dataclass so downstream code does not depend on numeric channel positions.

## 9. Parameter constraints

The source architecture requires positive-definiteness/positivity constraints for the LQ curvature parameters associated with `P11` and `P22`.

The exact parameterization should remain in `models/actor.py` until the final research implementation is fixed. A positive transform such as `softplus` is an implementation option, not a source-document fact and should therefore remain configurable/replaceable.

## 10. Action

The action presented to the route mapper is:

```text
action.edge_weights : [N,E]
```

Each row corresponds to one vehicle; each column corresponds to one directed graph edge.

Example:

```text
N = 3
E = 5

action.shape == [3,5]
```

## 11. Paths

The action mapper returns a variable-length edge sequence per vehicle:

```python
Paths.edge_ids: list[list[int]]
```

There is deliberately no fixed `[N,L]` tensor requirement because different vehicle routes can have different numbers of edges.

## 12. Critic

The Critic returns one scalar value estimate per vehicle:

```text
V(x) : [N]
```

For `N=8`:

```text
value.shape == [8]
```

## 13. TD target

The training target has the conceptual form:

```text
TD target = reward + gamma * target_value(next_state)
```

with terminal handling to suppress the bootstrap term for terminal transitions.

The current implementation uses vectorized tensors so the per-agent reward and value remain aligned along `[N]`.

## 14. Shape validation strategy

Every public tensor boundary should validate dimensions early. Recommended checks:

```python
assert agent_features.ndim == 2
assert agent_features.shape[0] == num_agents

assert edge_flow.ndim == 1
assert edge_flow.shape[0] == num_edges

assert invariant.shape == (num_agents, num_agents - 1, feature_dim)
assert non_invariant.shape == (num_agents, feature_dim + num_edges)
assert action.shape == (num_agents, num_edges)
```

Prefer explicit validation functions over scattered assertions inside unrelated layers.

## 15. Batching policy

The first implementation uses the document's single-scenario shape conventions. If batched training is introduced, add a leading batch dimension rather than changing the internal agent/edge semantics:

```text
Batched state:
agent_features      [B,N,F]
edge_flow           [B,E]
invariant           [B,N,N-1,F]
non_invariant       [B,N,F+E]
action              [B,N,E]
critic value        [B,N]
```

A batch dimension must never be confused with the agent dimension `N`.
