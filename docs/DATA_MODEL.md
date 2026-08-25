# Data Model and Tensor Contracts

## 1. Road network

The runtime map uses a CSR representation plus parallel edge-property arrays.

```text
row_ptr   : [V+1]
col_idx   : [E]
edge_ids  : [E]
edge_len  : [E]
edge_cap  : [E]
edge_flow : [E]
edge_from : [E]
```

`row_ptr[u]:row_ptr[u+1]` indexes the outgoing adjacency for node `u`.

`edge_ids` connects CSR adjacency positions to edge-property arrays.

## 2. Vehicle state

The model-facing vehicle features described by the source document include:

- current stop ID
- next destination node ID
- final destination ID
- remaining destination count
- remaining budget
- free-flow speed

The per-agent feature matrix is:

```text
agent_features: [N,F]
```

The project deliberately keeps the semantic features separate from the final neural input construction.

## 3. Global edge-flow features

```text
edge_flow_features: [E]
```

These represent current total flow on every edge and are included in the non-invariant model stream.

## 4. Neural input tensors

For each focal vehicle `i`, all rival vehicle features `j != i` form:

```text
invariant_input: [N,N-1,F]
```

The focal vehicle plus global edge-flow representation forms:

```text
non_invariant_input: [N,F+E]
```

The invariant stream is processed by a shared embedding function and aggregation operator so reordering rival vehicles does not change the representation.

## 5. Actor output

The Actor emits:

```text
actor_output: [5,N,E]
```

Channels:

```text
0 → μ
1 → P11
2 → P12
3 → P22
4 → Ψ
```

The implementation should expose named fields rather than requiring callers to remember numeric channel positions.

## 6. Action

The action is:

```text
action.edge_weights: [N,E]
```

Each row is the set of edge weights assigned by one vehicle.

The action is not itself a route. It becomes a route through the deterministic mapper.

## 7. Paths

The route-mapper output is variable-length per vehicle:

```text
Paths = [
    [edge_id, edge_id, ...],
    [edge_id, edge_id, ...],
    ...
]
```

There are `N` route lists.

## 8. Transition

A training transition conceptually contains:

```text
state
 → action
 → paths
 → reward
 → next_state
 → terminated/truncated
```

Diagnostics can additionally contain travel time, cost, edge flow, budget violations, and other environment outputs.

## 9. Batched tensors

Batching must preserve the leading batch dimension:

```text
agent_features : [B,N,F]
edge_flow      : [B,E]
invariant      : [B,N,N-1,F]
non_invariant  : [B,N,F+E]
action         : [B,N,E]
value          : [B,N]
```

If future implementations use variable `N` or `E` inside the same batch, explicit padding/masking or a packed representation must be introduced rather than silently changing shapes.
