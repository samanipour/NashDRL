# Neural Network Design

## 1. Source specification

The implementation follows Section 4 of **NashDRL-Model-V12**. The paper defines three neural modules: an Actor that parameterizes the local LQ advantage, a Critic that estimates per-agent baseline value, and a Target Critic used for stable TD targets. All use permutation-invariant processing of the other-agent population. fileciteturn1file0L401-L410

The state feature extraction contract is:

```text
Invariant input      [N, N-1, F]
Non-invariant input  [N, F+E]
```

where the invariant stream contains the focal vehicle's other agents and the non-invariant stream contains focal-agent features plus the global edge-flow vector. fileciteturn1file0L446-L455

## 2. Deep Sets encoder

For focal agent `i`, the rival set is:

```text
X_-i : [N-1, F]
```

The same embedding network `phi` is applied independently to every rival and the resulting embeddings are summed:

```text
[N,N-1,F]
      |
      v
shared phi
      |
      v
[N,N-1,D]
      |
      | sum over rivals
      v
[N,D]
```

The sum is the permutation-invariant aggregation described in Section 4. The document does not prescribe the internal layer count of `phi`; this implementation uses a configurable MLP, defaulting to two hidden SiLU layers with 32 units and a 64-dimensional embedding. The main Actor/Critic trunk dimensions below are directly specified by the paper. fileciteturn1file0L511-L522

## 3. Actor

The Actor has two streams:

```text
Stream 1: [N,N-1,F] -> Deep Sets -> [N,D]
Stream 2: [N,F+E]   -> linear + SiLU -> [N,D]
```

The streams are concatenated and fed to the main Actor trunk. The paper specifies four fully connected hidden layers with 32 nodes and SiLU activation. The final layer produces five edge-wise parameter channels:

```text
[5,N,E]

0: mu
1: P11
2: P12
3: P22
4: Psi
```

Each channel is `[N,E]`. `P11` and `P22` are transformed with a strictly-positive softplus mapping. `mu`, `P12`, and `Psi` remain unconstrained. fileciteturn1file0L116-L136

### Implementation API

```python
outputs = actor(inputs)
outputs.mu     # [N,E]
outputs.p11    # [N,E]
outputs.p12    # [N,E]
outputs.p22    # [N,E]
outputs.psi    # [N,E]
outputs.as_tensor()  # [5,N,E]
```

A leading training batch dimension is also accepted:

```text
inputs.invariant      [B,N,N-1,F]
inputs.non_invariant  [B,N,F+E]

actor output          [B,5,N,E]
```

## 4. Critic

The Critic uses the same permutation-invariant rival representation:

```text
[N,N-1,F] -> Deep Sets -> [N,D]
```

The focal-agent features and global edge flows are retained as the non-invariant stream. They are flattened into the focal-agent feature vector and concatenated with the crowd representation. The resulting representation is passed through the paper-specified four-layer, 32-unit, SiLU value trunk.

The output is one baseline value per agent:

```text
V(x) : [N]
```

or for training batches:

```text
V(x) : [B,N]
```

This follows the architecture and output described in Section 4.3. fileciteturn1file0L163-L175

## 5. Target Critic

The Target Critic is an exact deep copy of the main Critic. Its parameters have `requires_grad=False`, it is kept in evaluation mode, and it is updated only through an explicit hard synchronization:

```python
target.hard_update_from(critic)
```

This matches the document's requirement that the Target Critic not be updated by backpropagation and instead periodically receive a hard copy of the Critic parameters. fileciteturn1file0L176-L183

## 6. LQ advantage interface

The Actor parameters feed the LQ advantage calculation. For every agent:

```text
z_i = u_i - mu_i
```

The source equation contains the ego quadratic term, pairwise interaction term, rival quadratic term, and linear tilt term. The implementation is vectorized over agents and edges in `models/lq_advantage.py`. fileciteturn1file0L79-L83

## 7. Relationship to the original example

The supplied Nash-DQN example uses `PermInvariantQNN` with a moment-based summary of invariant variables. changes this requirement: Section 4 explicitly calls for a shared Deep Sets embedding followed by summation. Therefore the production implementation does **not** copy the example's mean-moment operation. Instead, it preserves the example's fully-connected/SiLU implementation style where compatible and adapts the invariant processing to the current paper specification.

## 8. Shape contracts

| Component | Input | Output |
|---|---|---|
| Deep Sets | `[N,N-1,F]` | `[N,D]` |
| Deep Sets batched | `[B,N,N-1,F]` | `[B,N,D]` |
| Actor | `[N,N-1,F]`, `[N,F+E]` | `[5,N,E]` via `as_tensor()` |
| Actor fields | same | five `[N,E]` tensors |
| Critic | same | `[N]` |
| Target Critic | same | `[N]` |
| LQ Advantage | Actor output + `[N,E]` action | `[N]` |

For a training batch, prepend `B` to all state-derived outputs as documented above.
