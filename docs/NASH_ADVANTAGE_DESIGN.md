# NashDRL Advantage and Value Design

## Tensor semantics

For a single state:

```text
Actor output parameters:
    μ, P11, P12, P22, Ψ : [N, E]

Executed action:
    u : [N, E]

Local advantage:
    A(x,u) : [N]

Critic:
    V(x) : [N]

TD target:
    y : [N]

TD error:
    δ : [N]
```

For a batch of `B` states, prepend `B` to the relevant dimensions:

```text
μ, P11, P12, P22, Ψ : [B,N,E]
u                  : [B,N,E]
A                  : [B,N]
V                  : [B,N]
y                  : [B,N]
δ                  : [B,N]
```

The final losses are scalar means/sums over active agents. A scalar loss is an optimization
reduction and does not change the underlying per-agent semantics of `A`, `V`, or `δ`.

## LQ advantage terms

With `z_i = u_i - μ_i`, the implementation evaluates:

```text
A_i = -||z_i||²_P11
      - Σ_{j≠i} <z_i,z_j>_P12
      - Σ_{j≠i} ||z_j||²_P22
      + Σ_{j≠i} z_jᵀ Ψ_i
```

The `P22` term is evaluated by summing the **individual** squared rival deviations. The implementation
must not square the sum of rival deviations, because that creates an additional pairwise cross-rival term.

## Training decomposition

The implementation uses:

```text
Q(x,u) = V(x) + A(x,u)

TD target = r + γ Vslow(x')

Critic residual = V(x) + stop_gradient(A(x,u)) - stop_gradient(TD target)

Actor residual = stop_gradient(V(x)) + A(x,u) - stop_gradient(TD target)
```

Therefore the scalar Actor and Critic loss values may be equal for the same transition/batch, while
only the intended parameter subset receives gradients in each update.
