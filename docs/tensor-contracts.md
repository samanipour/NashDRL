# Tensor Contracts

| Tensor / object | Single-state shape | Batched shape | Meaning |
|---|---|---|---|
| `agent_features` | `[N,F]` | `[B,N,F]` | per-vehicle semantic features |
| `edge_flow` | `[E]` | `[B,E]` | current flow on every edge |
| `invariant` | `[N,N-1,F]` | `[B,N,N-1,F]` | rival features for every focal vehicle |
| `non_invariant` | `[N,F+E]` | `[B,N,F+E]` | focal/self features + global edge flows |
| Actor `mu` | `[N,E]` | `[B,N,E]` | mean/raw edge action weights |
| Actor `p11` | `[N,E]` | `[B,N,E]` | strictly-positive ego curvature |
| Actor `p12` | `[N,E]` | `[B,N,E]` | interaction coefficient |
| Actor `p22` | `[N,E]` | `[B,N,E]` | strictly-positive rival curvature |
| Actor `psi` | `[N,E]` | `[B,N,E]` | linear tilt coefficient |
| Actor `as_tensor()` | `[5,N,E]` | `[B,5,N,E]` | stacked LQ parameter channels |
| Critic value | `[N]` | `[B,N]` | baseline state value per agent |
| Action | `[N,E]` | `[B,N,E]` | edge weights given to route mapping |
| LQ Advantage | `[N]` | `[B,N]` | per-agent advantage |

## Ordering of Actor channels

`ActorOutput.as_tensor()` preserves the paper's channel ordering:

```text
0 = μ
1 = P11
2 = P12
3 = P22
4 = Ψ
```

The document specifies the Actor output as `[5,N,E]` and identifies these five channels. fileciteturn1file0L130-L135

## Permutation invariance

For a fixed focal agent, permuting the `N-1` rival rows in the invariant input must not change the Actor or Critic outputs. The invariant stream achieves this through the shared `phi` embedding and sum aggregation.

## Runtime validation

The models validate rank, feature dimension, rival count (`N-1`), and edge-feature width before computation. This catches shape-contract violations near the boundary rather than later inside matrix multiplications.
