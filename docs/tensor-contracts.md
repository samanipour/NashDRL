# Tensor Contracts

The simulation stage does not train these tensors, but the interfaces are retained for the later Nash-DRL implementation.

| Tensor | Shape | Meaning |
|---|---|---|
| `agent_features` | `[N,F]` | semantic vehicle features |
| `edge_flow` | `[E]` | current flow per edge |
| `invariant` | `[N,N-1,F]` | rival features per focal vehicle |
| `non_invariant` | `[N,F+E]` | ego features + global edge flow |
| `action.edge_weights` | `[N,E]` | edge weight for each vehicle/edge |
| Actor output | `[5,N,E]` | `μ,P11,P12,P22,Ψ` |
| Critic value | `[N]` | per-agent baseline value |

The research document defines the CSR map as `row_ptr [V+1]`, `col_idx [E]`, `edge_ids [E]`, and parallel edge-property arrays. It defines the action as `[N,E]` edge weights subsequently mapped into routes. The neural design uses permutation-invariant rival aggregation and a non-invariant ego/global stream.

## Simulation boundary

The SUMO trace is not a replacement for the neural tensor contract. It is an external execution/measurement source. When DRL is introduced, the environment should convert its current semantic state into these tensors through `features/extractor.py`.
