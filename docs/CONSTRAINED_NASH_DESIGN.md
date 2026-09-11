# Constrained NashDRL — v0.9.0

## Purpose

This revision strengthens the NashDRL implementation for the hard vehicle-budget constraint while keeping the Section-4 five-channel Actor contract.

## LQ correction

For each focal agent i, the rival term is:

`- sum_{j != i} ||z_j||^2_{P22_i}`

The implementation now computes the sum of individual rival squared deviations. It does not compute the squared norm of their sum.

## Stable interaction curvature

`P11` and `P22` remain strictly positive. `P12` is parameterized as `2*rho*P11*tanh(raw_P12)`, with `rho in (0,1)`, so `|P12| < 2*rho*P11`.

## Hard-budget feasibility

`BudgetAwareDijkstraMapper` is the deterministic feasibility layer. It receives the live SUMO edge-flow state and remaining budgets, evaluates the Actor-preferred route, and if necessary searches for a route whose estimated charging cost is within the current remaining budget. Other controlled routes are included in the flow estimate.

This changes the practical feasible action set: a budget-infeasible route is repaired before SUMO whenever a feasible route exists. If none exists, the residual infeasibility is reported.

## Training

Nash training defaults to on-policy updates (`replay_enabled: false`) so the LQ local-equilibrium fit does not mix stale replayed actions from older policies. The paper's per-agent decomposition remains `Q_i = V_i + A_i`, with per-agent `[N]` / `[B,N]` advantage and value tensors and scalar reductions only at the final optimizer loss.
