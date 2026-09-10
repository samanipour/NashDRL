

## NashDRL advantage/value tensor contract

For a single global state with `N` vehicles and `E` edges:

- Actor parameters: `mu`, `p11`, `p12`, `p22`, `psi` -> `[N,E]`.
- Executed action `u` -> `[N,E]`.
- LQ advantage `A(x,u)` -> `[N]`, one local advantage per focal agent.
- Critic value `V(x)` -> `[N]`, one baseline value per agent.
- TD target -> `[N]`.
- TD error -> `[N]`.

For a batch of size `B`, the corresponding tensors are `[B,N,E]` for edge-wise quantities and `[B,N]` for per-agent quantities. The final training losses are scalar reductions over active agents. The scalar loss does not imply that the underlying Nash advantage or value is scalar.

The LQ rival quadratic term is implemented as `-sum_{j != i} ||z_j||^2_{P22,i}` rather than `-||sum_{j != i} z_j||^2_{P22,i}`.
