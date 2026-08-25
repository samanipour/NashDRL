# Neural Network Design

## 1. Design goals

The network implementation follows the research document's requirements:

- parameterize the LQ advantage through the Actor;
- use permutation-invariant processing of other agents;
- estimate per-agent state value;
- use a frozen/slow Target Critic for stable TD targets;
- produce edge-level action weights for every vehicle.

## 2. Deep Sets encoder

Input:

```text
[N,N-1,F]
```

Processing:

```text
shared φ applied to every rival
            ↓
element-wise sum aggregation
            ↓
[N,D]
```

The embedding weights are shared across rival vehicles.

## 3. Actor

Two information streams are combined:

```text
Stream 1: [N,N-1,F] → Deep Sets → crowd representation
Stream 2: [N,F+E]   → ego/global representation
```

After concatenation, the document specifies a main fully connected trunk with four hidden layers of 32 nodes and SiLU activation.

The output is:

```text
[5,N,E]
```

corresponding to:

```text
μ, P11, P12, P22, Ψ
```

`P11` and `P22` must be transformed so that the required positivity/concavity conditions are respected by the implementation.

## 4. Critic

The Critic mirrors the permutation-invariant processing but produces:

```text
V(x): [N]
```

The value trunk uses four fully connected hidden layers of 32 nodes with SiLU activation.

## 5. Target Critic

The Target Critic is an architectural copy of the Critic.

It should:

- have the same input/output contract;
- have gradients disabled during normal training;
- be synchronized from the main Critic according to the configured target-update strategy.

## 6. LQ advantage

The Actor output parameterizes the LQ advantage calculation. For each agent:

```text
z_i = u_i - μ_i
```

The implementation of the complete advantage equation belongs in `models/lq_advantage.py`, while higher-level game/equilibrium logic belongs in `game/`.

## 7. Network API

Recommended interfaces:

```python
class ActorNetwork(nn.Module):
    def forward(self, inputs: NetworkInputs) -> ActorOutput: ...

class CriticNetwork(nn.Module):
    def forward(self, inputs: NetworkInputs) -> Tensor: ...

class DeepSetEncoder(nn.Module):
    def forward(self, other_agents: Tensor) -> Tensor: ...

class LQAdvantage(nn.Module):
    def forward(self, actor_output: ActorOutput, action: Tensor) -> Tensor: ...
```

Shape checks should run in tests and may optionally be enabled at runtime in debug mode.
