# Dynamic SUMO Traffic Flow in the NashDRL State

## Purpose

The road-network topology is static during an episode and is therefore not encoded as an input tensor to the Actor or Critic. Dynamic traffic flow is a state variable and is supplied by SUMO/TraCI at each decision time.

## State timing contract

For decision time `t`:

```text
SUMO flow snapshot at t
        ↓
GlobalState.edge_flow(t) [E]
        ↓
StateFeatureExtractor
        ↓
Non-invariant input [N,F+E]
        ↓
Actor / Critic
        ↓
Action and route decisions
        ↓
SUMO advances selected trip legs
        ↓
SUMO flow snapshot at t+1
        ↓
GlobalState.edge_flow(t+1) [E]
```

The environment stores `flow_time_s` with the state for traceability and `edge_flow_source="sumo"` for the live SUMO training environment.

## Persistent TraCI session

`NashSUMOTrainingEnvironment` starts one `SumoTrafficSession` when an episode is reset. The same SUMO/TraCI process remains alive for all steps in that episode. This is required so that traffic present at the end of one RL step can still be present at the beginning of the next step.

The relevant implementation is:

- `src/nash_drl/environment/sumo.py::SumoTrafficSession`
- `src/nash_drl/environment/sumo_training.py::NashSUMOTrainingEnvironment.reset()`
- `src/nash_drl/environment/sumo_training.py::NashSUMOTrainingEnvironment.step()`

## Flow provenance

`SumoTrafficSession.snapshot_edge_flow(E)` queries `traci.edge.getLastStepVehicleNumber()` for every normal network edge and returns the resulting `[E]` tensor, indexed by the project's numeric edge ID (`e0`, `e1`, ...).

The planned paths for the current controlled vehicles are **not** used to overwrite the next state flow. Instead, the next state is populated from the actual post-step SUMO snapshot.

For the mathematical reward calculation, the current observed flow and the current controlled route contribution are combined to form the effective flow used by charging and congestion calculations. The post-step SUMO snapshot remains the state input for the next decision.

## Static topology vs dynamic state

The CSR graph remains available inside `GlobalState` because the deterministic route mapper needs graph connectivity. It is not included in `NetworkInputs`:

```text
GlobalState
├── csr_map          → routing/environment only
├── agent_features   → neural input
└── edge_flow        → neural input
```

Consequently:

```text
Topology: static / routing support / not NN input
Traffic flow: dynamic / SUMO observation / NN input
```

## Initial state

At episode reset, SUMO is started with the episode network and an initially empty controlled-vehicle route file. The environment immediately reads the first SUMO edge-flow snapshot. This becomes `GlobalState.edge_flow` for `t=0`.

## Background traffic extension

The persistent-session design intentionally leaves the SUMO world alive between NashDRL decisions. This makes it possible to add future background traffic or dynamically arriving trip requests without changing the state contract: the next Actor/Critic observation simply reads the current SUMO edge-flow vector.
