# Configuration Guide

Configuration is split by responsibility so an experiment can override only one part of the system.

## `configs/default.yaml`

Composition-level defaults. It identifies the active environment, network, training and experiment settings.

## `configs/network.yaml`

Expected fields include:

```yaml
network:
  agent_feature_dim: 6
  edge_feature_dim: 1
  hidden_dim: 32
  deep_set_dim: 64
  actor_hidden_layers: 4
  activation: silu
  actor_output_channels: 5
```

Exact dimensions must remain consistent with the selected feature encoder.

## `configs/environment.yaml`

Responsible for:

- charging price parameters
- congestion parameters
- travel-time parameters
- reward weights and budget penalty
- episode/trip settings

## `configs/training.yaml`

Responsible for:

- random seed
- device
- discount factor
- optimizer and learning rates
- rollout length
- gradient clipping
- target Critic synchronization interval
- checkpoint frequency

## Experiment overrides

`configs/experiments/small.yaml` and `large.yaml` are intended to override defaults without duplicating the complete configuration.

## Reproducibility rule

Every run should save the fully resolved configuration next to metrics and checkpoints.
