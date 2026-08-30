# Module Guide

## Runtime modules

### `domain/`
Pure research entities: graph, edge, vehicle, trip, charging model, and problem definition. No SUMO or PyTorch dependency should be required by domain classes.

### `data/`
Runtime representations and generated datasets. `CSRMap` implements the research CSR contract. `mock_dataset.py` creates and loads reproducible JSON datasets.

### `environment/`
Contains the analytical environment plus SUMO integration. `sumo.py` is the only module that should directly import TraCI. `simulation.py` orchestrates dataset → SUMO → telemetry → report.

### `routing/`
Converts edge weights into valid edge sequences. In the non-learning demo, `DijkstraMapper` is the deterministic baseline.

### `features/`
Transforms the environment state into the neural tensor streams retained for future DRL development.

### `models/` and `game/`
Reserved for the later learning stage. They are not invoked by the simulation demo.

### `evaluation/`
Metrics and report formatting. The SUMO simulation currently writes richer CSV analytics directly from `environment/simulation.py` so that every run is self-contained.

### `visualization/`
SUMO GUI is the live traffic visualization. `salabim_replay.py` provides a post-run Python trajectory replay from the saved trace.

### `utils/`
Cross-cutting deterministic seeding, logging, device and serialization helpers.

## Dependency rule

Prefer this direction:

```text
domain ← data ← features/models
   ↑       ↑       ↑
routing  environment  training
                 ↑
              scripts/CLI
```

Direct TraCI imports belong in `environment/sumo.py`; visualization packages should consume traces rather than controlling the traffic simulation.
