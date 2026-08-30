# Configuration

The main simulation controls live under `simulation` and `mock_data`.

| Key | Meaning |
|---|---|
| `simulation.mode` | `mock` or `real` |
| `simulation.visualization` | use `sumo-gui` when true |
| `simulation.generate_analytic_report` | write consolidated CSV when true |
| `simulation.output_dir` | run output directory |
| `simulation.dataset_path` | real-mode JSON dataset |
| `simulation.sumo.binary` | `auto`, `sumo`, `sumo-gui`, or explicit path |
| `mock_data.seed` | deterministic generator seed |
| `mock_data.num_vehicles` | number of vehicles |
| `mock_data.min_trips_per_vehicle` | lower trip count bound |
| `mock_data.max_trips_per_vehicle` | upper trip count bound |
| `mock_data.graph_nodes` | graph node count |
| `mock_data.graph_extra_edges` | additional directed edges |
| `mock_data.road_min_length_km` | minimum edge length |
| `mock_data.road_max_length_km` | maximum edge length |
| `mock_data.capacity_min_vph` | minimum edge capacity |
| `mock_data.capacity_max_vph` | maximum edge capacity |
| `mock_data.speed_min_kmh` | minimum vehicle free-flow speed |
| `mock_data.speed_max_kmh` | maximum vehicle free-flow speed |
| `mock_data.budget_min` | minimum vehicle budget |
| `mock_data.budget_max` | maximum vehicle budget |

The remaining `environment` and `reward` sections define the mathematical environment and remain useful when comparing SUMO observations with the analytical equations from the research model.
