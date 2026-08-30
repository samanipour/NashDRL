# Dataset Layout

`generated/` contains reproducibly generated mock datasets. `maps/` and `scenarios/` are reserved for externally supplied real datasets.

A simulation dataset is a JSON document with `schema_version`, `graph`, and `vehicles`. Mock generation also records the generator configuration and seed.

Do not place trained checkpoints in `datasets/`; use `checkpoints/` for learned model artifacts and `outputs/` for run-specific results.
