import json
from pathlib import Path

from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem


def test_mock_dataset_is_reproducible_and_budget_feasible(tmp_path: Path):
    cfg = MockDataConfig(
        seed=123,
        num_vehicles=10,
        min_trips_per_vehicle=1,
        max_trips_per_vehicle=4,
        graph_nodes=12,
        graph_extra_edges=10,
        budget_min=50,
        budget_max=500,
    )
    a = MockDatasetGenerator(cfg).generate()
    b = MockDatasetGenerator(cfg).generate()

    assert [v.trips for v in a.vehicles] == [v.trips for v in b.vehicles]
    assert [v.budget for v in a.vehicles] == [v.budget for v in b.vehicles]
    assert [e.length_km for e in a.graph.edges] == [e.length_km for e in b.graph.edges]
    assert all(v.budget > 0 for v in a.vehicles)

    path = tmp_path / "dataset.json"
    MockDatasetGenerator(cfg).save(a, path)
    _, payload = load_problem(path)
    assert payload["schema_version"] == 3
