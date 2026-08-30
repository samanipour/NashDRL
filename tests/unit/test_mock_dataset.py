from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator


def _signature(problem):
    return (
        [(e.source, e.destination, round(e.length_km, 6), round(e.capacity_vph, 6)) for e in problem.graph.edges],
        [(v.id, round(v.free_flow_speed_kmh, 6), round(v.budget, 6), [(t.origin, t.destination) for t in v.trips]) for v in problem.vehicles],
    )


def test_mock_generation_is_reproducible():
    cfg = MockDataConfig(seed=7, num_vehicles=5, graph_nodes=8, graph_extra_edges=4)
    a = MockDatasetGenerator(cfg).generate()
    b = MockDatasetGenerator(cfg).generate()
    assert _signature(a) == _signature(b)


def test_trip_counts_are_within_configured_bounds():
    cfg = MockDataConfig(seed=9, num_vehicles=20, min_trips_per_vehicle=1, max_trips_per_vehicle=4)
    problem = MockDatasetGenerator(cfg).generate()
    assert all(cfg.min_trips_per_vehicle <= len(v.trips) <= cfg.max_trips_per_vehicle for v in problem.vehicles)
