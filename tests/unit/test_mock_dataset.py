from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator


def test_mock_dataset_is_reproducible(tmp_path):
    cfg = MockDataConfig(seed=7, num_vehicles=5, graph_nodes=8, graph_extra_edges=3)
    gen1 = MockDatasetGenerator(cfg); gen2 = MockDatasetGenerator(cfg)
    p1 = gen1.generate(); p2 = gen2.generate()
    assert p1.graph == p2.graph
    assert [(v.free_flow_speed_kmh, v.budget, v.trips) for v in p1.vehicles] == [
        (v.free_flow_speed_kmh, v.budget, v.trips) for v in p2.vehicles
    ]
    out = gen1.save(p1, tmp_path / "dataset.json")
    assert out.exists()
