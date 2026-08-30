from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator
from nash_drl.environment.sumo import SumoConfig, SumoScenarioBuilder


def test_sumo_scenario_builder_creates_inputs(tmp_path, monkeypatch):
    cfg = MockDataConfig(seed=2, num_vehicles=2, graph_nodes=5, graph_extra_edges=1)
    problem = MockDatasetGenerator(cfg).generate()
    monkeypatch.setattr("nash_drl.environment.sumo.find_sumo_binary", lambda name="netconvert": "true")
    # Avoid depending on XML net conversion in this unit test; verify graph/route generation via direct helper call.
    builder = SumoScenarioBuilder(problem, SumoConfig())
    builder._write_nodes(tmp_path / "n.nod.xml")
    builder._write_edges(tmp_path / "e.edg.xml")
    assert (tmp_path / "n.nod.xml").exists()
    assert (tmp_path / "e.edg.xml").exists()


def test_sumo_routes_are_contiguous_for_multitrip_mock_data():
    cfg = MockDataConfig(seed=42, num_vehicles=6, graph_nodes=9, graph_extra_edges=5, max_trips_per_vehicle=2)
    problem = MockDatasetGenerator(cfg).generate()
    builder = SumoScenarioBuilder(problem, SumoConfig(allow_turnarounds=True))
    routes = builder._build_routes()
    builder._validate_routes(routes)

    edges = {edge.id: edge for edge in problem.graph.edges}
    for route in routes.values():
        for a, b in zip(route, route[1:]):
            assert edges[a].destination == edges[b].source


def test_netconvert_allows_turnarounds_when_configured(tmp_path, monkeypatch):
    cfg = MockDataConfig(seed=42, num_vehicles=2, graph_nodes=5, graph_extra_edges=1)
    problem = MockDatasetGenerator(cfg).generate()
    captured = {}

    monkeypatch.setattr("nash_drl.environment.sumo.find_sumo_binary", lambda name="netconvert": "netconvert")
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()
    monkeypatch.setattr("nash_drl.environment.sumo.subprocess.run", fake_run)

    builder = SumoScenarioBuilder(problem, SumoConfig(allow_turnarounds=True))
    builder._run_netconvert(tmp_path / "n.nod.xml", tmp_path / "e.edg.xml", tmp_path / "network.net.xml")
    assert ["--no-turnarounds", "false"] == captured["cmd"][-4:-2]
    assert ["--no-turnarounds.geometry", "false"] == captured["cmd"][-2:]
