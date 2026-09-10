from pathlib import Path

import nash_drl.environment.sumo_training as st
from nash_drl.config import load_yaml
from nash_drl.environment.sumo import SumoScenario, SumoTripResult
from nash_drl.training.runner import TrainingRunner


class FakeSession:
    def __init__(self, scenario, config, *, use_gui=False, label="test"):
        self.scenario = scenario
        self.config = config
        self.t = 0.0
        self.closed = False

    def start(self):
        pass

    def current_time(self):
        return self.t

    def snapshot_edge_flow(self, edge_count):
        import torch
        return torch.zeros(edge_count, dtype=torch.float32)

    def execute_routes(self, routes, *, step_index, max_sim_steps=None):
        self.t += 1.0
        return SumoTripResult(self.t, [
            {"vehicle_id": vehicle_id, "travel_time_s": 10.0, "waiting_time_s": 0.0, "time_loss_s": 0.0, "distance_m": 1000.0}
            for vehicle_id in routes
        ], [], None)

    def close(self):
        self.closed = True


def test_training_runner_executes_fixed_trip_sets_for_each_episode(tmp_path, monkeypatch):
    def fake_build_network(self, directory):
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        return SumoScenario(p, p / "network.net.xml", p / "routes.rou.xml", p / "simulation.sumocfg", {})

    def fake_run(problem, net_file, vehicle_routes, config, directory, *, use_gui=False):
        edge_by_id = {e.id: e for e in problem.graph.edges}
        vehicle_metrics = []
        edge_metrics = []
        for vehicle_id, route in vehicle_routes.items():
            distance = sum(edge_by_id[e].length_km for e in route)
            vehicle_metrics.append({
                "vehicle_id": vehicle_id,
                "travel_time_s": distance / 50.0 * 3600.0,
                "waiting_time_s": 0.0,
                "time_loss_s": 0.0,
                "distance_m": distance * 1000.0,
            })
        return SumoTripResult(1.0, vehicle_metrics, edge_metrics)

    monkeypatch.setattr(st.SumoScenarioBuilder, "build_network", fake_build_network)
    monkeypatch.setattr(st, "run_sumo_trip", fake_run)
    monkeypatch.setattr(st, "SumoTrafficSession", FakeSession)

    cfg = load_yaml("configs/experiments/small.yaml")
    cfg["training"]["episodes"] = 2
    cfg["training"]["save_checkpoints"] = False
    cfg["training"]["output_dir"] = str(tmp_path / "training")
    cfg["training"]["replay_warmup"] = 1
    cfg["training"]["replay_batch_size"] = 1
    cfg["training"]["target_update_interval"] = 2

    result = TrainingRunner(cfg).run()

    assert result["metadata"]["steps_per_episode"] == 2
    assert len(result["episodes"]) == 2
    assert all(row["steps"] == 2 for row in result["episodes"])
    assert (tmp_path / "training" / "episode_results.csv").exists()
    assert (tmp_path / "training" / "vehicle_results.csv").exists()
    assert (tmp_path / "training" / "edge_results.csv").exists()
    assert (tmp_path / "training" / "step_results.csv").exists()
