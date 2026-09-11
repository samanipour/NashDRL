from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nash_drl.data.mock_dataset import MockDataConfig, MockDatasetGenerator, load_problem
from nash_drl.environment.sumo import SumoConfig, SumoScenarioBuilder, run_sumo
from nash_drl.domain.charging import ChargingStationModel
from nash_drl.utils.seeding import seed_everything


@dataclass(slots=True)
class SimulationRunResult:
    dataset_path: Path
    scenario_dir: Path
    analytic_report: Path | None
    raw_trace: Path
    metadata: dict[str, Any]


class SimulationRunner:
    """End-to-end simulation runner for mock and real datasets; no DRL is used."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self) -> SimulationRunResult:
        sim_cfg = self.config.get("simulation", {})
        mode = str(sim_cfg.get("mode", "mock")).lower()
        output_dir = Path(sim_cfg.get("output_dir", "outputs/demo"))
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = int(self.config.get("project", {}).get("seed", 42))
        seed_everything(seed)

        if mode == "mock":
            dataset_path = self._generate_mock_dataset(seed, output_dir)
        elif mode == "real":
            dataset_path = Path(sim_cfg["dataset_path"])
            if not dataset_path.exists():
                raise FileNotFoundError(f"Real dataset not found: {dataset_path}")
        else:
            raise ValueError("simulation.mode must be 'mock' or 'real'")

        problem, payload = load_problem(dataset_path)
        sumo_cfg = SumoConfig(**sim_cfg.get("sumo", {}))
        scenario_dir = output_dir / "sumo"
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)
        scenario = SumoScenarioBuilder(problem, sumo_cfg).build(scenario_dir)

        raw_trace = output_dir / "simulation_trace.csv"
        result = run_sumo(
            scenario,
            sumo_cfg,
            use_gui=bool(sim_cfg.get("visualization", False)),
            output_csv=raw_trace,
        )

        # Attach reproducibility/configuration values to system observations.
        env_cfg = self.config.get("environment", {})
        from nash_drl.environment.reward import build_reward_config
        common_reward_cfg = build_reward_config(self.config, "common")
        for row in result["system_rows"]:
            row.update({
                "simulation_mode": mode,
                "simulation_seed": seed,
                "sumo_step_length_s": sumo_cfg.step_length_s,
                "sumo_end_time_s": sumo_cfg.end_time_s,
                "sumo_teleport_time_s": sumo_cfg.teleport_time_s,
                "energy_rate_kwh_per_km": env_cfg.get("energy_rate_kwh_per_km", 1.0),
                "charging_overhead": env_cfg.get("charging_overhead", 6.0),
                "charging_fixed_cost": env_cfg.get("charging_fixed_cost", 1.0),
                "charging_floor_price": env_cfg.get("charging_floor_price", 0.0),
                "congestion_alpha": env_cfg.get("congestion_alpha", 0.15),
                "congestion_beta": env_cfg.get("congestion_beta", 4.0),
                "reward_travel_time_weight": common_reward_cfg.travel_time_weight,
                "reward_charging_cost_weight": common_reward_cfg.charging_cost_weight,
                "reward_budget_penalty": common_reward_cfg.budget_penalty,
            })

        report_path = None
        if bool(sim_cfg.get("generate_analytic_report", True)):
            report_path = self._write_analytic_report(output_dir, problem, payload, result)

        metadata = {
            "mode": mode,
            "seed": seed,
            "num_vehicles": len(problem.vehicles),
            "num_nodes": problem.graph.num_nodes,
            "num_edges": problem.graph.num_edges,
            "steps": result["steps"],
            "visualization": bool(sim_cfg.get("visualization", False)),
            "analytic_report": str(report_path) if report_path else None,
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if bool(sim_cfg.get("salabim", {}).get("enabled", True)) and bool(sim_cfg.get("visualization", False)):
            try:
                from nash_drl.visualization.salabim_replay import SalabimReplay
                SalabimReplay.from_trace(raw_trace).show()
            except Exception as exc:  # visualizer must never invalidate a completed SUMO run
                (output_dir / "salabim_warning.txt").write_text(str(exc), encoding="utf-8")

        return SimulationRunResult(dataset_path, scenario_dir, report_path, raw_trace, metadata)

    def _generate_mock_dataset(self, seed: int, output_dir: Path) -> Path:
        mock = dict(self.config.get("mock_data", {}))
        mock.setdefault("seed", seed)
        cfg = MockDataConfig(**mock)
        dataset_dir = Path("datasets/generated") / f"mock_seed_{cfg.seed}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "dataset.json"
        problem = MockDatasetGenerator(cfg).generate()
        MockDatasetGenerator(cfg).save(problem, dataset_path)
        # Also copy the generated dataset into the run directory for provenance.
        shutil.copy2(dataset_path, output_dir / "dataset.json")
        return dataset_path

    def _write_analytic_report(self, output_dir: Path, problem: Any, payload: dict[str, Any], result: dict[str, Any]) -> Path:
        report_path = output_dir / "analytical_report.csv"
        rows: list[dict[str, Any]] = []
        vehicles_by_id = {str(v.id): v for v in problem.vehicles}
        edges_by_id = {f"e{e.id}": e for e in problem.graph.edges}
        config = payload.get("config", {})
        energy_rate = float(config.get("energy_rate_kwh_per_km", 1.0))
        charging = ChargingStationModel(
            overhead=float(config.get("charging_overhead", 6.0)),
            fixed_unit_cost=float(config.get("charging_fixed_cost", 1.0)),
            floor_price=float(config.get("charging_floor_price", 0.0)),
        )
        edge_index = {(float(r["sim_time_s"]), r["entity_id"]): r for r in result["edge_rows"]}
        last_distance: dict[str, float] = {}
        cumulative_cost: dict[str, float] = {}
        for row in result["vehicle_rows"]:
            vid = row["entity_id"]
            vid_num = str(vid).lstrip("v")
            v = vehicles_by_id.get(vid_num)
            base = {**row}
            road_id = row.get("road_id", "")
            edge_telemetry = edge_index.get((float(row["sim_time_s"]), road_id))
            flow = float(edge_telemetry.get("vehicle_count", 0.0)) if edge_telemetry else 0.0
            price = charging.unit_price(flow)
            distance_m = float(row["distance_m"])
            prev_distance = last_distance.get(vid, distance_m)
            delta_km = max(0.0, distance_m - prev_distance) / 1000.0
            increment = delta_km * energy_rate * price
            cumulative_cost[vid] = cumulative_cost.get(vid, 0.0) + increment
            last_distance[vid] = distance_m
            base.update(
                edge_flow=flow,
                charging_unit_price=price,
                energy_rate_kwh_per_km=energy_rate,
                incremental_energy_kwh=delta_km * energy_rate,
                cumulative_energy_kwh=(distance_m / 1000.0) * energy_rate,
                incremental_charging_cost=increment,
                cumulative_charging_cost=cumulative_cost[vid],
            )
            if v is not None:
                base.update(
                    vehicle_id=v.id,
                    free_flow_speed_kmh=v.free_flow_speed_kmh,
                    budget=v.budget,
                    estimated_remaining_budget=v.budget - cumulative_cost[vid],
                    estimated_budget_violation=(cumulative_cost[vid] > v.budget),
                    remaining_trip_count=v.remaining_trip_count,
                )
            rows.append(base)
        for row in result["edge_rows"]:
            edge = edges_by_id.get(row["entity_id"])
            base = {**row}
            flow = float(row.get("vehicle_count", 0.0))
            if edge is not None:
                price = charging.unit_price(flow)
                base.update(
                    edge_id=edge.id,
                    from_node=edge.source,
                    to_node=edge.destination,
                    length_km=edge.length_km,
                    capacity_vph=edge.capacity_vph,
                    charging_unit_price=price,
                    energy_capacity_kwh_per_step=flow * edge.length_km * energy_rate,
                    congestion_ratio=flow / max(edge.capacity_vph, 1e-9),
                )
            rows.append(base)
        # Trips are first-class dataset entities; include static route planning information in the report.
        trip_rows: list[dict[str, Any]] = []
        for vehicle in problem.vehicles:
            for trip_index, trip in enumerate(vehicle.trips):
                trip_rows.append({
                    "entity_type": "trip",
                    "entity_id": f"v{vehicle.id}_trip{trip_index}",
                    "vehicle_id": vehicle.id,
                    "trip_index": trip_index,
                    "origin_node": trip.origin,
                    "destination_node": trip.destination,
                })
        rows.extend(trip_rows)
        rows.extend(result["system_rows"])
        keys = sorted({k for r in rows for k in r})
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
        return report_path
