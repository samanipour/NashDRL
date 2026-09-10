from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET


from nash_drl.data.mock_dataset import load_problem
from nash_drl.domain import ProblemDefinition
from nash_drl.routing import DijkstraMapper
from nash_drl.data import Action, GlobalState


@dataclass(frozen=True, slots=True)
class SumoConfig:
    binary: str = "auto"
    step_length_s: float = 1.0
    end_time_s: float = 300.0
    teleport_time_s: int = 120
    seed: int = 42
    default_edge_speed_kmh: float = 50.0
    vehicle_departure_gap_s: float = 2.0
    allow_turnarounds: bool = True


@dataclass(slots=True)
class SumoScenario:
    directory: Path
    net_file: Path
    route_file: Path
    config_file: Path
    vehicle_routes: dict[int, list[int]]


class SumoScenarioBuilder:
    """Build a self-contained SUMO network, routes, and configuration."""

    def __init__(self, problem: ProblemDefinition, config: SumoConfig) -> None:
        self.problem = problem
        self.config = config

    def build_network(self, directory: str | Path) -> SumoScenario:
        """Build only the reusable SUMO network for multi-step training."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        node_file = out / "network.nod.xml"
        edge_file = out / "network.edg.xml"
        net_file = out / "network.net.xml"
        self._write_nodes(node_file)
        self._write_edges(edge_file)
        self._run_netconvert(node_file, edge_file, net_file)
        route_file = out / "routes.empty.rou.xml"
        config_file = out / "simulation.empty.sumocfg"
        self._write_routes(route_file, {})
        self._write_config(config_file, net_file, route_file)
        return SumoScenario(out, net_file, route_file, config_file, {})

    def build(self, directory: str | Path) -> SumoScenario:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        node_file = out / "network.nod.xml"
        edge_file = out / "network.edg.xml"
        net_file = out / "network.net.xml"
        route_file = out / "routes.rou.xml"
        config_file = out / "simulation.sumocfg"

        self._write_nodes(node_file)
        self._write_edges(edge_file)
        self._run_netconvert(node_file, edge_file, net_file)
        routes = self._build_routes()
        self._validate_routes(routes)
        self._write_routes(route_file, routes)
        self._write_config(config_file, net_file, route_file)
        return SumoScenario(out, net_file, route_file, config_file, routes)

    def _write_nodes(self, path: Path) -> None:
        root = ET.Element("nodes")
        n = self.problem.graph.num_nodes
        side = max(1, int(n**0.5))
        for node_id in range(n):
            x = float(node_id % side) * 100.0
            y = float(node_id // side) * 100.0
            ET.SubElement(root, "node", id=f"n{node_id}", x=str(x), y=str(y), type="priority")
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def _write_edges(self, path: Path) -> None:
        root = ET.Element("edges")
        for e in self.problem.graph.edges:
            ET.SubElement(
                root,
                "edge",
                id=f"e{e.id}",
                **{
                    "from": f"n{e.source}",
                    "to": f"n{e.destination}",
                    "numLanes": "1",
                    "speed": str(max(5.0, self.config.default_edge_speed_kmh / 3.6)),
                },
            )
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def _run_netconvert(self, node_file: Path, edge_file: Path, net_file: Path) -> None:
        binary = find_sumo_binary("netconvert")
        cmd = [binary, "-n", str(node_file), "-e", str(edge_file), "-o", str(net_file)]
        # Multi-trip vehicles may legitimately finish one trip and immediately
        # start the next trip in the opposite direction.  SUMO treats that as
        # a turnaround movement.  The previous implementation unconditionally
        # disabled turnarounds, making valid concatenated mock routes invalid
        # at the SUMO network level.
        if self.config.allow_turnarounds:
            cmd.extend(["--no-turnarounds", "false", "--no-turnarounds.geometry", "false"])
        else:
            cmd.extend(["--no-turnarounds", "true", "--no-turnarounds.geometry", "true"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"netconvert failed:\n{proc.stdout}\n{proc.stderr}")

    def _build_routes(self) -> dict[int, list[int]]:
        # Use a temporary zero-flow GlobalState-like structure to reuse the documented deterministic mapper.
        from nash_drl.data import CSRMap
        import torch
        csr = CSRMap.from_graph(self.problem.graph)
        vehicle_routes: dict[int, list[int]] = {}
        edge_weights = torch.zeros((len(self.problem.vehicles), self.problem.graph.num_edges), dtype=torch.float32)
        for edge in self.problem.graph.edges:
            edge_weights[:, edge.id] = 1.0 / max(edge.length_km, 1e-6)
        state = GlobalState(
            csr_map=csr,
            agent_features=torch.zeros((len(self.problem.vehicles), 6)),
            edge_flow=csr.edge_flow,
            current_nodes=torch.tensor([v.trips[0].origin for v in self.problem.vehicles], dtype=torch.long),
            next_destinations=torch.tensor([v.trips[0].destination for v in self.problem.vehicles], dtype=torch.long),
            final_destinations=torch.tensor([v.final_destination for v in self.problem.vehicles], dtype=torch.long),
        )
        mapper = DijkstraMapper()
        # Concatenate shortest path for each trip; each trip starts where the previous ends in generated data.
        for idx, vehicle in enumerate(self.problem.vehicles):
            all_edges: list[int] = []
            current = state.current_nodes[idx]
            for trip in vehicle.trips:
                state.current_nodes[idx] = int(current)
                state.next_destinations[idx] = int(trip.destination)
                action = Action(edge_weights[idx : idx + 1].clone())
                sub_state = GlobalState(
                    csr_map=csr,
                    agent_features=torch.zeros((1, 6)),
                    edge_flow=csr.edge_flow,
                    current_nodes=torch.tensor([int(current)]),
                    next_destinations=torch.tensor([trip.destination]),
                    final_destinations=torch.tensor([vehicle.final_destination]),
                )
                mapped = mapper.map(action, sub_state).edge_ids[0]
                if not mapped and current != trip.destination:
                    raise ValueError(f"No route found for vehicle {vehicle.id}: {current}->{trip.destination}")
                all_edges.extend(mapped)
                current = trip.destination
            vehicle_routes[vehicle.id] = all_edges
        return vehicle_routes


    def _validate_routes(self, routes: dict[int, list[int]]) -> None:
        """Validate route edge continuity before handing routes to SUMO."""
        edges_by_id = {edge.id: edge for edge in self.problem.graph.edges}
        for vehicle_id, route in routes.items():
            for previous_id, current_id in zip(route, route[1:]):
                previous = edges_by_id.get(previous_id)
                current = edges_by_id.get(current_id)
                if previous is None or current is None:
                    raise ValueError(
                        f"Vehicle {vehicle_id} route references unknown edge: "
                        f"{previous_id if previous is None else current_id}"
                    )
                if previous.destination != current.source:
                    raise ValueError(
                        f"Vehicle {vehicle_id} route is disconnected between "
                        f"e{previous.id} ({previous.source}->{previous.destination}) and "
                        f"e{current.id} ({current.source}->{current.destination})"
                    )

    def _write_routes(self, path: Path, routes: dict[int, list[int]]) -> None:
        root = ET.Element("routes")
        ET.SubElement(root, "vType", id="eav", accel="2.0", decel="4.5", sigma="0.5", length="5.0", maxSpeed="33.3")
        gap = 0.0
        for vehicle in self.problem.vehicles:
            edge_ids = routes.get(vehicle.id, [])
            if not edge_ids:
                continue
            ET.SubElement(root, "route", id=f"r{vehicle.id}", edges=" ".join(f"e{eid}" for eid in edge_ids))
            ET.SubElement(
                root,
                "vehicle",
                id=f"v{vehicle.id}",
                type="eav",
                route=f"r{vehicle.id}",
                depart=str(gap),
                departLane="best",
                departSpeed="max",
            )
            gap += self.config.vehicle_departure_gap_s
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def _write_config(self, path: Path, net_file: Path, route_file: Path) -> None:
        root = ET.Element("configuration")
    
        inp = ET.SubElement(root, "input")
    
        # SUMO resolves file references relative to the .sumocfg file.
        # The network is built once per episode and reused by every
        # trip-level step.
        net_ref = os.path.relpath(net_file, path.parent)
        route_ref = os.path.relpath(route_file, path.parent)
    
        ET.SubElement(inp, "net-file", value=net_ref)
        ET.SubElement(inp, "route-files", value=route_ref)
    
        time = ET.SubElement(root, "time")
        ET.SubElement(time, "begin", value="0")
        ET.SubElement(time, "end", value=str(self.config.end_time_s))
    
        misc = ET.SubElement(root, "processing")
        ET.SubElement(
            misc,
            "time-to-teleport",
            value=str(self.config.teleport_time_s),
        )
    
        report = ET.SubElement(root, "report")
        ET.SubElement(report, "verbose", value="false")
    
        ET.ElementTree(root).write(
            path,
            encoding="utf-8",
            xml_declaration=True,
        )


def find_sumo_binary(name: str = "sumo") -> str:
    candidates = [name, f"{name}.exe"]
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.extend([
            str(Path(sumo_home) / "bin" / name),
            str(Path(sumo_home) / "bin" / f"{name}.exe"),
        ])
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        p = Path(candidate)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find {name}. Install SUMO 1.27.1 and put its bin directory on PATH "
        "or set SUMO_HOME."
    )



class SumoTrafficSession:
    """Persistent SUMO/TraCI session for multi-step NashDRL episodes.

    The SUMO process stays alive for the whole episode.  At every RL step the
    current edge flow is read from SUMO before new controlled vehicles are
    inserted, and a second snapshot is taken after the selected trip legs are
    completed.  The latter becomes the traffic-flow component of the next
    ``GlobalState``.
    """

    def __init__(
        self,
        scenario: SumoScenario,
        config: SumoConfig,
        *,
        use_gui: bool = False,
        label: str = "nash_drl_episode",
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.use_gui = use_gui
        self.label = label
        self._traci = None
        self._closed = True
        self._edge_id_to_int: dict[str, int] = {}
        self._sim_step_count = 0

    def start(self) -> None:
        if not self._closed:
            return
        try:
            import traci
        except ImportError as exc:
            raise RuntimeError(
                "TraCI is required for SUMO training. Install traci==1.27.1."
            ) from exc

        binary_name = "sumo-gui" if self.use_gui else "sumo"
        if self.config.binary == "auto":
            binary = find_sumo_binary(binary_name)
        elif self.config.binary in {"sumo", "sumo-gui"}:
            binary = find_sumo_binary(self.config.binary)
        else:
            binary = str(Path(self.config.binary))
            if not Path(binary).exists():
                raise FileNotFoundError(f"Configured SUMO binary does not exist: {binary}")

        cmd = [
            binary,
            "-c", str(self.scenario.config_file),
            "--seed", str(self.config.seed),
            "--step-length", str(self.config.step_length_s),
            "--no-step-log",
        ]
        # Do not pass --start for GUI training: TraCI must receive control of
        # simulation advancement. simulationStep() starts/advances the GUI.
        traci.start(cmd, label=self.label)
        self._traci = traci.getConnection(self.label)
        self._closed = False
        self._sim_step_count = 0

    @property
    def traci(self):
        if self._closed or self._traci is None:
            raise RuntimeError("SUMO session is not started.")
        return self._traci

    def current_time(self) -> float:
        return float(self.traci.simulation.getTime())

    def snapshot_edge_flow(self, edge_count: int) -> Any:
        """Return SUMO's current last-step vehicle count as ``[E]`` tensor."""
        import torch

        flow = torch.zeros(edge_count, dtype=torch.float32)
        for edge_id in self.traci.edge.getIDList():
            if edge_id.startswith(":"):
                continue
            try:
                numeric_id = int(edge_id.lstrip("e"))
            except ValueError:
                continue
            if 0 <= numeric_id < edge_count:
                flow[numeric_id] = float(
                    self.traci.edge.getLastStepVehicleNumber(edge_id)
                )
        return flow

    def edge_metrics(self, step_index: int) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        sim_time = self.current_time()
        for eid in self.traci.edge.getIDList():
            if eid.startswith(":"):
                continue
            flow = int(self.traci.edge.getLastStepVehicleNumber(eid))
            metrics.append({
                "entity_type": "edge",
                "entity_id": eid,
                "step": step_index,
                "sim_time_s": sim_time,
                "traffic_flow": flow,
                "vehicle_count": flow,
                "mean_speed_mps": float(self.traci.edge.getLastStepMeanSpeed(eid)),
                "occupancy_pct": float(self.traci.edge.getLastStepOccupancy(eid)),
                "halting_number": int(self.traci.edge.getLastStepHaltingNumber(eid)),
                "sampled_travel_time_s": float(self.traci.edge.getTraveltime(eid)),
            })
        return metrics

    def _route_id(self, step_index: int, vehicle_id: int) -> str:
        return f"rl_r_{step_index}_{vehicle_id}"

    def _add_vehicle_for_route(self, vehicle_id: int, route: list[int], step_index: int) -> None:
        if not route:
            return
        traci = self.traci
        vid = f"v{vehicle_id}"
        rid = self._route_id(step_index, vehicle_id)
        edges = [f"e{edge_id}" for edge_id in route]
        traci.route.add(rid, edges)
        existing = set(traci.vehicle.getIDList())
        if vid in existing:
            # This is useful for real-data/background-traffic extensions. For
            # our controlled vehicles a completed leg normally removes the ID.
            traci.vehicle.setRoute(vid, edges)
        else:
            traci.vehicle.add(
                vid,
                rid,
                typeID="eav",
                depart="now",
                departLane="best",
                departSpeed="max",
            )

    def execute_routes(
        self,
        routes: dict[int, list[int]],
        *,
        step_index: int,
        max_sim_steps: int | None = None,
    ) -> SumoTripResult:
        """Execute current RL-step routes while keeping the SUMO session alive."""
        if not routes:
            return SumoTripResult(
                sumo_time_s=self.current_time(),
                vehicle_metrics=[],
                edge_metrics=self.edge_metrics(step_index),
                edge_flow=None,
            )

        traci = self.traci
        expected_ids = {f"v{vehicle_id}" for vehicle_id in routes if routes[vehicle_id]}
        for vehicle_id, route in routes.items():
            self._add_vehicle_for_route(vehicle_id, route, step_index)

        first_seen: dict[str, float] = {}
        last_observed: dict[str, dict[str, float]] = {}
        vehicle_metrics: list[dict[str, Any]] = []
        edge_metrics: list[dict[str, Any]] = []
        local_steps = 0
        limit = max_sim_steps or max(1, int(self.config.end_time_s / self.config.step_length_s))

        while expected_ids and local_steps < limit:
            traci.simulationStep()
            self._sim_step_count += 1
            local_steps += 1
            sim_time = self.current_time()

            for vid in traci.vehicle.getIDList():
                if vid not in expected_ids:
                    continue
                first_seen.setdefault(vid, sim_time)
                pos = traci.vehicle.getPosition(vid)
                last_observed[vid] = {
                    "distance_m": float(traci.vehicle.getDistance(vid)),
                    "waiting_time_s": float(traci.vehicle.getWaitingTime(vid)),
                    "time_loss_s": float(traci.vehicle.getTimeLoss(vid)),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                }

            edge_metrics.extend(self.edge_metrics(step_index))

            for vid in traci.simulation.getArrivedIDList():
                if vid not in expected_ids:
                    continue
                observed = last_observed.get(vid, {})
                vehicle_metrics.append({
                    "vehicle_id": int(vid.lstrip("v")),
                    "travel_time_s": max(0.0, sim_time - first_seen.get(vid, sim_time)),
                    "waiting_time_s": float(observed.get("waiting_time_s", 0.0)),
                    "time_loss_s": float(observed.get("time_loss_s", 0.0)),
                    "distance_m": float(observed.get("distance_m", 0.0)),
                })
                expected_ids.remove(vid)

            if self.current_time() >= self.config.end_time_s:
                break

        for vid in sorted(expected_ids):
            numeric_id = int(vid.lstrip("v"))
            observed = last_observed.get(vid, {})
            vehicle_metrics.append({
                "vehicle_id": numeric_id,
                "travel_time_s": max(0.0, self.current_time() - first_seen.get(vid, self.current_time())),
                "waiting_time_s": float(observed.get("waiting_time_s", 0.0)),
                "time_loss_s": float(observed.get("time_loss_s", 0.0)),
                "distance_m": float(observed.get("distance_m", 0.0)),
                "incomplete": True,
            })

        return SumoTripResult(
            sumo_time_s=self.current_time(),
            vehicle_metrics=vehicle_metrics,
            edge_metrics=edge_metrics,
            edge_flow=None,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.traci.close()
        finally:
            self._closed = True
            self._traci = None


def run_sumo(
    scenario: SumoScenario,
    config: SumoConfig,
    *,
    use_gui: bool,
    output_csv: Path,
) -> dict[str, Any]:
    binary_name = "sumo-gui" if use_gui else "sumo"
    if config.binary != "auto":
        if config.binary in {"sumo", "sumo-gui"}:
            binary = find_sumo_binary(config.binary)
        else:
            candidate = Path(config.binary)
            if not candidate.exists():
                raise FileNotFoundError(f"Configured SUMO binary does not exist: {candidate}")
            binary = str(candidate)
    else:
        binary = find_sumo_binary(binary_name)
    cmd = [binary, "-c", str(scenario.config_file), "--seed", str(config.seed), "--step-length", str(config.step_length_s)]
    try:
        import traci
    except ImportError as exc:
        raise RuntimeError("TraCI is required for SUMO simulation. Install traci==1.27.1.") from exc

    traci.start(cmd, label=f"nash_drl_{config.seed}")
    vehicle_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step * config.step_length_s <= config.end_time_s:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())
            vehicle_ids = list(traci.vehicle.getIDList())
            running_speeds = []
            for vid in vehicle_ids:
                speed = float(traci.vehicle.getSpeed(vid))
                running_speeds.append(speed)
                vehicle_rows.append(
                    {
                        "entity_type": "vehicle",
                        "entity_id": vid,
                        "step": step,
                        "sim_time_s": sim_time,
                        "speed_mps": speed,
                        "position_x": float(traci.vehicle.getPosition(vid)[0]),
                        "position_y": float(traci.vehicle.getPosition(vid)[1]),
                        "road_id": traci.vehicle.getRoadID(vid),
                        "lane_id": traci.vehicle.getLaneID(vid),
                        "distance_m": float(traci.vehicle.getDistance(vid)),
                        "waiting_time_s": float(traci.vehicle.getWaitingTime(vid)),
                        "time_loss_s": float(traci.vehicle.getTimeLoss(vid)),
                    }
                )
            for eid in traci.edge.getIDList():
                if eid.startswith(":"):
                    continue
                edge_rows.append(
                    {
                        "entity_type": "edge",
                        "entity_id": eid,
                        "step": step,
                        "sim_time_s": sim_time,
                        "vehicle_count": int(traci.edge.getLastStepVehicleNumber(eid)),
                        "mean_speed_mps": float(traci.edge.getLastStepMeanSpeed(eid)),
                        "occupancy_pct": float(traci.edge.getLastStepOccupancy(eid)),
                        "halting_number": int(traci.edge.getLastStepHaltingNumber(eid)),
                        "sampled_travel_time_s": float(traci.edge.getTraveltime(eid)),
                    }
                )
            system_rows.append(
                {
                    "entity_type": "system",
                    "entity_id": "system",
                    "step": step,
                    "sim_time_s": sim_time,
                    "active_vehicles": len(vehicle_ids),
                    "mean_speed_mps": (sum(running_speeds) / len(running_speeds)) if running_speeds else 0.0,
                }
            )
            step += 1
    finally:
        traci.close()

    import csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = vehicle_rows + edge_rows + system_rows
    keys = sorted({k for row in combined for k in row})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(combined)
    return {
        "vehicle_rows": vehicle_rows,
        "edge_rows": edge_rows,
        "system_rows": system_rows,
        "steps": step,
        "output_csv": str(output_csv),
    }

@dataclass(slots=True)
class SumoTripResult:
    sumo_time_s: float
    vehicle_metrics: list[dict[str, Any]]
    edge_metrics: list[dict[str, Any]]
    edge_flow: Any | None = None


def run_sumo_trip(
    problem: ProblemDefinition,
    net_file: Path,
    vehicle_routes: dict[int, list[int]],
    config: SumoConfig,
    directory: str | Path,
    *,
    use_gui: bool = False,
) -> SumoTripResult:
    """Execute one synchronous trip leg for all supplied vehicle routes.

    A fresh TraCI/SUMO process is used for each training step. This intentionally
    makes the RL time step equal to one complete trip leg, matching the problem
    definition where each step selects routes from the current stop to the next
    destination.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    builder = SumoScenarioBuilder(problem, config)
    route_file = out / "routes.rou.xml"
    config_file = out / "simulation.sumocfg"
    builder._validate_routes(vehicle_routes)
    builder._write_routes(route_file, vehicle_routes)
    builder._write_config(config_file, net_file, route_file)

    binary_name = "sumo-gui" if use_gui else "sumo"
    binary = find_sumo_binary(binary_name) if config.binary == "auto" else find_sumo_binary(config.binary) if config.binary in {"sumo", "sumo-gui"} else str(Path(config.binary))
    cmd = [binary, "-c", str(config_file), "--seed", str(config.seed), "--step-length", str(config.step_length_s)]
    if use_gui:
        cmd.append("--start")

    try:
        import traci
    except ImportError as exc:
        raise RuntimeError("TraCI is required for SUMO training. Install traci==1.27.1.") from exc

    traci.start(cmd, label=f"nash_drl_train_{config.seed}_{out.name}")
    first_seen: dict[str, float] = {}
    last_observed: dict[str, dict[str, float]] = {}
    vehicle_metrics: list[dict[str, Any]] = []
    edge_metrics: list[dict[str, Any]] = []
    step = 0
    try:
        max_sim_steps = max(1, int(config.end_time_s / config.step_length_s))
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_sim_steps:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())
            vehicle_ids = list(traci.vehicle.getIDList())
            for vid in vehicle_ids:
                first_seen.setdefault(vid, sim_time)
                pos = traci.vehicle.getPosition(vid)
                last_observed[vid] = {
                    "distance_m": float(traci.vehicle.getDistance(vid)),
                    "waiting_time_s": float(traci.vehicle.getWaitingTime(vid)),
                    "time_loss_s": float(traci.vehicle.getTimeLoss(vid)),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                }
            for eid in traci.edge.getIDList():
                if eid.startswith(":"):
                    continue
                edge_metrics.append({
                    "entity_type": "edge",
                    "entity_id": eid,
                    "step": step,
                    "sim_time_s": sim_time,
                    "vehicle_count": int(traci.edge.getLastStepVehicleNumber(eid)),
                    "mean_speed_mps": float(traci.edge.getLastStepMeanSpeed(eid)),
                    "occupancy_pct": float(traci.edge.getLastStepOccupancy(eid)),
                    "halting_number": int(traci.edge.getLastStepHaltingNumber(eid)),
                    "sampled_travel_time_s": float(traci.edge.getTraveltime(eid)),
                })
            for vid in traci.simulation.getArrivedIDList():
                observed = last_observed.get(vid, {})
                vehicle_metrics.append({
                    "vehicle_id": int(vid.lstrip("v")),
                    "travel_time_s": max(0.0, sim_time - first_seen.get(vid, sim_time)),
                    "waiting_time_s": float(observed.get("waiting_time_s", 0.0)),
                    "time_loss_s": float(observed.get("time_loss_s", 0.0)),
                    "distance_m": float(observed.get("distance_m", 0.0)),
                })
            step += 1
    finally:
        traci.close()

    expected_ids = set(vehicle_routes)
    observed_ids = {int(m["vehicle_id"]) for m in vehicle_metrics}
    for vid in sorted(expected_ids - observed_ids):
        vehicle_metrics.append({
            "vehicle_id": vid,
            "travel_time_s": float(config.end_time_s),
            "waiting_time_s": 0.0,
            "time_loss_s": float(config.end_time_s),
            "distance_m": float(last_observed.get(f"v{vid}", {}).get("distance_m", 0.0)),
        })

    return SumoTripResult(
        sumo_time_s=float(step * config.step_length_s),
        vehicle_metrics=vehicle_metrics,
        edge_metrics=edge_metrics,
    )
