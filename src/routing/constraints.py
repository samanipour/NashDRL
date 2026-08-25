from __future__ import annotations

from data import Paths


def validate_paths(paths: Paths, num_agents: int, num_edges: int) -> None:
    if len(paths.edge_ids) != num_agents:
        raise ValueError("One path is required per agent")
    for path in paths.edge_ids:
        for edge_id in path:
            if not 0 <= edge_id < num_edges:
                raise ValueError(f"Invalid edge id: {edge_id}")
