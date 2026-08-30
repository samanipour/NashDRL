from __future__ import annotations

from nash_drl.data import Paths


def route_lengths(paths: Paths, edge_lengths: list[float]) -> list[float]:
    return [sum(edge_lengths[e] for e in path) for path in paths.edge_ids]
