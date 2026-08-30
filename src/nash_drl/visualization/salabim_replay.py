from __future__ import annotations

import csv
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Callable


class SalabimReplay:
    """Replay SUMO vehicle trajectories using salabim animation objects."""

    def __init__(self, trajectories: dict[str, list[tuple[float, float, float]]]) -> None:
        self.trajectories = trajectories

    @classmethod
    def from_trace(cls, path: str | Path) -> "SalabimReplay":
        trajectories: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("entity_type") != "vehicle":
                    continue
                trajectories[row["entity_id"]].append(
                    (float(row["sim_time_s"]), float(row["position_x"]), float(row["position_y"]))
                )
        for points in trajectories.values():
            points.sort(key=lambda x: x[0])
        return cls(dict(trajectories))

    @staticmethod
    def _interpolator(points: list[tuple[float, float, float]], axis: int) -> Callable[[float], float]:
        times = [p[0] for p in points]

        def value(t: float) -> float:
            if not points:
                return 0.0
            if t <= times[0]:
                return points[0][axis]
            if t >= times[-1]:
                return points[-1][axis]
            idx = max(0, bisect_right(times, t) - 1)
            t0, a0 = points[idx][0], points[idx][axis]
            t1, a1 = points[idx + 1][0], points[idx + 1][axis]
            if t1 <= t0:
                return a0
            alpha = (t - t0) / (t1 - t0)
            return a0 + alpha * (a1 - a0)

        return value

    def show(self) -> None:
        if not self.trajectories:
            return
        import salabim as sim

        env = sim.Environment(trace=False)
        env.animate(True)
        env.modelname("Nash-DRL / SUMO trajectory replay")

        class ReplayClock(sim.Component):
            def process(self):
                max_t = max(points[-1][0] for points in self_outer.trajectories.values() if points)
                self.hold(max_t + 0.1)

        self_outer = self
        for vid, points in self.trajectories.items():
            if not points:
                continue
            sim.AnimateCircle(
                radius=4,
                x=self._interpolator(points, 1),
                y=self._interpolator(points, 2),
                text=vid,
                t0=points[0][0],
                t1=points[-1][0],
            )
        ReplayClock()
        env.run()
