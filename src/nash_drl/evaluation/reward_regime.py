from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RewardThreshold:
    """Break-even penalty threshold for a violating vehicle outcome.

    For the piecewise reward in the model, a violating outcome returns ``-P``;
    the same outcome without the hard-constraint replacement would return
    ``-(w_T*T + w_C*C)``. Therefore violation is individually reward-preferred
    exactly when ``P < w_T*T + w_C*C``.
    """

    travel_time_h: float
    charging_cost: float
    threshold_penalty: float


def violation_preferred_threshold(
    travel_time_h: float,
    charging_cost: float,
    travel_time_weight: float,
    charging_cost_weight: float,
) -> float:
    return float(travel_time_weight * travel_time_h + charging_cost_weight * charging_cost)


def summarize_thresholds(rows: Iterable[dict], *, travel_time_weight: float = 1.0,
                         charging_cost_weight: float = 1.0) -> dict[str, float]:
    thresholds = [
        violation_preferred_threshold(
            float(r.get("travel_time_h", r.get("travel_time_s", 0.0)) if "travel_time_h" in r else float(r.get("travel_time_s", 0.0))/3600.0),
            float(r.get("charging_cost", 0.0)),
            travel_time_weight,
            charging_cost_weight,
        )
        for r in rows
    ]
    if not thresholds:
        return {"count": 0.0}
    thresholds.sort()
    n = len(thresholds)
    return {
        "count": float(n),
        "min": thresholds[0],
        "p25": thresholds[int(0.25*(n-1))],
        "median": thresholds[int(0.50*(n-1))],
        "p75": thresholds[int(0.75*(n-1))],
        "max": thresholds[-1],
    }
