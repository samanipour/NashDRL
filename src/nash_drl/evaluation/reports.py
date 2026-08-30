from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


def format_result(result: EvaluationResult) -> str:
    return (
        f"episode={result.episode}, steps={result.steps}, total_reward={result.total_reward:.4f}, "
        f"budget_violations={result.budget_violations}, "
        f"travel_time_s={result.total_travel_time_s:.4f}, charging_cost={result.total_charging_cost:.4f}"
    )


def summarize(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {}
    n = float(len(results))
    return {
        "mean_total_reward": sum(r.total_reward for r in results) / n,
        "mean_budget_violations": sum(r.budget_violations for r in results) / n,
        "mean_travel_time_s": sum(r.total_travel_time_s for r in results) / n,
        "mean_charging_cost": sum(r.total_charging_cost for r in results) / n,
    }
