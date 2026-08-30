from __future__ import annotations

from .evaluator import EvaluationResult


def format_result(result: EvaluationResult) -> str:
    return f"total_reward={result.total_reward:.4f}, budget_violations={result.budget_violations}"
