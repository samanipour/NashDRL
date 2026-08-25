from __future__ import annotations

from dataclasses import dataclass

from training import NashDRLTrainer


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    total_reward: float
    budget_violations: int


class Evaluator:
    def __init__(self, trainer: NashDRLTrainer) -> None:
        self.trainer = trainer

    def run_once(self) -> EvaluationResult:
        state = self.trainer.env.reset()
        action = self.trainer.select_action(state, training=False)
        paths = self.trainer.mapper.map(action, state)
        _, reward, _, _ = self.trainer.env.step(paths)
        return EvaluationResult(float(reward.total_reward), int(reward.budget_violations.sum().item()))
