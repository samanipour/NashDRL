from __future__ import annotations

from pathlib import Path

from nash_drl.visualization.training import plot_training_history

from .runner import TrainingRunner


def run_training(config: dict, *, mode: str | None = None) -> dict:
    result = TrainingRunner(config, mode=mode).run()
    output_dir = Path(result["output_dir"])
    plot_paths = plot_training_history(result["episodes"], output_dir / "plots")
    result["plots"] = plot_paths
    return result
