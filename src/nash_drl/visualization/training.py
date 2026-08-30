from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def plot_training_history(history: list[dict], output_dir: str | Path) -> list[Path]:
    """Create one diagram per major learning metric and save PNG files."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not history:
        return []
    metrics = [
        ("total_reward", "Episode total reward", "episode_reward.png"),
        ("actor_loss_mean", "Actor loss", "actor_loss.png"),
        ("critic_loss_mean", "Critic loss", "critic_loss.png"),
        ("budget_violations", "Hard-constraint violations", "budget_violations.png"),
        ("total_travel_time_h", "Total travel time (h)", "travel_time.png"),
        ("total_charging_cost", "Total charging cost", "charging_cost.png"),
    ]
    paths: list[Path] = []
    x = [row["episode"] for row in history]
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, [row[key] for row in history], marker="o", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def load_episode_results(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
