from __future__ import annotations

import csv
from pathlib import Path


def _rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def plot_training_history(
    history: list[dict], output_dir: str | Path, *, rolling_window: int = 25
) -> list[Path]:
    """Create raw + rolling learning curves for the major training metrics."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not history:
        return []

    # Actor and Critic use the same squared TD residual in the paper's
    # alternating optimization. Their scalar losses are therefore expected
    # to be numerically identical; the optimization differs only by gradient
    # detachment. Plot update norms separately instead of implying two
    # independent scalar objectives.
    metrics = [
        ("total_reward", "Episode total reward", "episode_reward.png"),
        ("td_loss_mean", "Shared TD loss", "td_loss.png"),
        ("actor_gradient_norm_mean", "Actor gradient norm", "actor_gradient_norm.png"),
        ("critic_gradient_norm_mean", "Critic gradient norm", "critic_gradient_norm.png"),
        ("budget_violations", "Vehicles with hard-constraint violation", "budget_violations.png"),
        ("budget_violation_rate", "Hard-constraint violation rate", "budget_violation_rate.png"),
        ("mean_step_reward", "Mean step reward", "mean_step_reward.png"),
        ("total_travel_time_h", "Total travel time (h)", "travel_time.png"),
        ("total_charging_cost", "Total charging cost", "charging_cost.png"),
    ]
    paths: list[Path] = []
    x = [int(row["episode"]) for row in history]
    for key, ylabel, filename in metrics:
        raw = [float(row[key]) for row in history]
        smooth = _rolling(raw, rolling_window)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, raw, linewidth=0.7, alpha=0.28, label="Episode")
        if rolling_window > 1:
            ax.plot(x, smooth, linewidth=2.0, label=f"Rolling mean ({rolling_window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    # Keep legacy filenames so existing workflows do not break. These are the
    # same scalar TD objective seen from each alternating update path.
    for key, title, filename in [
        ("actor_loss_mean", "Actor objective (shared TD residual)", "actor_loss.png"),
        ("critic_loss_mean", "Critic objective (shared TD residual)", "critic_loss.png"),
    ]:
        raw = [float(row[key]) for row in history]
        smooth = _rolling(raw, rolling_window)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, raw, linewidth=0.7, alpha=0.28, label="Episode")
        if rolling_window > 1:
            ax.plot(x, smooth, linewidth=2.0, label=f"Rolling mean ({rolling_window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
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
