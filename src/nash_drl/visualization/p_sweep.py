from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    q: list[float] = []
    total = 0.0
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.pop(0)
        out.append(total / len(q))
    return out


def create_p_sweep_plots(root: str | Path, output_dir: str | Path, *, rolling_window: int = 10) -> list[Path]:
    import matplotlib.pyplot as plt

    root, output = Path(root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ps: list[float] = []
    pairs: dict[float, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    for pdir in sorted(root.glob("P_*")):
        try:
            p = float(pdir.name.split("_", 1)[1])
        except ValueError:
            continue
        nash_path, ppo_path = pdir / "nash" / "episode_results.csv", pdir / "ppo" / "episode_results.csv"
        if nash_path.exists() and ppo_path.exists():
            ps.append(p)
            pairs[p] = (_read(nash_path), _read(ppo_path))
    ps.sort()
    if not ps:
        return []

    metrics = [
        ("budget_violations", "Budget violations", "budget_violations_by_p.png"),
        ("budget_violation_rate", "Budget violation rate", "budget_violation_rate_by_p.png"),
        ("total_reward", "Episode reward", "episode_reward_by_p.png"),
        ("benchmark_total_reward", "Common benchmark episode reward", "benchmark_reward_by_p.png"),
        ("total_travel_time_h", "Total travel time (h)", "travel_time_by_p.png"),
        ("total_charging_cost", "Total charging cost", "charging_cost_by_p.png"),
        ("budget_violation_events", "Budget violation events", "budget_violation_events_by_p.png"),
    ]
    paths: list[Path] = []
    cols = min(3, len(ps))
    rows = (len(ps) + cols - 1) // cols
    for key, ylabel, filename in metrics:
        fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.8 * rows), squeeze=False)
        axes_flat = [ax for row in axes for ax in row]
        for idx, p in enumerate(ps):
            ax = axes_flat[idx]
            nash, ppo = pairs[p]
            nash = sorted(nash, key=lambda r: int(r["episode"]))
            ppo = sorted(ppo, key=lambda r: int(r["episode"]))
            nx = [int(r["episode"]) for r in nash]
            px = [int(r["episode"]) for r in ppo]
            nv = [float(r[key]) for r in nash]
            pv = [float(r[key]) for r in ppo]
            ax.plot(nx, _rolling(nv, rolling_window), linewidth=2.0, label="NashDRL")
            ax.plot(px, _rolling(pv, rolling_window), linewidth=2.0, label="PPO")
            ax.set_title(f"P = {p:g}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
        for ax in axes_flat[len(ps):]:
            ax.axis("off")
        fig.suptitle(f"{ylabel}: NashDRL vs PPO across common P", y=1.01)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Across-P mean comparison: one point per algorithm and P.
    summary_path = output.parent / "sweep_summary.csv"
    if summary_path.exists():
        summary = _read(summary_path)
        pvals = [float(r["P"]) for r in summary]
        for metric, ylabel, filename in [
            ("total_reward", "Mean episode reward", "mean_reward_vs_p.png"),
            ("budget_violations", "Mean budget violations", "mean_violations_vs_p.png"),
            ("total_travel_time_h", "Mean total travel time (h)", "mean_travel_time_vs_p.png"),
            ("total_charging_cost", "Mean total charging cost", "mean_charging_cost_vs_p.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(pvals, [float(r[f"nash_{metric}"]) for r in summary], marker="o", label="NashDRL")
            ax.plot(pvals, [float(r[f"ppo_{metric}"]) for r in summary], marker="o", label="PPO")
            ax.set_xlabel("Common budget penalty P")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = output / filename
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)

        # Reward-vs-violations trade-off map, each P connected through its two points.
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in summary:
            ax.plot(
                [float(r["nash_budget_violations"]), float(r["ppo_budget_violations"])],
                [float(r["nash_total_reward"]), float(r["ppo_total_reward"])],
                marker="o",
                linewidth=1.2,
            )
            ax.text(float(r["nash_budget_violations"]), float(r["nash_total_reward"]), f"N {float(r['P']):g}", fontsize=7)
            ax.text(float(r["ppo_budget_violations"]), float(r["ppo_total_reward"]), f"P {float(r['P']):g}", fontsize=7)
        ax.set_xlabel("Mean budget violations")
        ax.set_ylabel("Mean episode reward")
        ax.set_title("Reward–constraint trade-off across common P")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = output / "reward_vs_violations_tradeoff.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths
