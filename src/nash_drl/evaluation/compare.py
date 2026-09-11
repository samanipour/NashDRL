from __future__ import annotations

import csv
from pathlib import Path


def compare_episode_reports(
    nash_path: str | Path,
    ppo_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, float]]:
    def read(path: Path) -> dict[int, dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as f:
            return {int(r["episode"]): r for r in csv.DictReader(f)}

    nash = read(Path(nash_path))
    ppo = read(Path(ppo_path))
    rows: list[dict[str, float]] = []
    for episode in sorted(set(nash) & set(ppo)):
        a, b = nash[episode], ppo[episode]
        rows.append({
            "episode": float(episode),
            "nash_total_reward": float(a["total_reward"]),
            "ppo_total_reward": float(b["total_reward"]),
            "nash_benchmark_total_reward": float(a.get("benchmark_total_reward", a["total_reward"])),
            "ppo_benchmark_total_reward": float(b.get("benchmark_total_reward", b["total_reward"])),
            "benchmark_reward_delta_ppo_minus_nash": float(b.get("benchmark_total_reward", b["total_reward"])) - float(a.get("benchmark_total_reward", a["total_reward"])),
            "learning_reward_delta_ppo_minus_nash": float(b["total_reward"]) - float(a["total_reward"]),
            "nash_budget_violations": float(a["budget_violations"]),
            "ppo_budget_violations": float(b["budget_violations"]),
            "nash_budget_violation_rate": float(a["budget_violation_rate"]),
            "ppo_budget_violation_rate": float(b["budget_violation_rate"]),
            "nash_total_travel_time_h": float(a["total_travel_time_h"]),
            "ppo_total_travel_time_h": float(b["total_travel_time_h"]),
            "nash_total_charging_cost": float(a["total_charging_cost"]),
            "ppo_total_charging_cost": float(b["total_charging_cost"]),
        })
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
