from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_pair(nash_path: Path, ppo_path: Path, p_value: float) -> dict[str, Any]:
    nash = _read_csv(nash_path)
    ppo = _read_csv(ppo_path)
    if not nash or not ppo:
        raise ValueError(f"Empty episode report for P={p_value}: {nash_path}, {ppo_path}")
    nash_by_ep = {int(r["episode"]): r for r in nash}
    ppo_by_ep = {int(r["episode"]): r for r in ppo}
    episodes = sorted(set(nash_by_ep) & set(ppo_by_ep))
    if not episodes:
        raise ValueError(f"No common episodes for P={p_value}")

    def mean(algo: dict[int, dict[str, str]], key: str) -> float:
        return sum(_num(algo[e], key) for e in episodes) / len(episodes)

    metrics = {
        "total_reward": ("total_reward", "total_reward"),
        "learning_total_reward": ("learning_total_reward", "learning_total_reward"),
        "benchmark_total_reward": ("benchmark_total_reward", "benchmark_total_reward"),
        "budget_violations": ("budget_violations", "budget_violations"),
        "budget_violation_rate": ("budget_violation_rate", "budget_violation_rate"),
        "budget_violation_events": ("budget_violation_events", "budget_violation_events"),
        "total_travel_time_h": ("total_travel_time_h", "total_travel_time_h"),
        "total_charging_cost": ("total_charging_cost", "total_charging_cost"),
        "mean_step_reward": ("mean_step_reward", "mean_step_reward"),
    }
    out: dict[str, Any] = {"P": float(p_value), "episodes": len(episodes)}
    for name, (nk, pk) in metrics.items():
        nm, pm = mean(nash_by_ep, nk), mean(ppo_by_ep, pk)
        out[f"nash_{name}"] = nm
        out[f"ppo_{name}"] = pm
        out[f"delta_ppo_minus_nash_{name}"] = pm - nm

    out["hypothesis_reward_ppo_better"] = bool(out["ppo_total_reward"] > out["nash_total_reward"])
    out["hypothesis_violations_ppo_higher"] = bool(out["ppo_budget_violations"] > out["nash_budget_violations"])
    weak_episode_hits = 0
    strong_episode_hits = 0
    for e in episodes:
        reward_hit = _num(ppo_by_ep[e], "total_reward") > _num(nash_by_ep[e], "total_reward")
        violation_hit = _num(ppo_by_ep[e], "budget_violations") > _num(nash_by_ep[e], "budget_violations")
        travel_hit = _num(ppo_by_ep[e], "total_travel_time_h") <= _num(nash_by_ep[e], "total_travel_time_h")
        weak_episode_hits += int(reward_hit and violation_hit)
        strong_episode_hits += int(reward_hit and violation_hit and travel_hit)
    out["weak_hypothesis_episode_rate"] = weak_episode_hits / len(episodes)
    out["strong_hypothesis_episode_rate"] = strong_episode_hits / len(episodes)
    out["hypothesis_travel_time_ppo_not_worse"] = bool(out["ppo_total_travel_time_h"] <= out["nash_total_travel_time_h"])
    out["hypothesis_weak"] = bool(out["hypothesis_reward_ppo_better"] and out["hypothesis_violations_ppo_higher"])
    out["hypothesis_strong"] = bool(
        out["hypothesis_weak"] and out["hypothesis_travel_time_ppo_not_worse"]
    )
    return out


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze_sweep(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    output = Path(output_dir) if output_dir is not None else root / "analysis"
    rows: list[dict[str, Any]] = []
    for sweep_dir in sorted(root.glob("P_*")):
        if not sweep_dir.is_dir():
            continue
        try:
            p = float(sweep_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        nash_path = sweep_dir / "nash" / "episode_results.csv"
        ppo_path = sweep_dir / "ppo" / "episode_results.csv"
        if nash_path.exists() and ppo_path.exists():
            rows.append(summarize_pair(nash_path, ppo_path, p))
    if not rows:
        raise ValueError(f"No completed P-sweep pairs found under {root}")
    rows.sort(key=lambda r: r["P"])
    write_summary(rows, output / "sweep_summary.csv")

    weak = [r for r in rows if r["hypothesis_weak"]]
    strong = [r for r in rows if r["hypothesis_strong"]]
    candidates = {
        "weak_hypothesis_P_values": [r["P"] for r in weak],
        "strong_hypothesis_P_values": [r["P"] for r in strong],
        "weak_range": _contiguous_range([r["P"] for r in weak]),
        "strong_range": _contiguous_range([r["P"] for r in strong]),
    }
    (output / "hypothesis_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    _write_candidate_table(rows, output / "hypothesis_candidates.csv")
    return {"rows": rows, "candidates": candidates, "output_dir": output}


def _contiguous_range(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    return [values[0], values[-1]]


def _write_candidate_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "P",
        "episodes",
        "nash_total_reward",
        "ppo_total_reward",
        "delta_ppo_minus_nash_total_reward",
        "nash_budget_violations",
        "ppo_budget_violations",
        "delta_ppo_minus_nash_budget_violations",
        "nash_total_travel_time_h",
        "ppo_total_travel_time_h",
        "nash_total_charging_cost",
        "ppo_total_charging_cost",
        "hypothesis_reward_ppo_better",
        "hypothesis_violations_ppo_higher",
        "hypothesis_weak",
        "hypothesis_strong",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
