from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nash_drl.config import load_yaml
from nash_drl.environment.reward import build_reward_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the budget-penalty break-even regime for an experiment.")
    parser.add_argument("--config", default="configs/experiments/medium.yaml")
    parser.add_argument("--vehicles", default=None, help="vehicle_results.csv path")
    parser.add_argument("--output", default=None, help="output CSV path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    vehicle_path = Path(args.vehicles or (Path(cfg.get("training", {}).get("output_dir", "outputs/training")) / "vehicle_results.csv"))
    out_path = Path(args.output or (vehicle_path.parent / "reward_threshold_analysis.csv"))
    nash = build_reward_config(cfg, "nash_drl")
    ppo = build_reward_config(cfg, "ppo")
    common = build_reward_config(cfg, "common")

    rows = []
    with vehicle_path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("inactive") == "True":
                continue
            threshold = float(r.get("violation_break_even_penalty", 0.0))
            rows.append({
                **r,
                "break_even_penalty": threshold,
                "nash_violation_reward_preferred": int(nash.budget_penalty < threshold),
                "ppo_violation_reward_preferred": int(ppo.budget_penalty < threshold),
                "common_violation_reward_preferred": int(common.budget_penalty < threshold),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def frac(k: str) -> float:
        return sum(int(r[k]) for r in rows) / max(1, len(rows))

    print(f"vehicle rows: {len(rows)}")
    print(f"NashDRL P={nash.budget_penalty:g}: violation reward-preferred fraction={frac('nash_violation_reward_preferred'):.3f}")
    print(f"PPO     P={ppo.budget_penalty:g}: violation reward-preferred fraction={frac('ppo_violation_reward_preferred'):.3f}")
    print(f"Common  P={common.budget_penalty:g}: violation reward-preferred fraction={frac('common_violation_reward_preferred'):.3f}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
