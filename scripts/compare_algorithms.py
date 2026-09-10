from __future__ import annotations

import argparse
from pathlib import Path

from nash_drl.config import load_yaml
from nash_drl.evaluation.compare import compare_episode_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NashDRL and PPO episode reports from one experiment")
    parser.add_argument("--config", default="configs/experiments/medium.yaml")
    parser.add_argument("--nash", default=None, help="Override NashDRL episode_results.csv path")
    parser.add_argument("--ppo", default=None, help="Override PPO episode_results.csv path")
    parser.add_argument("--output", default="outputs/comparison/nash_vs_ppo.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    nash_path = args.nash or str(Path(config.get("training", {}).get("output_dir", "outputs/training")) / "episode_results.csv")
    ppo_path = args.ppo or str(Path(config.get("ppo", {}).get("output_dir", "outputs/ppo_training")) / "episode_results.csv")

    rows = compare_episode_reports(nash_path, ppo_path, args.output)
    print(f"config:             {args.config}")
    print(f"NashDRL results:    {nash_path}")
    print(f"PPO results:        {ppo_path}")
    print(f"comparison:         {Path(args.output)}")
    print(f"episodes compared:  {len(rows)}")


if __name__ == "__main__":
    main()
