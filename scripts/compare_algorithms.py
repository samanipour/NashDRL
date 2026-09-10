from __future__ import annotations

import argparse
from pathlib import Path

from nash_drl.evaluation.compare import compare_episode_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NashDRL and PPO episode reports")
    parser.add_argument("--nash", default="outputs/training_medium/episode_results.csv")
    parser.add_argument("--ppo", default="outputs/ppo_training_medium/episode_results.csv")
    parser.add_argument("--output", default="outputs/comparison/nash_vs_ppo.csv")
    args = parser.parse_args()
    rows = compare_episode_reports(args.nash, args.ppo, args.output)
    print(f"comparison: {Path(args.output)}")
    print(f"episodes compared: {len(rows)}")


if __name__ == "__main__":
    main()
