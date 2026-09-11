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
    parser.add_argument("--reward-metric", choices=["learning", "benchmark"], default="benchmark", help="Which total reward metric to emphasize in the console summary.")
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
    if rows:
        metric = "nash_benchmark_total_reward" if args.reward_metric == "benchmark" else "nash_total_reward"
        metric_b = "ppo_benchmark_total_reward" if args.reward_metric == "benchmark" else "ppo_total_reward"
        nash_mean = sum(r[metric] for r in rows) / len(rows)
        ppo_mean = sum(r[metric_b] for r in rows) / len(rows)
        print(f"mean NashDRL {args.reward_metric} reward: {nash_mean:.3f}")
        print(f"mean PPO {args.reward_metric} reward:     {ppo_mean:.3f}")


if __name__ == "__main__":
    main()
