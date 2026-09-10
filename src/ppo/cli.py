from __future__ import annotations

import argparse

from nash_drl.config import load_yaml
from ppo.evaluation import PPOEvaluationRunner
from ppo.training import PPOTrainingRunner


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/experiments/medium_ppo.yaml")
    parser.add_argument("--mode", choices=["mock", "real"])
    parser.add_argument("--episodes", type=int)
    return parser


def train() -> None:
    parser = _build_parser("Train the standalone PPO baseline")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config.setdefault("ppo", {})
    if args.mode:
        config["ppo"]["mode"] = args.mode
    if args.episodes is not None:
        config["ppo"]["episodes"] = args.episodes
    result = PPOTrainingRunner(config, mode=args.mode).run()
    print("PPO training completed")
    print(f"dataset:          {result['dataset_path']}")
    print(f"output directory: {result['output_dir']}")
    print(f"episodes:         {len(result['episodes'])}")
    print(f"steps/episode:    {result['metadata']['steps_per_episode']}")


def evaluate() -> None:
    parser = _build_parser("Evaluate a trained standalone PPO baseline")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config.setdefault("evaluation", {})
    config["evaluation"]["algorithm"] = "ppo"
    if args.mode:
        config["evaluation"]["mode"] = args.mode
    if args.episodes is not None:
        config["evaluation"]["episodes"] = args.episodes
    results = PPOEvaluationRunner(
        config,
        mode=args.mode,
        checkpoint=args.checkpoint,
    ).run()
    for result in results:
        print(result)
