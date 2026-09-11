from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from nash_drl.config import load_yaml
from nash_drl.evaluation.p_sweep import analyze_sweep
from nash_drl.training.runner import TrainingRunner
from ppo.training.ppo_runner import PPOTrainingRunner


def _parse_values(text: str) -> list[float]:
    values: list[float] = []
    for token in re.split(r"[,\s]+", text.strip()):
        if token:
            values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError("P sweep must contain at least one value")
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("P must be non-negative")
    return sorted(set(values))


def _p_label(p: float) -> str:
    return f"P_{p:g}"


def _prepare_config(base: dict[str, Any], p: float, root: Path, episodes: int | None, mode: str | None, seed: int | None) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("training", {})
    cfg.setdefault("ppo", {})
    cfg.setdefault("reward", {})
    cfg["reward"].setdefault("common", {})["budget_penalty"] = float(p)
    cfg["reward"].setdefault("nash_drl", {})["budget_penalty"] = float(p)
    cfg["reward"].setdefault("ppo", {})["budget_penalty"] = float(p)
    if episodes is not None:
        cfg["training"]["episodes"] = episodes
        cfg["ppo"]["episodes"] = episodes
    if mode is not None:
        cfg["training"]["mode"] = mode
        cfg["ppo"]["mode"] = mode
    if seed is not None:
        cfg["project"]["seed"] = seed
        cfg["training"]["seed"] = seed
        cfg["ppo"]["seed"] = seed
        cfg["simulation"]["sumo"]["seed"] = seed
        cfg["mock_data"]["seed"] = seed
    cfg["training"]["output_dir"] = str(root / _p_label(p) / "nash")
    cfg["ppo"]["output_dir"] = str(root / _p_label(p) / "ppo")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run identical-common-P medium experiments for NashDRL and PPO and analyze the sweep."
    )
    parser.add_argument("--config", default="configs/experiments/medium.yaml")
    parser.add_argument("--p-values", type=_parse_values, default=None, help="Comma/space separated common P values")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes for both algorithms")
    parser.add_argument("--mode", choices=["mock", "real"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/p_sweep_medium")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Do not retrain a P when both episode reports already exist")
    args = parser.parse_args()

    base = load_yaml(args.config)
    sweep_cfg = base.get("p_sweep", {})
    p_values = args.p_values or _parse_values(",".join(str(x) for x in sweep_cfg.get("values", [0, 2, 5, 10, 15, 20, 30])))
    episodes = args.episodes if args.episodes is not None else sweep_cfg.get("episodes")
    mode = args.mode or sweep_cfg.get("mode")
    seed = args.seed if args.seed is not None else sweep_cfg.get("seed")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": args.config,
        "p_values": p_values,
        "episodes": episodes,
        "mode": mode,
        "seed": seed,
        "common_P_for_both_algorithms": True,
        "reward_profiles": {"nash_drl": "common P", "ppo": "common P"},
    }
    (root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.analyze_only:
        for p in p_values:
            cfg = _prepare_config(base, p, root, episodes, mode, seed)
            label = _p_label(p)
            nash_ep = Path(cfg["training"]["output_dir"]) / "episode_results.csv"
            ppo_ep = Path(cfg["ppo"]["output_dir"]) / "episode_results.csv"
            if args.skip_existing and nash_ep.exists() and ppo_ep.exists():
                print(f"[{label}] skip-existing")
                continue
            print(f"[{label}] training NashDRL with common P={p:g}")
            TrainingRunner(cfg, mode=mode).run()
            print(f"[{label}] training PPO with common P={p:g}")
            PPOTrainingRunner(cfg, mode=mode).run()

    result = analyze_sweep(root, root / "analysis")
    print("\nP-sweep completed")
    print(f"root:              {root}")
    print(f"summary:            {root / 'analysis' / 'sweep_summary.csv'}")
    print(f"candidate table:    {root / 'analysis' / 'hypothesis_candidates.csv'}")
    print(f"candidate JSON:     {root / 'analysis' / 'hypothesis_candidates.json'}")
    print(f"weak candidates:    {result['candidates']['weak_hypothesis_P_values']}")
    print(f"strong candidates:  {result['candidates']['strong_hypothesis_P_values']}")


if __name__ == "__main__":
    main()
