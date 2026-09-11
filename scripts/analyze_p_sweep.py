from __future__ import annotations

import argparse
from pathlib import Path
from nash_drl.evaluation.p_sweep import analyze_sweep
from nash_drl.visualization.p_sweep import create_p_sweep_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed NashDRL/PPO P-sweep results.")
    parser.add_argument("--root", default="outputs/p_sweep_medium")
    parser.add_argument("--rolling-window", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.root)
    result = analyze_sweep(root, root / "analysis")
    paths = create_p_sweep_plots(root, root / "analysis" / "plots", rolling_window=args.rolling_window)
    print(f"summary:           {root / 'analysis' / 'sweep_summary.csv'}")
    print(f"candidates:        {root / 'analysis' / 'hypothesis_candidates.json'}")
    print(f"plots generated:   {len(paths)}")
    for path in paths:
        print(path)
    print(f"weak candidates:   {result['candidates']['weak_hypothesis_P_values']}")
    print(f"strong candidates: {result['candidates']['strong_hypothesis_P_values']}")


if __name__ == "__main__":
    main()
