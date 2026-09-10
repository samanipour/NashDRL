from pathlib import Path

from nash_drl.config import load_yaml
from nash_drl.visualization.training import plot_training_history


def test_all_experiments_include_ppo_settings():
    root = Path(__file__).parents[2]
    for name in ("small.yaml", "medium.yaml", "large.yaml"):
        cfg = load_yaml(root / "configs" / "experiments" / name)
        assert "training" in cfg
        assert "ppo" in cfg
        assert "ppo_network" in cfg
        assert cfg["project"]["seed"] == cfg["ppo"]["seed"] == cfg["training"]["seed"]
        assert cfg["mock_data"]["seed"] == cfg["project"]["seed"]


def test_ppo_history_can_be_plotted_with_shared_visualizer(tmp_path):
    history = [{
        "episode": 0,
        "total_reward": -100.0,
        "td_loss_mean": 10.0,
        "actor_gradient_norm_mean": 1.0,
        "critic_gradient_norm_mean": 2.0,
        "budget_violations": 3,
        "budget_violation_rate": 0.3,
        "mean_step_reward": -50.0,
        "total_travel_time_h": 4.0,
        "total_charging_cost": 100.0,
        "actor_loss_mean": 0.5,
        "critic_loss_mean": 0.7,
    }]
    paths = plot_training_history(history, tmp_path)
    names = {p.name for p in paths}
    assert "mean_step_reward.png" in names
    assert "actor_loss.png" in names
    assert "critic_loss.png" in names
