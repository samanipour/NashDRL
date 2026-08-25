from cli import build_trainer


def test_training_step_runs():
    trainer = build_trainer({})
    metrics = trainer.train_step()
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "mean_reward" in metrics
