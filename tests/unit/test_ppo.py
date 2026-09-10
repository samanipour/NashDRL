import torch

from nash_drl.data import NetworkInputs
from ppo.models.ppo import PPOActorCritic
from ppo.training.ppo_rollout import compute_gae


def inputs(n=4, f=6, e=8):
    return NetworkInputs(
        torch.randn(n, n - 1, f),
        torch.randn(n, f + e),
    )


def test_ppo_forward_and_action_shapes():
    model = PPOActorCritic(6, 8)
    out = model(inputs())
    assert out.mean.shape == (4, 8)
    assert out.log_std.shape == (4, 8)
    assert out.value.ndim == 0

    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    action, logp, entropy, value = model.sample_action(inputs(), active_mask=mask)
    assert action.shape == (4, 8)
    assert logp.ndim == 0
    assert entropy.ndim == 0
    assert value.ndim == 0
    assert torch.all(action[2] == 0)


def test_ppo_evaluate_action_matches_shapes():
    model = PPOActorCritic(6, 8)
    x = inputs()
    action, old_logp, _, _ = model.sample_action(x)
    logp, entropy, value = model.evaluate_actions(x, action)
    assert logp.shape == old_logp.shape
    assert entropy.ndim == 0
    assert value.ndim == 0


def test_gae_scalar_system_reward():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.5])
    dones = torch.tensor([0.0, 1.0])
    adv, returns = compute_gae(rewards, values, dones, torch.tensor(0.0), 0.99, 0.95)
    assert adv.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.allclose(returns, adv + values)
