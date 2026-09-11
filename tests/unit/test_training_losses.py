import pytest
import torch

from nash_drl.models import ActorOutput
from nash_drl.training.losses import compute_training_losses


def _params(batch=2, agents=3, edges=4):
    return ActorOutput(
        mu=torch.zeros(batch, agents, edges, requires_grad=True),
        p11=torch.ones(batch, agents, edges),
        p12=torch.zeros(batch, agents, edges),
        p22=torch.ones(batch, agents, edges),
        psi=torch.zeros(batch, agents, edges),
    )


def test_loss_requires_detached_executed_action():
    p = _params()
    action = torch.randn(2, 3, 4, requires_grad=True)
    value = torch.zeros(2, 3, requires_grad=True)
    target = torch.zeros(2, 3)
    reward = torch.ones(2, 3)
    done = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="action must be detached"):
        compute_training_losses(value, target, reward, p, action, 0.99, done)


def test_loss_produces_actor_and_critic_gradients_with_detached_action():
    p = _params()
    action = torch.randn(2, 3, 4).detach()
    value = torch.zeros(2, 3, requires_grad=True)
    target = torch.zeros(2, 3)
    reward = torch.ones(2, 3)
    done = torch.zeros(2, 3)
    result = compute_training_losses(value, target, reward, p, action, 0.99, done)

    result.critic_loss.backward(retain_graph=True)
    assert value.grad is not None
    assert value.grad.abs().sum() > 0

    p.mu.grad = None
    result.actor_loss.backward()
    assert p.mu.grad is not None
    assert p.mu.grad.abs().sum() > 0
