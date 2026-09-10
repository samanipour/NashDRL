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


def test_nash_losses_share_residual_value_but_have_distinct_gradient_paths():
    p = _params(batch=1, agents=2, edges=2)
    action = torch.tensor([[[1.0, -0.5], [0.2, 0.8]]])
    value = torch.tensor([[0.4, -0.2]], requires_grad=True)
    target = torch.tensor([[0.1, 0.3]])
    reward = torch.tensor([[0.2, 0.5]])
    done = torch.zeros(1, 2)

    result = compute_training_losses(value, target, reward, p, action, 0.99, done)

    # Same scalar squared-residual value is expected; detach semantics make
    # the gradient paths different, which is the important distinction.
    assert torch.allclose(result.actor_loss, result.critic_loss)

    result.critic_loss.backward(retain_graph=True)
    assert value.grad is not None
    assert p.mu.grad is None or torch.all(p.mu.grad == 0)

    p.mu.grad = None
    value.grad = None
    result.actor_loss.backward()
    assert p.mu.grad is not None
    assert p.mu.grad.abs().sum() > 0
    assert value.grad is None or torch.all(value.grad == 0)
