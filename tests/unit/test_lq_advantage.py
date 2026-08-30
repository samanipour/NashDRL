import torch

from nash_drl.models import ActorOutput, LQAdvantage


def test_lq_advantage_single_agent_contains_only_ego_term():
    params = ActorOutput(
        mu=torch.tensor([[1.0, 2.0]]),
        p11=torch.tensor([[2.0, 3.0]]),
        p12=torch.tensor([[7.0, 7.0]]),
        p22=torch.tensor([[11.0, 11.0]]),
        psi=torch.tensor([[13.0, 13.0]]),
    )
    action = torch.tensor([[2.0, 4.0]])
    output = LQAdvantage()(params, action)
    expected = torch.tensor([-(2.0 * 1.0**2 + 3.0 * 2.0**2)])
    assert torch.allclose(output, expected)


def test_lq_advantage_batch_shape():
    b, n, e = 2, 4, 3
    params = ActorOutput(
        mu=torch.zeros(b, n, e),
        p11=torch.ones(b, n, e),
        p12=torch.ones(b, n, e),
        p22=torch.ones(b, n, e),
        psi=torch.ones(b, n, e),
    )
    action = torch.randn(b, n, e)
    output = LQAdvantage()(params, action)
    assert output.shape == (b, n)
