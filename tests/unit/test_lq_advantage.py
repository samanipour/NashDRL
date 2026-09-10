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


def test_lq_advantage_uses_sum_of_rival_squared_norms_not_squared_sum():
    # One edge is enough to distinguish:
    # z_1 = 1, z_2 = -1 => sum_j z_j = 0, but sum_j z_j^2 = 2.
    params = ActorOutput(
        mu=torch.zeros(3, 1),
        p11=torch.zeros(3, 1),
        p12=torch.zeros(3, 1),
        p22=torch.ones(3, 1),
        psi=torch.zeros(3, 1),
    )
    action = torch.tensor([[0.0], [1.0], [-1.0]])
    output = LQAdvantage()(params, action)
    # Agent 0 sees rivals +1 and -1 -> -2.
    # Agent 1 sees rivals 0 and -1 -> -1.
    # Agent 2 sees rivals 0 and +1 -> -1.
    expected = torch.tensor([-2.0, -1.0, -1.0])
    assert torch.allclose(output, expected)
