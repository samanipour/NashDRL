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


def test_lq_rival_term_is_sum_of_individual_squares():
    params = ActorOutput(
        mu=torch.zeros(2, 1),
        p11=torch.ones(2, 1),
        p12=torch.zeros(2, 1),
        p22=torch.ones(2, 1),
        psi=torch.zeros(2, 1),
    )
    # z1=1, z2=2. Rival penalty for agent 0 is -(2^2)=-4.
    # If squared-sum were used it would still be -(2^2) here;
    # use 2D with three agents to distinguish the formulas.
    params = ActorOutput(
        mu=torch.zeros(3, 1), p11=torch.ones(3, 1), p12=torch.zeros(3, 1),
        p22=torch.ones(3, 1), psi=torch.zeros(3, 1)
    )
    action=torch.tensor([[0.0],[1.0],[2.0]])
    out=LQAdvantage()(params, action)
    # agent 0 has rivals 1,2 => -(1^2+2^2)=-5; own is zero.
    assert torch.allclose(out[0], torch.tensor(-5.0))
