import torch

from nash_drl.data import NetworkInputs
from nash_drl.models import CriticNetwork


def test_critic_returns_one_value_per_agent():
    n, f, e = 4, 6, 7
    model = CriticNetwork(f, e)
    inputs = NetworkInputs(
        invariant=torch.randn(n, n - 1, f),
        non_invariant=torch.randn(n, f + e),
    )
    values = model(inputs)
    assert values.shape == (n,)
