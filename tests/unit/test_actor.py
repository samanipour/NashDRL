import torch

from nash_drl.data import NetworkInputs
from nash_drl.models import ActorNetwork


def test_actor_output_channel_and_edge_shapes():
    n, f, e = 4, 6, 7
    model = ActorNetwork(f, e)
    inputs = NetworkInputs(
        invariant=torch.randn(n, n - 1, f),
        non_invariant=torch.randn(n, f + e),
    )
    output = model(inputs)
    assert output.mu.shape == (n, e)
    assert output.p11.shape == (n, e)
    assert output.p12.shape == (n, e)
    assert output.p22.shape == (n, e)
    assert output.psi.shape == (n, e)
    assert output.as_tensor().shape == (5, n, e)


def test_actor_curvature_parameters_are_strictly_positive():
    model = ActorNetwork(6, 5)
    inputs = NetworkInputs(
        invariant=torch.randn(3, 2, 6),
        non_invariant=torch.randn(3, 11),
    )
    output = model(inputs)
    assert torch.all(output.p11 > 0)
    assert torch.all(output.p22 > 0)
