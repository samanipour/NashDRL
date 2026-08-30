import torch

from nash_drl.models import DeepSetEncoder


def test_deep_sets_sum_is_permutation_invariant():
    torch.manual_seed(11)
    model = DeepSetEncoder(feature_dim=6, embedding_dim=10, hidden_dim=32)
    x = torch.randn(5, 4, 6)
    permutation = torch.tensor([3, 1, 0, 2])
    y1 = model(x)
    y2 = model(x[:, permutation, :])
    assert torch.allclose(y1, y2, atol=1e-6)


def test_deep_sets_supports_empty_rival_set():
    model = DeepSetEncoder(feature_dim=6, embedding_dim=10, hidden_dim=32)
    x = torch.empty(1, 0, 6)
    output = model(x)
    assert output.shape == (1, 10)
    assert torch.allclose(output, torch.zeros_like(output))
