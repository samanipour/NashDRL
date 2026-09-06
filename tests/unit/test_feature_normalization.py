import torch

from nash_drl.domain import Trip, Vehicle
from nash_drl.features.agent_features import encode_agent_features


def test_agent_features_are_normalized():
    vehicles = [
        Vehicle(0, [Trip(0, 1), Trip(1, 2)], 50.0, 200.0),
        Vehicle(1, [Trip(1, 2)], 75.0, 100.0),
    ]
    features = encode_agent_features(
        vehicles,
        5,
        dtype=torch.float32,
        max_trips=2,
        max_budget=200,
        max_speed_kmh=75,
    )
    assert features.shape == (2, 6)
    assert torch.all(features >= 0)
    assert torch.all(features <= 1)
