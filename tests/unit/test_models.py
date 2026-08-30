import torch

from nash_drl.cli import build_demo_problem
from nash_drl.environment import EnvironmentConfig, NashEnvironment, RewardConfig, RewardModel
from nash_drl.features import StateFeatureExtractor
from nash_drl.models import ActorNetwork, CriticNetwork, TargetCriticNetwork


def _inputs():
    problem = build_demo_problem()
    env = NashEnvironment(problem, EnvironmentConfig(), RewardModel(RewardConfig()))
    state = env.reset()
    return problem, state, StateFeatureExtractor()(state)


def test_actor_critic_target_shapes():
    problem, state, inputs = _inputs()
    f = state.agent_features.shape[1]
    e = state.csr_map.num_edges
    actor = ActorNetwork(f, e)
    critic = CriticNetwork(f, e)
    target = TargetCriticNetwork(critic)

    out = actor(inputs)
    assert tuple(out.as_tensor().shape) == (5, len(problem.vehicles), e)
    assert tuple(critic(inputs).shape) == (len(problem.vehicles),)
    assert tuple(target(inputs).shape) == (len(problem.vehicles),)


def test_target_critic_is_frozen_and_hard_updates():
    _, _, inputs = _inputs()
    critic = CriticNetwork(agent_feature_dim=6, num_edges=5)
    target = TargetCriticNetwork(critic)

    for parameter in target.parameters():
        assert not parameter.requires_grad

    with torch.no_grad():
        initial = target(inputs)
        reference = critic(inputs)
    assert torch.allclose(initial, reference, atol=1e-6)

    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(0.01)
    assert not torch.allclose(target(inputs), critic(inputs), atol=1e-6)

    target.hard_update_from(critic)
    with torch.no_grad():
        assert torch.allclose(target(inputs), critic(inputs), atol=1e-6)


def test_actor_and_critic_are_invariant_to_rival_reordering():
    problem, state, inputs = _inputs()
    actor = ActorNetwork(state.agent_features.shape[1], state.csr_map.num_edges)
    critic = CriticNetwork(state.agent_features.shape[1], state.csr_map.num_edges)

    permutation = torch.arange(len(problem.vehicles) - 2, -1, -1)
    permuted = type(inputs)(
        invariant=inputs.invariant[:, permutation, :],
        non_invariant=inputs.non_invariant,
    )

    out_a = actor(inputs)
    out_b = actor(permuted)
    value_a = critic(inputs)
    value_b = critic(permuted)

    assert torch.allclose(out_a.mu, out_b.mu, atol=1e-6)
    assert torch.allclose(out_a.p11, out_b.p11, atol=1e-6)
    assert torch.allclose(out_a.p12, out_b.p12, atol=1e-6)
    assert torch.allclose(out_a.p22, out_b.p22, atol=1e-6)
    assert torch.allclose(out_a.psi, out_b.psi, atol=1e-6)
    assert torch.allclose(value_a, value_b, atol=1e-6)
