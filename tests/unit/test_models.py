from cli import build_demo_problem
from environment import EnvironmentConfig, NashEnvironment, RewardConfig, RewardModel
from features import StateFeatureExtractor
from models import ActorNetwork, CriticNetwork, TargetCriticNetwork


def test_actor_critic_shapes():
    problem = build_demo_problem()
    env = NashEnvironment(problem, EnvironmentConfig(), RewardModel(RewardConfig()))
    state = env.reset()
    inputs = StateFeatureExtractor()(state)
    f = state.agent_features.shape[1]
    e = state.csr_map.num_edges
    actor = ActorNetwork(f, e)
    critic = CriticNetwork(f, e)
    target = TargetCriticNetwork(critic)
    out = actor(inputs)
    assert tuple(out.mu.shape) == (len(problem.vehicles), e)
    assert tuple(out.p11.shape) == (len(problem.vehicles), e)
    assert tuple(critic(inputs).shape) == (len(problem.vehicles),)
    assert tuple(target(inputs).shape) == (len(problem.vehicles),)
