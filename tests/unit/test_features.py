from nash_drl.cli import build_demo_problem
from nash_drl.environment import EnvironmentConfig, NashEnvironment, RewardConfig, RewardModel
from nash_drl.features import StateFeatureExtractor


def test_feature_shapes():
    problem = build_demo_problem()
    env = NashEnvironment(problem, EnvironmentConfig(), RewardModel(RewardConfig()))
    state = env.reset()
    inputs = StateFeatureExtractor()(state)
    n = len(problem.vehicles)
    f = state.agent_features.shape[1]
    e = state.csr_map.num_edges
    assert tuple(inputs.invariant.shape) == (n, n - 1, f)
    assert tuple(inputs.non_invariant.shape) == (n, f + e)
