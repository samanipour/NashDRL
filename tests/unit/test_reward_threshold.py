from nash_drl.evaluation.reward_regime import violation_preferred_threshold


def test_violation_preferred_when_penalty_is_below_regular_outcome_cost():
    threshold = violation_preferred_threshold(2.0, 150.0, 1.0, 1.0)
    assert threshold == 152.0
    assert 25.0 < threshold < 500.0
