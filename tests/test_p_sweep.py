from pathlib import Path

from nash_drl.evaluation.p_sweep import summarize_pair


def _write(path: Path, rows: list[dict]) -> None:
    import csv
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _row(ep, reward, vio, tt):
    return {
        'episode': ep,
        'total_reward': reward,
        'learning_total_reward': reward,
        'benchmark_total_reward': reward,
        'budget_violations': vio,
        'budget_violation_rate': vio / 20,
        'budget_violation_events': vio,
        'total_travel_time_h': tt,
        'total_charging_cost': 100.0,
        'mean_step_reward': reward / 2,
    }


def test_common_p_summary_hypothesis(tmp_path: Path):
    nash = tmp_path / 'nash.csv'
    ppo = tmp_path / 'ppo.csv'
    _write(nash, [_row(0, -100, 3, 10), _row(1, -120, 4, 11)])
    _write(ppo, [_row(0, -90, 5, 9), _row(1, -110, 6, 10)])
    result = summarize_pair(nash, ppo, 10)
    assert result['P'] == 10.0
    assert result['ppo_total_reward'] > result['nash_total_reward']
    assert result['ppo_budget_violations'] > result['nash_budget_violations']
    assert result['hypothesis_weak'] is True
    assert result['hypothesis_strong'] is True
    assert result['weak_hypothesis_episode_rate'] == 1.0
