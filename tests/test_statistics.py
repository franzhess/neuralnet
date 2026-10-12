import pytest

from neuralnet.statistics import MATCHUPS, summarize_runs, wilson_interval


def test_wilson_interval_matches_known_example_and_has_nonzero_zero_event_bound():
    lower, upper = wilson_interval(44, 200)
    assert lower == pytest.approx(0.1682, abs=0.0001)
    assert upper == pytest.approx(0.2824, abs=0.0001)
    lower, upper = wilson_interval(0, 1000)
    assert lower == pytest.approx(0, abs=1e-15)
    assert 0 < upper < 0.004
    assert wilson_interval(1000, 1000)[1] == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_cohort_intervals_resample_networks_not_individual_games():
    def run(wins):
        return {
            'evaluation': {
                'games_per_matchup': 1000,
                'held_out_loss': 0.2,
                'neural_optimal_move_agreement': 0.9,
                'random_optimal_move_agreement': 0.6,
                **{
                    matchup: {'wins': wins, 'draws': 0, 'losses': 1000 - wins}
                    for matchup in MATCHUPS
                },
            }
        }

    summary = summarize_runs([run(0), run(1000)])
    wins = summary['matchups']['neural_vs_random']['rates']['wins']
    assert wins['mean'] == 0.5
    assert wins['std'] == pytest.approx(2**-0.5)
    assert wins['bootstrap_mean_ci95'] == [0, 1]
    single = summarize_runs([run(0)])
    assert single['matchups']['neural_vs_random']['rates']['wins']['bootstrap_mean_ci95'] is None
    with pytest.raises(ValueError):
        summarize_runs([])
