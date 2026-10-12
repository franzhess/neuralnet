"""Descriptive game intervals and network-level uncertainty without SciPy."""

from math import sqrt
from statistics import NormalDist

import numpy as np

OUTCOMES = ('wins', 'draws', 'losses')
MATCHUPS = (
    'neural_vs_random',
    'neural_vs_minimax',
    'random_vs_random',
    'minimax_vs_random',
    'minimax_vs_minimax',
)


def wilson_interval(successes: int, trials: int) -> list[float]:
    """Approximate two-sided 95% binomial interval, including zero/all successes."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError('successes must be between zero and a positive trial count')
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z * sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def outcome_statistics(counts: dict[str, int]) -> dict:
    """Marginal outcome intervals for a fixed policy, not a training-population CI."""
    games = sum(counts.values())
    return {
        'games': games,
        'rates': {
            outcome: {
                'estimate': counts[outcome] / games,
                'wilson_ci95': wilson_interval(counts[outcome], games),
            }
            for outcome in OUTCOMES
        },
    }


def summarize_runs(
    runs: list[dict], *, bootstrap_seed: int = 0, bootstrap_resamples: int = 10000
) -> dict:
    """Summarize equal-sized runs, resampling whole networks rather than games.

    Percentile-bootstrap intervals describe uncertainty in the mean over training
    and gameplay randomness, conditional on this fixed dataset and held-out split.
    They are approximate, not simultaneous intervals or significance tests.
    """
    if not runs or bootstrap_resamples < 1:
        raise ValueError('need runs and a positive bootstrap resample count')
    evaluations = [run['evaluation'] for run in runs]
    games = evaluations[0]['games_per_matchup']
    if any(evaluation['games_per_matchup'] != games for evaluation in evaluations):
        raise ValueError('all runs must use the same game count')
    indices = np.random.default_rng(bootstrap_seed).integers(
        0, len(runs), size=(bootstrap_resamples, len(runs))
    )

    def distribution(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        interval = (
            np.quantile(array[indices].mean(axis=1), [0.025, 0.975]).tolist()
            if len(runs) > 1
            else None
        )
        return {
            'mean': float(array.mean()),
            'std': float(array.std(ddof=1)) if len(runs) > 1 else 0.0,
            'min': float(array.min()),
            'max': float(array.max()),
            'bootstrap_mean_ci95': interval,
        }

    metrics = ('held_out_loss', 'neural_optimal_move_agreement', 'random_optimal_move_agreement')
    matchups = {}
    for matchup in MATCHUPS:
        counts = {
            outcome: sum(evaluation[matchup][outcome] for evaluation in evaluations)
            for outcome in OUTCOMES
        }
        matchups[matchup] = {
            'games': games * len(runs),
            'counts': counts,
            'rates': {
                outcome: distribution(
                    [evaluation[matchup][outcome] / games for evaluation in evaluations]
                )
                for outcome in OUTCOMES
            },
        }
    return {
        'networks': len(runs),
        'games_per_network_per_matchup': games,
        'bootstrap_seed': bootstrap_seed,
        'bootstrap_resamples': bootstrap_resamples,
        'interval_method': '95% percentile bootstrap of whole network runs',
        'metrics': {
            metric: distribution([evaluation[metric] for evaluation in evaluations])
            for metric in metrics
        },
        'matchups': matchups,
    }
