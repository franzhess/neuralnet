"""Reproducible train/held-out evaluation and game-level baseline comparisons."""

import platform
from dataclasses import asdict, replace

import numpy as np

from .config import ExperimentConfig
from .data import Array, fingerprint, split_indices
from .game import EMPTY
from .network import NeuralNetwork, Tanh
from .players import (
    Policy,
    minimax_move,
    neural_policy,
    play_game,
    random_aware_minimax_move,
    random_aware_win_probability,
    random_move,
)
from .statistics import OUTCOMES, outcome_statistics, summarize_runs


def train_model(
    inputs: Array,
    targets: Array,
    *,
    config: ExperimentConfig | None = None,
) -> NeuralNetwork:
    """Train with shared settings and record the exact configuration in the model."""
    config = config if config is not None else ExperimentConfig()
    train, test = split_indices(inputs, seed=config.split_seed, test_fraction=config.test_fraction)
    model = NeuralNetwork(
        9, config.hidden_nodes, 9, Tanh(), seed=config.seed, learning_rate=config.learning_rate
    )
    initial_loss = model.loss(inputs[train], targets[train])
    history = model.train(
        inputs[train], targets[train], config.epochs, batch_size=config.batch_size
    )
    model.metadata = {
        **asdict(config),
        'dataset_sha256': fingerprint(inputs, targets),
        'train_samples': len(train),
        'test_samples': len(test),
        'held_out_indices': test.tolist(),
        'initial_train_loss': initial_loss,
        'final_train_loss': model.loss(inputs[train], targets[train]),
        'held_out_loss': model.loss(inputs[test], targets[test]),
        'training_loss_history': history,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
    }
    return model


def agreement(model: NeuralNetwork, inputs: Array, targets: Array) -> dict[str, float]:
    """Agreement with *any* optimal move; ties are scored as uniform choices."""
    predictions = model.forward(inputs)
    agreements, random_agreements = [], []
    for board, target, scores in zip(inputs, targets, predictions, strict=True):
        legal = board == 0
        optimal = legal & (target == np.max(target[legal]))
        chosen = legal & (scores == np.max(scores[legal]))
        agreements.append(float(np.sum(optimal & chosen) / np.sum(chosen)))
        random_agreements.append(float(np.sum(optimal) / np.sum(legal)))
    return {
        'neural_optimal_move_agreement': float(np.mean(agreements)),
        'random_optimal_move_agreement': float(np.mean(random_agreements)),
    }


def tournament(player: Policy, opponent: Policy, *, games: int, seed: int) -> dict[str, int]:
    """Alternate player roles equally, without training during evaluation."""
    counts, _ = _matchup(player, opponent, games=games, seed=seed)
    return counts


def _matchup(
    player: Policy, opponent: Policy, *, games: int, seed: int
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    if games < 2 or games % 2:
        raise ValueError('games must be a positive even number of at least two')
    rng = np.random.default_rng(seed)
    results = dict.fromkeys(OUTCOMES, 0)
    by_role = {role: dict.fromkeys(OUTCOMES, 0) for role in ('x', 'o')}
    for index in range(games):
        sign = 1 if index % 2 == 0 else -1
        outcome = (
            play_game(player, opponent, rng) if sign == 1 else play_game(opponent, player, rng)
        ) * sign
        label = {1: 'wins', 0: 'draws', -1: 'losses'}[outcome]
        results[label] += 1
        by_role['x' if sign == 1 else 'o'][label] += 1
    return results, by_role


def evaluate(
    model: NeuralNetwork, inputs: Array, targets: Array, *, config: ExperimentConfig | None = None
) -> dict:
    """Evaluate using the recorded training split and configurable game settings.

    Without an override, use the checkpoint's settings. An override changes only
    the game count and evaluation seed, never the training/test split.
    """
    metadata = model.metadata
    if metadata.get('dataset_sha256') != fingerprint(inputs, targets):
        raise ValueError('evaluation dataset does not match the training checkpoint')
    training_config = ExperimentConfig.from_metadata(metadata)
    config = config if config is not None else training_config
    if 'held_out_indices' in metadata:
        test = np.asarray(metadata['held_out_indices'])
        if (
            test.ndim != 1
            or not len(test)
            or not np.issubdtype(test.dtype, np.integer)
            or np.any(test < 0)
            or np.any(test >= len(inputs))
            or len(np.unique(test)) != len(test)
        ):
            raise ValueError('invalid recorded held-out indices')
    else:
        _, test = split_indices(
            inputs, seed=training_config.split_seed, test_fraction=training_config.test_fraction
        )
    neural = neural_policy(model)
    match_settings = {'games': config.games, 'seed': config.evaluation_seed}
    policies = {
        'neural_vs_random': (neural, random_move),
        'neural_vs_minimax': (neural, minimax_move),
        'random_vs_random': (random_move, random_move),
        'minimax_vs_random': (minimax_move, random_move),
        'minimax_vs_minimax': (minimax_move, minimax_move),
        'neural_vs_random_aware_minimax': (neural, random_aware_minimax_move),
        'random_aware_minimax_vs_random': (random_aware_minimax_move, random_move),
        'random_aware_minimax_vs_minimax': (random_aware_minimax_move, minimax_move),
    }
    results, statistics = {}, {}
    for name, (player, opponent) in policies.items():
        counts, by_role = _matchup(player, opponent, **match_settings)
        results[name] = counts
        statistics[name] = {
            'overall': outcome_statistics(counts),
            'by_role': {
                role: outcome_statistics(role_counts) for role, role_counts in by_role.items()
            },
        }
    x_probability = random_aware_win_probability(EMPTY, 1, 1)
    o_probability = random_aware_win_probability(EMPTY, 1, -1)
    exact = {
        role: {'fraction': str(probability), 'probability': float(probability)}
        for role, probability in (
            ('x', x_probability),
            ('o', o_probability),
            ('balanced', (x_probability + o_probability) / 2),
        )
    }
    return {
        'evaluation_seed': config.evaluation_seed,
        'games_per_matchup': config.games,
        'held_out_samples': len(test),
        'held_out_loss': model.loss(inputs[test], targets[test]),
        **agreement(model, inputs[test], targets[test]),
        **results,
        'matchup_statistics': statistics,
        'exact_baselines': {'random_aware_minimax_vs_random': exact},
    }


def run_experiment(
    inputs: Array, targets: Array, *, config: ExperimentConfig | None = None
) -> tuple[list[NeuralNetwork], dict]:
    """Train independent networks on one split, retaining every run, not just the best."""
    config = config if config is not None else ExperimentConfig()
    models, runs = [], []
    for index in range(config.networks):
        run_config = replace(
            config,
            seed=config.seed + index,
            evaluation_seed=config.evaluation_seed + index,
            networks=1,
        )
        model = train_model(inputs, targets, config=run_config)
        models.append(model)
        runs.append(
            {
                'network_index': index + 1,
                'training': model.metadata,
                'evaluation': evaluate(model, inputs, targets),
            }
        )
    summary = summarize_runs(runs)
    summary['exact_baselines'] = runs[0]['evaluation']['exact_baselines']
    return models, {
        'schema_version': 3,
        'config': asdict(config),
        'dataset_sha256': fingerprint(inputs, targets),
        'runs': runs,
        'summary': summary,
    }
