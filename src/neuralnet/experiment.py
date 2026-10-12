"""Reproducible train/held-out evaluation and game-level baseline comparisons."""

import platform
from dataclasses import asdict

import numpy as np

from .config import ExperimentConfig
from .data import Array, fingerprint, split_indices
from .network import NeuralNetwork, Tanh
from .players import Policy, minimax_move, neural_policy, play_game, random_move


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
    if games < 2 or games % 2:
        raise ValueError('games must be a positive even number of at least two')
    rng = np.random.default_rng(seed)
    results = {'wins': 0, 'draws': 0, 'losses': 0}
    for index in range(games):
        sign = 1 if index % 2 == 0 else -1
        outcome = (
            play_game(player, opponent, rng) if sign == 1 else play_game(opponent, player, rng)
        ) * sign
        results[{1: 'wins', 0: 'draws', -1: 'losses'}[outcome]] += 1
    return results


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
    _, test = split_indices(
        inputs, seed=training_config.split_seed, test_fraction=training_config.test_fraction
    )
    neural = neural_policy(model)
    match_settings = {'games': config.games, 'seed': config.evaluation_seed}
    return {
        'evaluation_seed': config.evaluation_seed,
        'games_per_matchup': config.games,
        'held_out_samples': len(test),
        'held_out_loss': model.loss(inputs[test], targets[test]),
        **agreement(model, inputs[test], targets[test]),
        'neural_vs_random': tournament(neural, random_move, **match_settings),
        'neural_vs_minimax': tournament(neural, minimax_move, **match_settings),
        'random_vs_random': tournament(random_move, random_move, **match_settings),
        'minimax_vs_random': tournament(minimax_move, random_move, **match_settings),
        'minimax_vs_minimax': tournament(minimax_move, minimax_move, **match_settings),
    }
