import numpy as np
import pytest

from neuralnet import ExperimentConfig
from neuralnet.data import generate_dataset
from neuralnet.experiment import agreement, evaluate, train_model
from neuralnet.network import NeuralNetwork


def test_training_checkpoint_and_evaluation_are_repeatable(tmp_path):
    inputs, targets = generate_dataset()
    config = ExperimentConfig(epochs=2, hidden_nodes=8, games=4)
    model = train_model(inputs, targets, config=config)
    assert model.metadata['final_train_loss'] < model.metadata['initial_train_loss']
    path = tmp_path / 'model.npz'
    model.save(path)
    loaded = NeuralNetwork.load(path)
    assert ExperimentConfig.from_metadata(loaded.metadata) == config
    report = evaluate(loaded, inputs, targets)
    assert report == evaluate(model, inputs, targets)
    assert report['minimax_vs_minimax'] == {'wins': 0, 'draws': 4, 'losses': 0}
    assert report['minimax_vs_random']['losses'] == 0
    for name, results in report.items():
        if '_vs_' in name:
            assert sum(results.values()) == 4
    with pytest.raises(ValueError, match='does not match'):
        evaluate(model, inputs[::-1], targets[::-1])


def test_evaluation_override_cannot_change_recorded_training_split():
    inputs, targets = generate_dataset()
    training = ExperimentConfig(epochs=1, hidden_nodes=8, seed=7, test_fraction=0.3, games=4)
    model = train_model(inputs, targets, config=training)
    override = ExperimentConfig(seed=99, test_fraction=0.1, games=6, evaluation_seed=10)
    report = evaluate(model, inputs, targets, config=override)
    assert report['held_out_samples'] == model.metadata['test_samples']
    assert report['held_out_loss'] == model.metadata['held_out_loss']
    assert report['games_per_matchup'] == 6
    assert report['evaluation_seed'] == 10


def test_older_checkpoints_preserve_legacy_evaluation_defaults():
    inputs, targets = generate_dataset()
    model = train_model(
        inputs, targets, config=ExperimentConfig(epochs=0, hidden_nodes=8, seed=7, split_seed=7)
    )
    del model.metadata['games']
    del model.metadata['evaluation_seed']
    del model.metadata['split_seed']
    del model.metadata['networks']
    report = evaluate(model, inputs, targets)
    assert report['games_per_matchup'] == 200
    assert report['held_out_samples'] == model.metadata['test_samples']
    assert report['evaluation_seed'] == ExperimentConfig().evaluation_seed


def test_training_seed_does_not_change_the_held_out_split():
    from dataclasses import replace

    inputs, targets = generate_dataset()
    config = ExperimentConfig(epochs=0, hidden_nodes=8, games=4)
    first = train_model(inputs, targets, config=config)
    second = train_model(inputs, targets, config=replace(config, seed=99))
    assert first.metadata['test_samples'] == second.metadata['test_samples']
    assert first.metadata['split_seed'] == second.metadata['split_seed']
    assert not np.array_equal(first.input_weights, second.input_weights)


def test_agreement_counts_all_optimal_moves_not_one_teacher_choice():
    model = NeuralNetwork(9, 2, 9)
    model.forward = lambda inputs: np.array([[10, 10, 0, 0, 0, 0, 0, 0, 0]])
    inputs = np.zeros((1, 9))
    targets = np.array([[1, 1, -1, -1, -1, -1, -1, -1, -1]])
    result = agreement(model, inputs, targets)
    assert result['neural_optimal_move_agreement'] == 1
    assert result['random_optimal_move_agreement'] == pytest.approx(2 / 9)
