import numpy as np
import pytest

from neuralnet import ExperimentConfig
from neuralnet.data import generate_dataset
from neuralnet.experiment import agreement, evaluate, run_experiment, train_model
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
    assert report['random_aware_minimax_vs_random']['losses'] == 0
    assert report['random_aware_minimax_vs_minimax'] == {'wins': 0, 'draws': 4, 'losses': 0}
    assert report['neural_vs_random_aware_minimax']['wins'] == 0
    assert report['exact_baselines']['random_aware_minimax_vs_random']['balanced']['fraction'] == (
        '115589/120960'
    )
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
    del model.metadata['held_out_indices']
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
    assert first.metadata['held_out_indices'] == second.metadata['held_out_indices']


def test_cohort_has_distinct_seeds_one_split_and_repeatable_statistics():
    inputs, targets = generate_dataset()
    config = ExperimentConfig(epochs=1, hidden_nodes=8, networks=2, games=4)
    models, report = run_experiment(inputs, targets, config=config)
    again_models, again = run_experiment(inputs, targets, config=config)
    assert report == again
    assert report['schema_version'] == 3
    assert (
        report['summary']['exact_baselines'] == report['runs'][0]['evaluation']['exact_baselines']
    )
    assert [run['training']['seed'] for run in report['runs']] == [42, 43]
    assert [run['evaluation']['evaluation_seed'] for run in report['runs']] == [123, 124]
    assert (
        report['runs'][0]['training']['held_out_indices']
        == report['runs'][1]['training']['held_out_indices']
    )
    assert not np.array_equal(models[0].input_weights, models[1].input_weights)
    for model, other in zip(models, again_models, strict=True):
        np.testing.assert_array_equal(model.input_weights, other.input_weights)
    for matchup, summary in report['summary']['matchups'].items():
        assert summary['games'] == sum(summary['counts'].values()) == 8
        for run in report['runs']:
            roles = run['evaluation']['matchup_statistics'][matchup]['by_role']
            assert roles['x']['games'] == roles['o']['games'] == 2


@pytest.mark.parametrize('indices', [[], [0, 0], [-1], [999999], [0.5]])
def test_evaluation_rejects_invalid_saved_holdout_indices(indices):
    inputs, targets = generate_dataset()
    model = train_model(inputs, targets, config=ExperimentConfig(epochs=0, games=4))
    model.metadata['held_out_indices'] = indices
    with pytest.raises(ValueError, match='held-out indices'):
        evaluate(model, inputs, targets)


def test_agreement_counts_all_optimal_moves_not_one_teacher_choice():
    model = NeuralNetwork(9, 2, 9)
    model.forward = lambda inputs: np.array([[10, 10, 0, 0, 0, 0, 0, 0, 0]])
    inputs = np.zeros((1, 9))
    targets = np.array([[1, 1, -1, -1, -1, -1, -1, -1, -1]])
    result = agreement(model, inputs, targets)
    assert result['neural_optimal_move_agreement'] == 1
    assert result['random_optimal_move_agreement'] == pytest.approx(2 / 9)
