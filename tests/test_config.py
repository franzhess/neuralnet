from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from neuralnet import ExperimentConfig


def test_config_is_immutable_and_can_be_replaced_explicitly():
    config = ExperimentConfig()
    with pytest.raises(FrozenInstanceError):
        config.epochs = 1
    updated = replace(config, epochs=2, games=4)
    assert updated.epochs == 2
    assert updated.games == 4
    assert config == ExperimentConfig()


def test_config_round_trips_with_unrelated_checkpoint_metadata():
    config = ExperimentConfig(seed=7, epochs=2, games=6, evaluation_seed=10)
    metadata = {**asdict(config), 'dataset_sha256': 'example', 'final_train_loss': 0.1}
    assert ExperimentConfig.from_metadata(metadata) == config
    assert ExperimentConfig.from_metadata({}) == ExperimentConfig()


@pytest.mark.parametrize(
    'settings',
    [
        {'seed': -1},
        {'evaluation_seed': -1},
        {'epochs': -1},
        {'epochs': 1.5},
        {'epochs': True},
        {'hidden_nodes': 0},
        {'batch_size': 0},
        {'games': 0},
        {'games': 1},
        {'games': 3},
        {'games': 2.5},
        {'test_fraction': 0},
        {'test_fraction': 1},
        {'test_fraction': float('nan')},
        {'test_fraction': '0.2'},
        {'learning_rate': 0},
        {'learning_rate': float('inf')},
        {'learning_rate': True},
    ],
)
def test_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_zero_epochs_supports_an_untrained_baseline():
    assert ExperimentConfig(epochs=0).epochs == 0
