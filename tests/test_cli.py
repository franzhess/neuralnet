import json
import subprocess
import sys

import numpy as np
import pytest

from neuralnet import ExperimentConfig, NeuralNetwork, cli


def test_cli_training_defaults_match_python_api(monkeypatch, tmp_path):
    configs = []

    def fake_train(inputs, targets, *, config):
        configs.append(config)
        return NeuralNetwork(9, 2, 9, seed=0)

    monkeypatch.setattr(cli, 'load_dataset', lambda path: (np.zeros((1, 9)), np.zeros((1, 9))))
    monkeypatch.setattr(cli, 'train_model', fake_train)
    cli.main(['train', 'unused.json', str(tmp_path / 'model.npz')])
    assert configs == [ExperimentConfig()]


@pytest.mark.parametrize(
    'arguments',
    [
        ['--games', '3'],
        ['--epochs', '-1'],
        ['--learning-rate', 'nan'],
        ['--seed', '-1'],
    ],
)
def test_cli_rejects_invalid_configuration_before_creating_artifacts(arguments, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['demo', '--output-dir', str(tmp_path), *arguments])
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_cli_end_to_end_from_outside_source_tree(tmp_path):
    def run(*arguments):
        return subprocess.run(
            [sys.executable, '-m', 'neuralnet', *map(str, arguments)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    run('demo', '--epochs', 2, '--hidden-nodes', 8, '--games', 4, '--output-dir', tmp_path)
    report = json.loads((tmp_path / 'experiment.json').read_text())
    restored = run('evaluate', tmp_path / 'moves.json', tmp_path / 'model.npz')
    assert json.loads(restored.stdout) == report['evaluation']
    generated = tmp_path / 'separate.json'
    checkpoint = tmp_path / 'separate.npz'
    run('generate', generated)
    run('train', generated, checkpoint, '--epochs', 2, '--hidden-nodes', 8)
    result = run('evaluate', generated, checkpoint, '--games', 4)
    assert json.loads(result.stdout) == report['evaluation']
