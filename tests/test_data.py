import json
import subprocess
import sys

import numpy as np
import pytest

from neuralnet.data import generate_dataset, load_dataset, save_dataset, split_indices, symmetry_key


def test_symmetry_groups_cannot_leak_between_train_and_test():
    inputs, _ = generate_dataset()
    train, test = split_indices(inputs)
    assert len(train) + len(test) == len(inputs)
    assert not set(train) & set(test)
    assert not ({symmetry_key(inputs[i]) for i in train} & {symmetry_key(inputs[i]) for i in test})
    again = split_indices(inputs)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(test, again[1])


def test_round_trip(tmp_path):
    inputs, targets = generate_dataset()
    path = tmp_path / 'moves.json'
    save_dataset(path, inputs, targets)
    loaded = load_dataset(path)
    np.testing.assert_array_equal(inputs, loaded[0])
    np.testing.assert_array_equal(targets, loaded[1])


@pytest.mark.parametrize(
    'payload',
    [
        {'version': 2},
        {'version': 1, 'inputs': [[0] * 9], 'targets': [[1] * 9]},
        {'version': 1, 'inputs': [[1] * 9], 'targets': [[0] * 9]},
        {'version': 1, 'inputs': [], 'targets': []},
    ],
)
def test_rejects_invalid_data(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_imports_have_no_side_effects(tmp_path):
    # Import from an empty cwd: no training, printing, or data-file dependency.
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import neuralnet, neuralnet.cli, neuralnet.data, neuralnet.experiment, '
            'neuralnet.game, neuralnet.network, neuralnet.players',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ''
    assert result.stderr == ''
    assert not list(tmp_path.iterdir())
