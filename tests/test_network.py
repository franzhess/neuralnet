import numpy as np
import pytest

from neuralnet.network import NeuralNetwork, Sigmoid, Tanh


@pytest.mark.parametrize('batch_size', [1, 2, 3, 8])
def test_batches_include_every_sample(batch_size, monkeypatch):
    network = NeuralNetwork(1, 2, 1)
    seen = []
    monkeypatch.setattr(network, 'backward', lambda targets: seen.extend(targets[:, 0].tolist()))
    data = np.arange(5).reshape(-1, 1)
    network.train(data, data, 1, batch_size=batch_size, shuffle=False)
    assert seen == list(range(5))


@pytest.mark.parametrize('activation', [Sigmoid(), Tanh()])
def test_full_backpropagation_matches_numerical_gradient(activation):
    model = NeuralNetwork(2, 3, 2, activation, seed=7)
    inputs = np.array([[0.2, -0.4], [0.7, 0.1]])
    targets = np.array([[0.1, 0.8], [-0.3, 0.5]])
    model.forward(inputs)
    gradients = model.gradients(targets)
    epsilon = 1e-6
    for weights, analytical in zip(
        (model.input_weights, model.output_weights), gradients, strict=True
    ):
        numerical = np.zeros_like(weights)
        for index in np.ndindex(weights.shape):
            original = weights[index]
            weights[index] = original + epsilon
            plus = model.loss(inputs, targets)
            weights[index] = original - epsilon
            minus = model.loss(inputs, targets)
            weights[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(analytical, numerical, atol=1e-9, rtol=1e-5)


def test_seeded_training_is_reproducible_and_does_not_change_global_rng():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    models = [NeuralNetwork(2, 4, 1, seed=42) for _ in range(2)]
    histories = [model.train(inputs, targets, 100, batch_size=3) for model in models]
    np.testing.assert_array_equal(histories[0], histories[1])
    np.testing.assert_array_equal(expected, np.random.random(3))


def test_xor_learns():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    model = NeuralNetwork(2, 4, 1, seed=42, learning_rate=1.0)
    initial = model.loss(inputs, targets)
    model.train(inputs, targets, 5000)
    assert model.loss(inputs, targets) < initial / 10
    np.testing.assert_array_equal(model.forward(inputs) > 0.5, targets.astype(bool))


def test_model_round_trip(tmp_path):
    model = NeuralNetwork(2, 3, 1, Tanh(), seed=5)
    inputs = np.array([[0.2, 0.7]])
    path = tmp_path / 'model.npz'
    model.save(path)
    loaded = NeuralNetwork.load(path)
    np.testing.assert_array_equal(model.forward(inputs), loaded.forward(inputs))


def test_loads_previous_activation_spelling(tmp_path):
    model = NeuralNetwork(2, 3, 1, Tanh(), seed=5)
    path = tmp_path / 'legacy.npz'
    np.savez(
        path,
        version=1,
        input_weights=model.input_weights,
        output_weights=model.output_weights,
        activation='TanH',
        learning_rate=0.1,
    )
    loaded = NeuralNetwork.load(path)
    inputs = np.array([[0.2, 0.7]])
    np.testing.assert_array_equal(model.forward(inputs), loaded.forward(inputs))


@pytest.mark.parametrize(
    'kwargs',
    [
        {'learning_rate': 0},
        {'learning_rate': float('nan')},
        {'input_nodes': 0},
    ],
)
def test_invalid_constructor(kwargs):
    arguments = dict(input_nodes=2, hidden_nodes=3, output_nodes=1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        NeuralNetwork(**arguments)


def test_invalid_training_inputs():
    model = NeuralNetwork(2, 3, 1)
    inputs, targets = np.ones((2, 2)), np.ones((2, 1))
    for bad in (np.ones((2, 3)), np.empty((0, 2)), np.array([[np.nan, 0]])):
        with pytest.raises(ValueError):
            model.forward(bad)
    for kwargs in ({'epochs': -1}, {'epochs': 1, 'batch_size': 0}):
        with pytest.raises(ValueError):
            model.train(inputs, targets, **kwargs)
    with pytest.raises(ValueError):
        model.train(inputs, np.ones((3, 1)), 1)
    with pytest.raises(ValueError):
        model.backward(targets)
