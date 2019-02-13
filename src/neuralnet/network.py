"""One-hidden-layer network with explicit NumPy forward/backward propagation."""

import json
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class Activation(Protocol):
    def __call__(self, values: Array) -> Array: ...
    def derivative(self, output: Array) -> Array: ...


class Sigmoid:
    def __call__(self, values: Array) -> Array:
        return np.exp(-np.logaddexp(0, -values))

    def derivative(self, output: Array) -> Array:
        return output * (1 - output)


class Tanh:
    def __call__(self, values: Array) -> Array:
        return np.tanh(values)

    def derivative(self, output: Array) -> Array:
        return 1 - output**2


class NeuralNetwork:
    """Dense network with bias rows, mean squared-error loss and SGD.

    The loss is ``0.5 * sum((prediction - target)**2) / batch_size``.
    Derivatives consume activation outputs, not pre-activation values.
    Randomness is local to the model, so seeds do not alter global NumPy state.
    """

    def __init__(
        self,
        input_nodes: int,
        hidden_nodes: int,
        output_nodes: int,
        activation: Activation | None = None,
        *,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ):
        if any(not isinstance(n, int) or n < 1 for n in (input_nodes, hidden_nodes, output_nodes)):
            raise ValueError('layer sizes must be positive integers')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        self.rng = np.random.default_rng(seed)
        self.activation = activation if activation is not None else Sigmoid()
        self.learning_rate = learning_rate
        self.input_weights = self._init_layer(input_nodes, hidden_nodes)
        self.output_weights = self._init_layer(hidden_nodes, output_nodes)
        self._cache: tuple[Array, Array, Array] | None = None
        self.metadata: dict[str, Any] = {}

    def _init_layer(self, inputs: int, outputs: int) -> Array:
        # Xavier uniform initialization, with zero initial biases.
        bound = np.sqrt(6 / (inputs + outputs))
        weights = self.rng.uniform(-bound, bound, (inputs + 1, outputs))
        weights[-1] = 0
        return weights

    @staticmethod
    def add_bias(values: Array) -> Array:
        return np.column_stack((values, np.ones(len(values))))

    def _inputs(self, inputs: Array) -> Array:
        values = np.asarray(inputs, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.input_weights.shape[0] - 1
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError('inputs must be a nonempty finite matrix matching input size')
        return values

    def _targets(self, targets: Array, samples: int) -> Array:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (samples, self.output_weights.shape[1]) or not np.isfinite(values).all():
            raise ValueError('targets must be finite and match sample and output sizes')
        return values

    def forward(self, inputs: Array) -> Array:
        biased_inputs = self.add_bias(self._inputs(inputs))
        hidden = self.add_bias(self.activation(biased_inputs @ self.input_weights))
        output = self.activation(hidden @ self.output_weights)
        self._cache = biased_inputs, hidden, output
        return output

    def gradients(self, targets: Array) -> tuple[Array, Array]:
        """Compute gradients without changing weights; requires a forward pass."""
        if self._cache is None:
            raise ValueError('call forward before gradients or backward')
        inputs, hidden, output = self._cache
        targets = self._targets(targets, len(inputs))
        output_delta = (output - targets) * self.activation.derivative(output) / len(inputs)
        hidden_delta = (
            output_delta @ self.output_weights[:-1].T * self.activation.derivative(hidden[:, :-1])
        )
        return inputs.T @ hidden_delta, hidden.T @ output_delta

    def backward(self, targets: Array) -> None:
        input_gradient, output_gradient = self.gradients(targets)
        self.input_weights -= self.learning_rate * input_gradient
        self.output_weights -= self.learning_rate * output_gradient
        self._cache = None

    def loss(self, inputs: Array, targets: Array) -> float:
        prediction = self.forward(inputs)
        target = self._targets(targets, len(prediction))
        return float(0.5 * np.sum((prediction - target) ** 2) / len(prediction))

    def train(
        self,
        inputs: Array,
        targets: Array,
        epochs: int,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> list[float]:
        """Train on every sample each epoch, including the final partial batch."""
        inputs = self._inputs(inputs)
        targets = self._targets(targets, len(inputs))
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError('epochs must be a nonnegative integer')
        batch_size = len(inputs) if batch_size is None else batch_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        history = []
        for _ in range(epochs):
            indices = self.rng.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
            for start in range(0, len(inputs), batch_size):
                batch = indices[start : start + batch_size]
                self.forward(inputs[batch])
                self.backward(targets[batch])
            history.append(self.loss(inputs, targets))
        return history

    def save(self, path: str | Path) -> None:
        """Store weights and metadata as an NPZ archive (never Python pickle)."""
        if type(self.activation) not in (Sigmoid, Tanh):
            raise ValueError('only built-in activations can be saved')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            np.savez(
                handle,
                version=1,
                input_weights=self.input_weights,
                output_weights=self.output_weights,
                activation=type(self.activation).__name__,
                learning_rate=self.learning_rate,
                metadata=json.dumps(self.metadata, allow_nan=False),
            )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a prediction checkpoint; RNG state is not a training resume state."""
        try:
            with np.load(path, allow_pickle=False) as data:
                if data['version'].item() != 1:
                    raise ValueError('unsupported model version')
                first = np.asarray(data['input_weights'], dtype=np.float64)
                second = np.asarray(data['output_weights'], dtype=np.float64)
                # Accept the old spelling in existing version-1 checkpoints.
                activations = {'Sigmoid': Sigmoid, 'Tanh': Tanh, 'TanH': Tanh}
                activation = activations[data['activation'].item()]()
                rate = float(data['learning_rate'].item())
                metadata = json.loads(data['metadata'].item()) if 'metadata' in data else {}
                if not isinstance(metadata, dict):
                    raise ValueError('model metadata must be an object')
            if (
                first.ndim != 2
                or second.ndim != 2
                or second.shape[0] != first.shape[1] + 1
                or not np.isfinite(first).all()
                or not np.isfinite(second).all()
            ):
                raise ValueError('invalid weight matrices')
            model = cls(
                first.shape[0] - 1, first.shape[1], second.shape[1], activation, learning_rate=rate
            )
            model.input_weights = first.copy()
            model.output_weights = second.copy()
            model.metadata = metadata
            return model
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError('invalid model archive') from error
