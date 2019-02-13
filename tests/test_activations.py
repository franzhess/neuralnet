import numpy as np
import pytest

from neuralnet import Sigmoid, Tanh


@pytest.mark.parametrize('activation', [Sigmoid(), Tanh()])
def test_activation_derivatives_match_finite_differences(activation):
    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    epsilon = 1e-6
    numerical = (activation(values + epsilon) - activation(values - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(activation.derivative(activation(values)), numerical, atol=1e-9)
