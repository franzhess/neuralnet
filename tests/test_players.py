import numpy as np
import pytest

from neuralnet.network import NeuralNetwork, Tanh
from neuralnet.players import minimax_move, neural_policy, play_game


def test_neural_player_masks_occupied_squares_and_normalizes_perspective():
    model = NeuralNetwork(9, 4, 9, Tanh(), seed=1)
    captured = []

    def prediction(inputs):
        captured.append(inputs.copy())
        return np.array([[100, 2, 1, 1, 1, 1, 1, 1, 1]])

    model.forward = prediction
    board = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert neural_policy(model)(board, -1, np.random.default_rng(0)) == 1
    np.testing.assert_array_equal(captured[0], -np.array(board).reshape(1, 9))
    with pytest.raises(ValueError):
        neural_policy(model)((1,) * 9, 1, np.random.default_rng(0))


def test_seeded_minimax_game_is_a_draw():
    assert play_game(minimax_move, minimax_move, np.random.default_rng(42)) == 0


def test_game_rejects_illegal_player_moves():
    def invalid_policy(board, sign, rng):
        return 9

    with pytest.raises(ValueError, match='illegal move'):
        play_game(invalid_policy, minimax_move, np.random.default_rng(42))
