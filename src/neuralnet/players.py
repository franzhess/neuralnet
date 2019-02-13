"""Legal-move policies and a quiet, deterministic-with-a-seed game runner."""

from collections.abc import Callable

import numpy as np

from .game import EMPTY, Board, legal_moves, move_values, play, winner
from .network import NeuralNetwork

Policy = Callable[[Board, int, np.random.Generator], int]


def random_move(board: Board, sign: int, rng: np.random.Generator) -> int:
    moves = legal_moves(board)
    if not moves:
        raise ValueError('no legal moves remain')
    return int(rng.choice(moves))


def minimax_move(board: Board, sign: int, rng: np.random.Generator) -> int:
    values = move_values(board, sign)
    if not values:
        raise ValueError('no legal moves remain')
    best = max(values.values())
    return int(rng.choice([move for move, value in values.items() if value == best]))


def neural_policy(model: NeuralNetwork) -> Policy:
    """Normalize to the mover's perspective; mask occupied squares before argmax."""
    if model.input_weights.shape[0] != 10 or model.output_weights.shape[1] != 9:
        raise ValueError('tic-tac-toe models need nine inputs and nine outputs')

    def choose(board: Board, sign: int, rng: np.random.Generator) -> int:
        moves = legal_moves(board)
        if not moves:
            raise ValueError('no legal moves remain')
        inputs = np.asarray(board, dtype=np.float64).reshape(1, 9) * sign
        scores = model.forward(inputs)[0]
        best = max(scores[move] for move in moves)
        return int(rng.choice([move for move in moves if scores[move] == best]))

    return choose


def play_game(x: Policy, o: Policy, rng: np.random.Generator) -> int:
    """X starts; return 1/-1 for the winner or 0 for a draw."""
    board, sign = EMPTY, 1
    while legal_moves(board):
        policy = x if sign == 1 else o
        board = play(board, policy(board, sign, rng), sign)
        sign = -sign
    return winner(board)
