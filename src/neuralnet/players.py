"""Legal-move policies and a quiet, deterministic-with-a-seed game runner."""

from collections.abc import Callable
from fractions import Fraction
from functools import cache

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


@cache
def random_aware_win_probability(board: Board, turn: int, player: int) -> Fraction:
    """Exact win probability for a safe policy against a uniformly random opponent.

    On the player's turns, maximize win probability only among minimax-optimal
    moves. On the opponent's turns, average over every legal move. Draws and losses
    contribute zero to the win probability; worst-case safety is a separate,
    higher-priority constraint. Fractions keep equal expectations exactly equal.
    """
    if turn not in (-1, 1) or player not in (-1, 1):
        raise ValueError('player signs must be -1 or 1')
    won = winner(board)
    if won:
        return Fraction(int(won == player))
    moves = legal_moves(board)
    if not moves:
        return Fraction(0)
    if turn == player:
        return max(probability for _, probability in _random_aware_move_values(board, player))
    return sum(
        (random_aware_win_probability(play(board, move, turn), -turn, player) for move in moves),
        Fraction(0),
    ) / len(moves)


@cache
def _random_aware_move_values(board: Board, sign: int) -> tuple[tuple[int, Fraction], ...]:
    values = move_values(board, sign)
    if not values:
        return ()
    best = max(values.values())
    return tuple(
        (move, random_aware_win_probability(play(board, move, sign), -sign, sign))
        for move, value in values.items()
        if value == best
    )


def random_aware_minimax_move(board: Board, sign: int, rng: np.random.Generator) -> int:
    """Preserve the best worst-case result, then maximize wins against random play."""
    values = _random_aware_move_values(board, sign)
    if not values:
        raise ValueError('no legal moves remain')
    best = max(probability for _, probability in values)
    return int(rng.choice([move for move, probability in values if probability == best]))


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
