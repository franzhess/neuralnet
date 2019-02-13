from functools import cache

import pytest

from neuralnet.game import (
    EMPTY,
    LINES,
    legal_moves,
    minimax,
    move_values,
    play,
    positions,
    winner,
)


@pytest.mark.parametrize('line', LINES)
@pytest.mark.parametrize('sign', [-1, 1])
def test_every_winning_line(line, sign):
    board = tuple(sign if i in line else 0 for i in range(9))
    assert winner(board) == sign
    assert legal_moves(board) == ()


def test_draw_and_illegal_moves():
    board = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    assert winner(board) == 0
    assert legal_moves(board) == ()
    assert minimax(board, 1) == 0
    with pytest.raises(ValueError):
        play(board, 0, 1)
    for move, sign in ((-1, 1), (9, 1), (0, 0)):
        with pytest.raises(ValueError):
            play(EMPTY, move, sign)
    assert EMPTY == (0,) * 9


def test_teacher_wins_immediately_and_blocks():
    win = (1, 1, 0, -1, -1, 0, 0, 0, 0)
    block = (-1, -1, 0, 1, 0, 0, 0, 1, 0)
    assert move_values(win, 1)[2] == 1
    values = move_values(block, 1)
    assert values[2] == max(values.values())
    assert all(value == -1 for move, value in values.items() if move != 2)


def test_reachable_positions_and_bellman_equation():
    assert len(positions()) == 4520
    assert minimax(EMPTY, 1) == 0
    for board, sign in positions():
        assert winner(board) == 0
        assert board.count(1) - board.count(-1) == (0 if sign == 1 else 1)
        values = move_values(board, sign)
        assert minimax(board, sign) == max(values.values())
        assert set(values.values()) <= {-1, 0, 1}


@pytest.mark.parametrize('teacher_sign', [-1, 1])
def test_minimax_cannot_lose_against_any_opponent_continuation(teacher_sign):
    @cache
    def outcomes(board, sign):
        if not legal_moves(board):
            return frozenset((winner(board) * teacher_sign,))
        values = move_values(board, sign)
        moves = list(values)
        if sign == teacher_sign:
            moves = [move for move in moves if values[move] == max(values.values())]
        return frozenset(
            value for move in moves for value in outcomes(play(board, move, sign), -sign)
        )

    assert -1 not in outcomes(EMPTY, 1)
