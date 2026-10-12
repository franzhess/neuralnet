from fractions import Fraction
from functools import cache

import numpy as np
import pytest

from neuralnet.game import EMPTY, legal_moves, move_values, play, positions, winner
from neuralnet.players import (
    minimax_move,
    play_game,
    random_aware_minimax_move,
    random_aware_win_probability,
)


class RecordingRng:
    """Expose all equally preferred choices, not just one sampled continuation."""

    def choice(self, moves):
        self.moves = tuple(moves)
        return self.moves[0]


def test_every_preferred_move_preserves_minimax_value_in_every_reachable_position():
    for board, sign in positions():
        rng = RecordingRng()
        random_aware_minimax_move(board, sign, rng)
        values = move_values(board, sign)
        assert all(values[move] == max(values.values()) for move in rng.moves)
        expectations = {
            move: random_aware_win_probability(play(board, move, sign), -sign, sign)
            for move, value in values.items()
            if value == max(values.values())
        }
        assert set(rng.moves) == {
            move
            for move, probability in expectations.items()
            if probability == max(expectations.values())
        }


@pytest.mark.parametrize('player', [-1, 1])
def test_actual_policy_never_loses_against_any_opponent_continuation(player):
    @cache
    def outcomes(board, turn):
        moves = legal_moves(board)
        if not moves:
            return frozenset((winner(board) * player,))
        if turn == player:
            rng = RecordingRng()
            random_aware_minimax_move(board, turn, rng)
            moves = rng.moves
        return frozenset(
            outcome for move in moves for outcome in outcomes(play(board, move, turn), -turn)
        )

    assert -1 not in outcomes(EMPTY, 1)


@pytest.mark.parametrize('player', [-1, 1])
def test_exact_probability_matches_independent_policy_tree_and_improves_uniform_ties(player):
    @cache
    def probability(board, turn, aware):
        won = winner(board)
        if won:
            return Fraction(int(won == player))
        moves = legal_moves(board)
        if not moves:
            return Fraction(0)
        if turn == player:
            if aware:
                rng = RecordingRng()
                random_aware_minimax_move(board, turn, rng)
                moves = rng.moves
            else:
                values = move_values(board, turn)
                moves = tuple(
                    move for move, value in values.items() if value == max(values.values())
                )
        return sum(
            (probability(play(board, move, turn), -turn, aware) for move in moves), Fraction(0)
        ) / len(moves)

    exact = random_aware_win_probability(EMPTY, 1, player)
    assert exact == (Fraction(191, 192) if player == 1 else Fraction(866, 945))
    assert exact == probability(EMPTY, 1, True)
    assert exact > probability(EMPTY, 1, False)


def test_reproducible_draws_against_perfect_play_and_terminal_rejection():
    for x, o in (
        (random_aware_minimax_move, minimax_move),
        (minimax_move, random_aware_minimax_move),
    ):
        assert play_game(x, o, np.random.default_rng(42)) == 0
    with pytest.raises(ValueError, match='no legal moves'):
        random_aware_minimax_move((1,) * 9, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match='player signs'):
        random_aware_win_probability(EMPTY, 0, 1)
