"""Exact tic-tac-toe rules and a cached negamax (minimax) teacher.

Boards are immutable row-major tuples: X=1, O=-1, empty=0.
Scores are outcomes from the player-to-move's perspective, not heuristics.
"""

from functools import cache, lru_cache

Board = tuple[int, ...]
EMPTY: Board = (0,) * 9
LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def winner(board: Board) -> int:
    """Return the winning sign, or zero if neither player has won."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal_moves(board: Board) -> tuple[int, ...]:
    """Terminal boards have no legal continuations."""
    if winner(board):
        return ()
    return tuple(i for i, value in enumerate(board) if value == 0)


def play(board: Board, move: int, sign: int) -> Board:
    """Apply a legal move without mutating the original board."""
    if sign not in (-1, 1) or move not in legal_moves(board):
        raise ValueError('illegal move or player sign')
    return board[:move] + (sign,) + board[move + 1 :]


@cache
def minimax(board: Board, sign: int) -> int:
    """Return -1/0/1 for loss/draw/win with perfect play."""
    won = winner(board)
    if won:
        return won * sign
    moves = legal_moves(board)
    if not moves:
        return 0
    return max(-minimax(play(board, move, sign), -sign) for move in moves)


def move_values(board: Board, sign: int) -> dict[int, int]:
    """Evaluate each legal move; occupied squares are excluded."""
    return {move: -minimax(play(board, move, sign), -sign) for move in legal_moves(board)}


@lru_cache(maxsize=1)
def positions() -> tuple[tuple[Board, int], ...]:
    """Enumerate reachable nonterminal states, starting with X.

    Stop at wins, so impossible post-win positions never enter the dataset.
    Transpositions are deduplicated independently of move order.
    """
    seen: set[tuple[Board, int]] = set()

    def visit(board: Board, sign: int) -> None:
        if not legal_moves(board) or (board, sign) in seen:
            return
        seen.add((board, sign))
        for move in legal_moves(board):
            visit(play(board, move, sign), -sign)

    visit(EMPTY, 1)
    return tuple(sorted(seen))
