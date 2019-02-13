"""Versioned JSON training data; no executable input or implicit file access."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .game import move_values, positions

Array = NDArray[np.float64]


def generate_dataset() -> tuple[Array, Array]:
    """Encode boards relative to the player to move and exact move outcomes."""
    boards = sorted({tuple(value * sign for value in board) for board, sign in positions()})
    inputs = np.asarray(boards, dtype=np.float64)
    targets = np.full_like(inputs, -1)
    for row, board in enumerate(boards):
        for move, value in move_values(board, 1).items():
            targets[row, move] = value
    return inputs, targets


def save_dataset(path: str | Path, inputs: Array, targets: Array) -> None:
    """Write an explicit versioned format, creating the output directory."""
    validate_dataset(inputs, targets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({'version': 1, 'inputs': inputs.tolist(), 'targets': targets.tolist()}) + '\n',
        encoding='utf-8',
    )


def load_dataset(path: str | Path) -> tuple[Array, Array]:
    """Reject malformed, unreachable, duplicate, or incorrectly labelled data."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('unsupported dataset format')
    try:
        inputs = np.asarray(payload['inputs'], dtype=np.float64)
        targets = np.asarray(payload['targets'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('dataset requires numeric inputs and targets') from error
    validate_dataset(inputs, targets)
    return inputs, targets


def validate_dataset(inputs: Array, targets: Array) -> None:
    if inputs.ndim != 2 or inputs.shape[1] != 9 or not len(inputs):
        raise ValueError('inputs must have shape (samples, 9) and be nonempty')
    if targets.shape != inputs.shape or not np.isfinite(targets).all():
        raise ValueError('targets must be finite and match inputs')
    if not np.isin(inputs, (-1, 0, 1)).all():
        raise ValueError('boards must contain only -1, 0 and 1')
    reachable = {tuple(value * sign for value in board) for board, sign in positions()}
    seen = set()
    for raw, target in zip(inputs, targets, strict=True):
        board = tuple(int(value) for value in raw)
        if board not in reachable or board in seen:
            raise ValueError('boards must be reachable, nonterminal and unique')
        seen.add(board)
        expected = np.full(9, -1.0)
        for move, value in move_values(board, 1).items():
            expected[move] = value
        if not np.array_equal(target, expected):
            raise ValueError('targets must match exact minimax outcomes')


def symmetry_key(board: Array) -> tuple[int, ...]:
    """Canonicalize under all eight rotations/reflections of the square."""
    square = np.asarray(board).reshape(3, 3)
    variants = (
        variant
        for k in range(4)
        for variant in (np.rot90(square, k), np.fliplr(np.rot90(square, k)))
    )
    return min(tuple(int(value) for value in variant.flat) for variant in variants)


def split_indices(
    inputs: Array, *, seed: int = 42, test_fraction: float = 0.2
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split symmetry groups, not rows, to avoid rotated copies leaking into test."""
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    keys = [symmetry_key(board) for board in inputs]
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise ValueError('a split needs at least two symmetry groups')
    count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    permutation = np.random.default_rng(seed).permutation(len(groups))
    held_out = {groups[index] for index in permutation[:count]}
    test = np.array([i for i, key in enumerate(keys) if key in held_out], dtype=np.int64)
    train = np.array([i for i, key in enumerate(keys) if key not in held_out], dtype=np.int64)
    return train, test


def fingerprint(inputs: Array, targets: Array) -> str:
    """Hash ordered arrays rather than JSON whitespace or platform endianness."""
    return hashlib.sha256(
        inputs.astype('<f8').tobytes() + targets.astype('<f8').tobytes()
    ).hexdigest()
