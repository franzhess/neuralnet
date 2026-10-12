"""Shared, validated settings for training and evaluating an experiment."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, Self


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """Immutable defaults shared by the Python API, CLI and saved checkpoints."""

    seed: int = 42
    split_seed: int = 42
    test_fraction: float = 0.2
    hidden_nodes: int = 64
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.1
    games: int = 1000
    evaluation_seed: int = 123
    networks: int = 10

    def __post_init__(self) -> None:
        for name in (
            'seed',
            'split_seed',
            'evaluation_seed',
            'epochs',
            'hidden_nodes',
            'batch_size',
            'games',
            'networks',
        ):
            value = getattr(self, name)
            minimum = 0 if name in ('seed', 'split_seed', 'evaluation_seed', 'epochs') else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer of at least {minimum}')
        if self.games < 2 or self.games % 2:
            raise ValueError('games must be a positive even number of at least two')
        for name in ('test_fraction', 'learning_rate'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f'{name} must be a finite number')
        if not 0 < self.test_fraction < 1:
            raise ValueError('test_fraction must be between zero and one')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> Self:
        """Read configuration fields, ignoring metrics and defaulting missing fields.

        Legacy checkpoints coupled their training and split seeds and defaulted
        to 200 evaluation games. Preserve those settings when fields are absent.
        """
        settings = {
            field.name: metadata[field.name] for field in fields(cls) if field.name in metadata
        }
        if 'split_seed' not in settings and 'seed' in settings:
            settings['split_seed'] = settings['seed']
        if metadata and 'games' not in settings:
            settings['games'] = 200
        if metadata and 'networks' not in settings:
            settings['networks'] = 1
        return cls(**settings)
