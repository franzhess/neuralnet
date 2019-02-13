# Neural network from scratch: learning tic-tac-toe from minimax

A small Python/NumPy experiment in supervised learning: implement a neural network,
generate exact move values with minimax, and see how closely the network can imitate
an optimal tic-tac-toe player. No PyTorch, TensorFlow, or reinforcement learning.

## Main learning: don't use fancy AI when an optimal strategy already solves the game

**The main lesson was not to use fancy AI for a game with a known optimal strategy.**
Tic-tac-toe is solved: a simple minimax player never needs to lose. A neural network
adds training, approximation errors, and evaluation work without improving on that
guarantee. It is useful here as a way to learn backpropagation—not as the best tool
for playing the game.

In everyday terms, this was my “don't use AI when there's already a dominant
strategy” lesson. Strictly speaking, tic-tac-toe has an optimal strategy, not a
single dominant strategy in the game-theory sense. Perfect play from both sides
ends in a draw.

## Quick start

Requires Python 3.11 or newer. From a clone of this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m neuralnet demo
```

The demo generates data, trains for 300 epochs, and evaluates 200 games per matchup.
It writes `artifacts/moves.json`, `artifacts/model.npz`, and
`artifacts/experiment.json`. The JSON report includes hyperparameters, seeds,
environment versions, dataset fingerprint, losses, and game outcomes. Generated
artifacts are ignored by Git. Existing outputs in the chosen directory are replaced.

For a quick smoke test:

```sh
python -m neuralnet demo --epochs 2 --games 4 --output-dir artifacts/smoke
```

Or run each stage independently:

```sh
python -m neuralnet generate artifacts/moves.json
python -m neuralnet train artifacts/moves.json artifacts/model.npz --seed 42
python -m neuralnet evaluate artifacts/moves.json artifacts/model.npz \
    --games 200 --seed 123 --output artifacts/evaluation.json
```

`neuralnet-demo` is an equivalent installed entry point. Use `--help` on any command
to see its options. Game counts must be even so each player gets equal turns as X
and O. Evaluation checks the dataset fingerprint and reuses the split recorded in
the checkpoint.

## Results—not an unbeatable neural player

One default run, using Python 3.11.13 and NumPy 2.4.6, produced the following results.
The recorded configuration and full metrics are in [docs/results.json](docs/results.json).

- Training positions: **3,603**; held-out positions: **917**.
- Held-out optimal-move agreement: **98.6%** for the network versus **59.8%** for
  a uniformly random legal move.
- Training loss fell from **4.792** to **0.156**; held-out loss was **0.202**.

| Player vs opponent | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| Neural network vs random | 187 | 11 | 2 |
| Neural network vs minimax | 0 | 156 | 44 |
| Random vs random | 81 | 25 | 94 |
| Minimax vs random | 176 | 24 | 0 |
| Minimax vs minimax | 0 | 200 | 0 |

High per-position agreement does **not** imply reliable play over an entire game.
The network still loses to optimal play, whereas minimax has an exact guarantee.
These are single-seed samples, not evidence that the network is better than minimax
because it won more games against random in this particular run. Gameplay starts
from the empty board and can visit training positions; only the position-level
metrics use the held-out set.

## How it works

1. Enumerate all **4,520 reachable nonterminal positions**, stopping at wins and
   deduplicating different move orders that reach the same board.
2. Normalize each board to the player-to-move's perspective: `1` is “me”, `-1` is
   “opponent”, and `0` is empty. X always starts in actual games.
3. Label every legal move with its exact minimax outcome: loss `-1`, draw `0`, or
   win `1`. Occupied squares receive a sentinel target of `-1`.
4. Split by rotation/reflection symmetry groups before training. A rotated copy of
   a training board cannot appear in the held-out set. The default holds out 20%
   of groups, so the row fraction is approximately, not exactly, 20%.
5. Train a **9 → 64 → 9** network with tanh activations, bias rows, Xavier uniform
   initialization, mini-batch SGD, and an explicit learning rate. All gradients
   are computed with NumPy matrix operations.
6. Predict move scores, mask occupied squares, and choose the highest-scoring legal
   move. Minimax and neural policies break equal-score ties with a seeded RNG.

The loss is `0.5 * sum((prediction - target)²) / batch_size`, summed across output
squares, including occupied-square targets. Gradients are averaged over the batch;
the final partial batch is included. Activation derivatives take **activation
outputs**, not pre-activation inputs.

Optimal-move agreement accepts **any** equally optimal move, rather than comparing
against one arbitrary teacher choice. The random baseline is the exact average
fraction of legal moves that are optimal. Tied neural scores are scored as uniform
choices among tied best legal moves.

### Design choices and limitations

- **Minimax is the baseline, not the novelty.** Cached negamax implements the same
  exact search with a compact player-relative recurrence.
- **Supervised imitation, not self-play.** The network learns from a perfect teacher
  and has no search or rule-based tactical fallback when choosing a move.
- **Small by design.** Only 3×3 tic-tac-toe and one hidden layer are supported.
  Symmetry-safe holdouts are harder than random row splits, but don't establish
  generalization to other games.
- **Reproducible, not universally bit-identical.** Seeds control initialization,
  shuffling, and tie-breaking. Numerical results can differ across NumPy/BLAS
  versions and platforms. Reported package versions make comparisons traceable.
- **Explicit, safe I/O.** Data is validated JSON; checkpoints are non-pickled NPZ
  archives. Checkpoints preserve prediction weights and experiment metadata, not
  optimizer/RNG state for an identical training resume.

## Code map

| File | Responsibility |
| --- | --- |
| `src/neuralnet/network.py` | Activations, forward/backward passes, SGD, checkpoints |
| `src/neuralnet/game.py` | Immutable boards, rules, reachable states, exact minimax |
| `src/neuralnet/data.py` | Teacher data, validation, fingerprints, symmetry-aware splits |
| `src/neuralnet/players.py` | Random/minimax/neural policies and game runner |
| `src/neuralnet/experiment.py` | Training, held-out metrics, baseline tournaments |
| `src/neuralnet/config.py` | Immutable experiment settings, shared defaults and validation |
| `src/neuralnet/cli.py` | Explicit command-line workflow |
| `tests/` | Numerical, rules, data, reproducibility, and end-to-end tests |

Tests mirror those responsibilities: `test_game.py` covers rules and the exact
solver, `test_players.py` covers policies and matches, `test_experiment.py` covers
training/evaluation, and `test_cli.py` exercises the installed command from outside
the source tree. Dataset splitting and validation live in `test_data.py`.

The `src` layout separates installable code from tests and documentation. Install
the package before running tests; pytest's importlib mode exercises the installed
package without adding the repository root to its import path.

The package and its CLI are the only entry points. The old standalone XOR,
bit-shifting, plotting, and interactive-game examples have been removed to keep
the repository focused on the minimax-teacher experiment. Importing the package
does not train, print, or access data files.

The Python API uses snake_case functions and descriptive class names:

```python
from neuralnet import NeuralNetwork, Tanh

network = NeuralNetwork(2, 4, 1, Tanh(), seed=42, learning_rate=0.1)
history = network.train(inputs, targets, epochs=1000, batch_size=4)
predictions = network.forward(inputs)
```

Activations are callable (`Tanh()(values)`). Mini-batch training uses the same
`train` method as full-batch training; there is no separate `trainSingle` API.
Player policies are functions in `neuralnet.players`, and `play_game` runs a match
without printing. Existing version-1 model checkpoints still load.

### Shared experiment configuration

`ExperimentConfig` is a frozen, typed dataclass that owns training and evaluation
defaults. Both the CLI and Python API validate settings through it:

```python
from neuralnet import ExperimentConfig
from neuralnet.data import generate_dataset
from neuralnet.experiment import evaluate, train_model

inputs, targets = generate_dataset()
config = ExperimentConfig(epochs=100, hidden_nodes=32, games=200)
model = train_model(inputs, targets, config=config)
report = evaluate(model, inputs, targets)
```

The checkpoint records the full configuration. Evaluation without an override
reuses those settings; an explicit configuration can change the game count and
evaluation seed, but **never changes the recorded training/test split**. Older
checkpoints use shared defaults for evaluation settings they did not record.
Zero epochs are allowed to evaluate an untrained baseline.

## Verification

```sh
python -m pytest -q
ruff check .
ruff format --check .
python -m build
```

Tests include finite-difference checks of **every weight and bias** for both sigmoid
and tanh, XOR convergence, complete batch coverage, all eight winning lines,
dataset rejection cases, symmetry separation, model round-trips, and a CLI pipeline.
An exhaustive continuation test verifies minimax cannot lose from the empty board
as either player, against every legal opponent continuation.

To check packaging independently, install the wheel from `dist/` into a fresh
virtual environment and run the demo outside the source tree.
The project is [MIT licensed](LICENSE).
