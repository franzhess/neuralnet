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

The demo trains **10 networks for 300 epochs each**, with **1,000 games per network
per matchup**: 80,000 games across eight matchups. All networks share one held-out
split, with training seeds 42–51 and gameplay seeds 123–132.

It writes `artifacts/moves.json`, ten checkpoints under `artifacts/models/`, and
`artifacts/experiment.json`. The report preserves every network's configuration,
held-out indices, losses, game outcomes, role-specific statistics, and aggregate
uncertainty estimates. Generated artifacts are ignored by Git. Existing outputs
with the same names are replaced; use a fresh output directory for a new cohort.

For a quick smoke test:

```sh
python -m neuralnet demo --networks 2 --epochs 2 --games 4 --output-dir artifacts/smoke
```

Or train and evaluate a single network independently:

```sh
python -m neuralnet generate artifacts/moves.json
python -m neuralnet train artifacts/moves.json artifacts/model.npz --seed 42 --split-seed 42
python -m neuralnet evaluate artifacts/moves.json artifacts/model.npz \
    --games 1000 --seed 123 --output artifacts/evaluation.json
```

`neuralnet-demo` is an equivalent installed entry point. Use `--help` on any command
to see its options. Game counts must be even so each player gets equal turns as X
and O. Evaluation checks the dataset fingerprint and reuses the split recorded in
the checkpoint.

`demo --networks` controls the cohort size. `--seed` and `--evaluation-seed` are
the first training/gameplay seeds; each subsequent network increments them by
one. `--split-seed` remains fixed for the entire cohort. To re-evaluate an individual
cohort checkpoint, use `evaluate artifacts/moves.json artifacts/models/network-01.npz`.

## Results—not an unbeatable neural player

The aggregate snapshot is in [docs/results.json](docs/results.json); generating
the experiment also saves all individual runs in `artifacts/experiment.json`.

The recorded cohort used Python 3.11.13 and NumPy 2.4.6, with **3,603 training
positions** and the same **917 held-out positions** for every network:

- Optimal-move agreement: **99.24% mean**, **0.44 percentage points standard
  deviation**, range **98.47–99.89%**. Bootstrap interval for the mean: **98.96–99.49%**.
- Held-out loss: **0.189 mean**, **0.012 standard deviation**.
- Neural loss rate against minimax: **12.44% mean**, **8.93 percentage points
  standard deviation**, range **0–22.6%**. Bootstrap interval for the mean: **7.13–17.65%**.

| Player vs opponent | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| Neural network vs random | 9,317 | 623 | 60 |
| Neural network vs minimax | 0 | 8,756 | 1,244 |
| Random vs random | 4,402 | 1,256 | 4,342 |
| Minimax vs random | 8,719 | 1,281 | 0 |
| Minimax vs minimax | 0 | 10,000 | 0 |
| Neural network vs random-aware minimax | 0 | 7,533 | 2,467 |
| Random-aware minimax vs random | 9,546 | 454 | 0 |
| Random-aware minimax vs minimax | 0 | 10,000 | 0 |

Each row totals 10,000 games, half with the evaluated player as X and half as O.
Two networks recorded no losses against minimax, but that is **not a proof** they
are unbeatable. Random-tie minimax optimizes the worst-case outcome, not the win
rate against a stochastic opponent. The new random-aware baseline resolves that
distinction while retaining the same safety guarantee.

Against random play, random-aware minimax won **95.46%** of sampled games
(bootstrap interval **95.09–95.79%**), versus the networks' **93.17%**
(**92.39–93.87%**). It had no losses; the networks lost **0.60%**.
The networks lost **24.67%** against the new policy, versus **12.44%** against
random-tie minimax. The new tie-breaker is optimized for uniform random play,
not specifically for exploiting these networks; those head-to-head results
remain empirical observations, not a guarantee about all neural opponents.

High per-position agreement does **not** imply reliable play over an entire game.
Minimax has an exact guarantee; the network only approximates its strategy.
Gameplay starts from the empty board and can visit training positions; only the
position-level metrics use the held-out set.

### Stronger baseline: random-aware minimax

Ordinary minimax gives the same value to moves that guarantee the same worst-case
outcome, then chooses among them uniformly. Those moves need not have the same
chance of winning against a random opponent.

The random-aware policy uses a lexicographic objective:

1. Only consider moves with the **best minimax value**.
2. Among those moves, maximize the probability of eventually winning against an
   opponent that chooses **uniformly among all legal moves**.
3. Break remaining equal-probability ties uniformly with the seeded RNG.

The probability calculation recursively averages random opponent turns and
maximizes safe player turns. It uses cached `Fraction` arithmetic, so comparisons
and ties are exact rather than dependent on floating-point rounding. Winning
contributes one; draws and losses contribute zero. Avoiding losses is enforced by
the higher-priority minimax constraint, not traded away for more wins.

From the empty board, the exact win probabilities for this policy are:

| Role | Exact probability | Win rate |
| --- | --- | ---: |
| X | `191/192` | 99.48% |
| O | `866/945` | 91.64% |
| Balanced X/O | `115589/120960` | **95.56%** |

These are analytical probabilities, not sampled estimates. They assume precisely
the uniform-random opponent above; this tie-breaker is not claimed to maximize
wins against every other opponent. The dataset and neural training objective
remain unchanged: the network still imitates worst-case minimax move values.

### What the uncertainty estimates mean

- **Between-run variation:** mean, sample standard deviation, minimum, and maximum
  across networks. Neural matchups vary with training and gameplay randomness;
  the fixed-policy baselines vary only with gameplay randomness.
- **Cohort mean intervals:** approximate 95% percentile-bootstrap intervals from
  10,000 resamples of **whole network runs**, using bootstrap seed 0. The 10,000
  games per matchup are not treated as 10,000 independent trained networks.
- **Individual-network intervals:** marginal 95% Wilson intervals for win/draw/loss
  rates, with separate X/O outcome rates and intervals. Overall intervals pool the balanced
  roles and are approximations, not simultaneous intervals for all three outcomes.
- **Limits:** ten networks are still a small sample. Intervals are conditional on
  this dataset, held-out split, architecture, and hyperparameters. They are not
  corrected for multiple comparisons and are not significance tests. If every
  observed run has zero losses, an empirical bootstrap can produce `[0, 0]`; that
  does **not** prove a zero loss probability. Per-network Wilson upper bounds remain
  nonzero. Minimax's guarantee comes from exhaustive verification, not sampling.

No best-performing network is selected: every trained network is saved and included.

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
   Save the actual held-out indices in each checkpoint, and keep them identical
   across networks while changing initialization and shuffling seeds.
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
  exact search with a compact player-relative recurrence. Keep both the ordinary
  random-tie policy and the random-aware safe policy to distinguish safety from
  opponent-specific win-rate optimization.
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
| `src/neuralnet/players.py` | Random, ordinary/random-aware minimax, neural policies and matches |
| `src/neuralnet/experiment.py` | Training, held-out metrics, baseline tournaments |
| `src/neuralnet/config.py` | Immutable experiment settings, shared defaults and validation |
| `src/neuralnet/statistics.py` | Wilson intervals and network-level bootstrap summaries |
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
config = ExperimentConfig(epochs=100, hidden_nodes=32, games=1000)
model = train_model(inputs, targets, config=config)
report = evaluate(model, inputs, targets)
```

For a full cohort, call `run_experiment(inputs, targets, config=config)` from
`neuralnet.experiment`; it returns the list of networks and the complete report.

The checkpoint records the full configuration and held-out indices. Evaluation
without an override
reuses those settings; an explicit configuration can change the game count and
evaluation seed, but **never changes the recorded training/test split**. Older
checkpoints retain their coupled split/training seed and legacy 200-game default
when those fields were not recorded.
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
Exhaustive continuation tests verify that ordinary and random-aware minimax cannot
lose from the empty board as either player, against every legal opponent
continuation. Additional checks cover every reachable position, all tied preferred
moves, and exact win probabilities via independent enumeration of the policy tree.
Additional tests verify cohort reproducibility, distinct seeds with identical
held-out membership, X/O balancing, statistical interval calculations, and reload
of each independently saved checkpoint.

To check packaging independently, install the wheel from `dist/` into a fresh
virtual environment and run the demo outside the source tree.
The project is [MIT licensed](LICENSE).
