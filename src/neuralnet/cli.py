"""Explicit commands for generating data, training and evaluating the experiment."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .data import generate_dataset, load_dataset, save_dataset
from .experiment import evaluate, train_model
from .network import NeuralNetwork

_DEFAULTS = ExperimentConfig()


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--epochs', type=int, default=_DEFAULTS.epochs)
    parser.add_argument('--hidden-nodes', type=int, default=_DEFAULTS.hidden_nodes)
    parser.add_argument('--batch-size', type=int, default=_DEFAULTS.batch_size)
    parser.add_argument('--learning-rate', type=float, default=_DEFAULTS.learning_rate)
    parser.add_argument('--seed', type=int, default=_DEFAULTS.seed)
    parser.add_argument('--split-seed', type=int, default=_DEFAULTS.split_seed)
    parser.add_argument('--test-fraction', type=float, default=_DEFAULTS.test_fraction)


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        seed=getattr(args, 'seed', _DEFAULTS.seed),
        split_seed=getattr(args, 'split_seed', _DEFAULTS.split_seed),
        test_fraction=getattr(args, 'test_fraction', _DEFAULTS.test_fraction),
        hidden_nodes=getattr(args, 'hidden_nodes', _DEFAULTS.hidden_nodes),
        epochs=getattr(args, 'epochs', _DEFAULTS.epochs),
        batch_size=getattr(args, 'batch_size', _DEFAULTS.batch_size),
        learning_rate=getattr(args, 'learning_rate', _DEFAULTS.learning_rate),
        games=getattr(args, 'games', _DEFAULTS.games),
        evaluation_seed=getattr(args, 'evaluation_seed', _DEFAULTS.evaluation_seed),
        networks=getattr(args, 'networks', _DEFAULTS.networks),
    )


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Learn tic-tac-toe from a minimax teacher.')
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', help='generate validated teacher data')
    generate_parser.add_argument('output', type=Path)
    train_parser = commands.add_parser('train', help='train with a symmetry-group holdout')
    train_parser.add_argument('data', type=Path)
    train_parser.add_argument('model', type=Path)
    _add_training_options(train_parser)
    evaluate_parser = commands.add_parser('evaluate', help='evaluate a saved checkpoint')
    evaluate_parser.add_argument('data', type=Path)
    evaluate_parser.add_argument('model', type=Path)
    evaluate_parser.add_argument('--games', type=int, help='override the checkpoint game count')
    evaluate_parser.add_argument(
        '--seed', dest='evaluation_seed', type=int, help='override the checkpoint evaluation seed'
    )
    evaluate_parser.add_argument('--output', type=Path)
    demo_parser = commands.add_parser('demo', help='run generate, train and evaluate')
    demo_parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    demo_parser.add_argument('--games', type=int, default=_DEFAULTS.games)
    demo_parser.add_argument('--evaluation-seed', type=int, default=_DEFAULTS.evaluation_seed)
    _add_training_options(demo_parser)
    args = parser.parse_args(argv)
    try:
        if args.command == 'generate':
            inputs, targets = generate_dataset()
            save_dataset(args.output, inputs, targets)
            print(f'Saved {len(inputs)} positions to {args.output}')
        elif args.command == 'train':
            config = _experiment_config(args)
            model = train_model(*load_dataset(args.data), config=config)
            model.save(args.model)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in model.metadata.items()
                        if key != 'training_loss_history'
                    },
                    indent=2,
                )
            )
        elif args.command == 'evaluate':
            model = NeuralNetwork.load(args.model)
            saved_config = ExperimentConfig.from_metadata(model.metadata)
            config = replace(
                saved_config,
                games=args.games if args.games is not None else saved_config.games,
                evaluation_seed=(
                    args.evaluation_seed
                    if args.evaluation_seed is not None
                    else saved_config.evaluation_seed
                ),
            )
            report = evaluate(
                model,
                *load_dataset(args.data),
                config=config,
            )
            if args.output:
                _write_report(args.output, report)
            print(json.dumps(report, indent=2))
        else:
            config = _experiment_config(args)
            inputs, targets = generate_dataset()
            model = train_model(inputs, targets, config=config)
            report = {
                'training': model.metadata,
                'evaluation': evaluate(model, inputs, targets, config=config),
            }
            save_dataset(args.output_dir / 'moves.json', inputs, targets)
            model.save(args.output_dir / 'model.npz')
            _write_report(args.output_dir / 'experiment.json', report)
            print(json.dumps(report['evaluation'], indent=2))
            print(f'Artifacts saved to {args.output_dir}')
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')
