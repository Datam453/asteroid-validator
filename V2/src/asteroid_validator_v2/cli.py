"""Command-line interfaces for locally building, training and using V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_dataset, write_json
from .demo import demo_dataset
from .features import extract_case
from .geometry import load_mesh, geometric_error
from .pipeline import fit_dataset, predict_features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="measure candidate/reference geometric error")
    label.add_argument("--candidate", type=Path, required=True)
    label.add_argument("--reference", type=Path, required=True)
    label.add_argument("--samples", type=int, default=2048)
    label.add_argument("--seed", type=int, default=0)
    label.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract", help="compute observation-only physics features")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build-dataset", help="assemble fixed features and geometric labels")
    build.add_argument("--cases", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--samples", type=int, default=2048)
    build.add_argument("--seed", type=int, default=0)
    fit = commands.add_parser("train", help="fit on train, select on validation, evaluate on test")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="score candidates without reference geometry")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="generate and run a CPU procedural engineering demo")
    demo.add_argument("--output", type=Path, default=Path("runs/local-demo"))
    demo.add_argument("--train-objects", type=int, default=16)
    demo.add_argument("--validation-objects", type=int, default=4)
    demo.add_argument("--test-objects", type=int, default=4)
    demo.add_argument("--candidates", type=int, default=4)
    demo.add_argument("--resolution", type=int, default=64)
    demo.add_argument("--samples", type=int, default=512)
    for command in (fit, demo):
        command.add_argument("--epochs", type=int, default=2500)
        command.add_argument("--seed", type=int, default=2026)
        command.add_argument("--learning-rate", type=float, default=.03)
        command.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            result = geometric_error(load_mesh(args.candidate), load_mesh(args.reference),
                                     samples=args.samples, seed=args.seed)
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "extract":
            result = extract_case(args.manifest, args.output.parent / "overlays")
            write_json(args.output, result)
            print(f"Extracted {len(result['records'])} candidates; {len(result['failures'])} failures")
        elif args.command == "build-dataset":
            result = build_dataset(args.cases, args.output, samples=args.samples, seed=args.seed)
            print(f"Saved {len(result['records'])} feature/label rows")
        elif args.command in ("train", "demo"):
            path = args.dataset if args.command == "train" else demo_dataset(
                args.output, train_objects=args.train_objects, validation_objects=args.validation_objects,
                test_objects=args.test_objects, candidates=args.candidates, resolution=args.resolution,
                samples=args.samples, seed=args.seed)
            result = fit_dataset(path, args.output if args.command == "train" else args.output / "experiment",
                epochs=args.epochs, seed=args.seed, learning_rate=args.learning_rate, batch_size=args.batch_size)
            print(json.dumps({"split_object_counts": result["split_object_counts"],
                              "held_out": {key: value for key, value in result['results']['v2_monotone_regression'].items()
                                           if key not in ('rankings', 'predictions')}}, indent=2))
        else:
            result = predict_features(args.model, args.features, args.output)
            print(f"Rated {len(result['candidates'])} candidates without reference geometry")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0
