"""Train, freeze, evaluate and predict with auditable saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .dataset import dataset_arrays, validate_splits, write_json
from .evaluation import evaluate, calibration_svg, spearman
from .model import ErrorEvaluator, train, linear_baseline, baseline_predict, FEATURES, FEATURE_VERSION


def _v1_grid_baseline(records: list[dict], x: np.ndarray, y: np.ndarray, scales: np.ndarray) -> dict:
    """Rank-based grid search, with V2 labels and a fitted absolute-error scale."""
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault((row["asteroid_id"], row.get("view_id", "default")), []).append(i)
    best = None
    for a in range(21):
        for b in range(21 - a):
            w = np.array([a, b, 20 - a - b]) / 20
            score = (x / scales) @ w
            correlations = [spearman(score[indices], y[indices]) for indices in groups.values()]
            objective = (np.mean([c if c is not None else 0 for c in correlations]),
                         np.mean([np.argmin(score[i]) == np.argmin(y[i]) for i in groups.values()]),
                         -np.linalg.norm(w - 1 / 3))
            if best is None or objective > best[0]:
                best = objective, w
    w = best[1]
    score = (x / scales) @ w
    intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(len(y)), score)), y, rcond=None)[0]
    return {"weights": w.tolist(), "normalization_scales": scales.tolist(),
            "intercept": float(intercept), "slope": float(slope),
            "scope": "V1-style rank grid search on V2 surface labels; training affine error calibration added"}


def fit_dataset(dataset_path: Path, output: Path, *, epochs: int = 2500, seed: int = 0,
                learning_rate: float = .03, batch_size: int = 64) -> dict:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    groups = validate_splits(data)
    x, y = dataset_arrays(groups["train"])
    vx, vy = dataset_arrays(groups["validation"])
    tx, ty = dataset_arrays(groups["test"])
    model, history = train(x, y, vx, vy, epochs=epochs, seed=seed,
                           learning_rate=learning_rate, batch_size=batch_size)
    policy = groups["train"][0].get("alignment_policy")
    model.metadata.update({"alignment_policy": policy,
        "training_objects": sorted({r["asteroid_id"] for r in groups["train"]}),
        "validation_objects": sorted({r["asteroid_id"] for r in groups["validation"]}),
        "label_metric": data.get("label_metric"), "feature_extraction_version": FEATURE_VERSION,
        "dataset": str(Path(dataset_path).resolve())})
    output = Path(output)
    model.save(output / "model.json")
    write_json(output / "training_history.json", {"history": history, "best_epoch": model.metadata["best_epoch"]})
    baseline = linear_baseline(x, y, model.scales)
    write_json(output / "linear_baseline.json", baseline)
    grid = _v1_grid_baseline(groups["train"], x, y, model.scales)
    write_json(output / "rank_grid_baseline.json", grid)
    prediction = model.predict(tx)
    results = {"v2_monotone_regression": evaluate(groups["test"], prediction),
               "linear_regression": evaluate(groups["test"], baseline_predict(baseline, tx)),
               "training_mean_constant": evaluate(groups["test"], np.full(len(tx), y.mean())),
               "v1_style_rank_grid_calibrated": evaluate(groups["test"], np.maximum(0,
                   grid["intercept"] + grid["slope"] * ((tx / model.scales) @ grid["weights"])))}
    for column, name in enumerate(FEATURES):
        coeff = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x[:, column])), y, rcond=None)[0]
        single_predictions = np.maximum(0, coeff[0] + coeff[1] * tx[:, column])
        results[f"{name}_only_regression"] = evaluate(groups["test"], single_predictions)
    report = {"schema_version": 1, "feature_version": FEATURE_VERSION,
              "split_object_counts": {s: len({r["asteroid_id"] for r in rows}) for s, rows in groups.items()},
              "failed_candidate_count": len(data.get("failures", [])),
              "failure_records": data.get("failures", []), "results": results,
              "scope": "Held-out results for this dataset only. Procedural examples do not validate real generators."}
    write_json(output / "evaluation.json", report)
    (output / "calibration.svg").write_text(calibration_svg(groups["test"], prediction), encoding="utf-8")
    lines = ["# V2 local experiment", "", report["scope"], "",
             "| Evaluator | MAE (% diameter) | RMSE (% diameter) | Top-1 |", "|---|---:|---:|---:|"]
    for name, metrics in results.items():
        lines.append(f"| {name} | {100*metrics['mae']:.3f} | {100*metrics['rmse']:.3f} | {metrics['top1_accuracy']:.1%} |")
    lines.extend(["", "![Held-out predicted versus measured surface error](calibration.svg)", "",
                  f"Best validation epoch: {model.metadata['best_epoch']}. Failed candidates: {report['failed_candidate_count']}.",
                  "", "Feature inputs contain no reference geometry or geometric-error labels."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def predict_features(model_path: Path, feature_path: Path, output: Path) -> dict:
    model = ErrorEvaluator.load(model_path)
    data = json.loads(Path(feature_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("feature_version") != FEATURE_VERSION:
        raise ValueError("incompatible feature schema/version")
    if model.metadata.get("alignment_policy") != data.get("alignment_policy"):
        raise ValueError("feature alignment policy differs from the trained model")
    x, _ = dataset_arrays(data["records"], labels=False)
    prediction = model.predict(x)
    rows = [{"candidate_id": row["candidate_id"], "mesh_path": row.get("mesh_path"),
             "predicted_geometric_error": float(value), "predicted_error_percent_diameter": float(100*value),
             "physics_features": {key: row[key] for key in FEATURES},
             "alignment": row.get("alignment"), "overlay_path": row.get("overlay_path")}
            for row, value in zip(data["records"], prediction)]
    rows.sort(key=lambda row: row["predicted_geometric_error"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    result = {"schema_version": 1, "reference_geometry_required": False,
              "model": str(Path(model_path).resolve()), "candidates": rows,
              "failures": data.get("failures", []),
              "interpretation": "Estimated normalized mean surface error, not a correctness probability or certified bound."}
    write_json(output, result)
    return result
