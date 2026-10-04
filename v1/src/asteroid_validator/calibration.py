"""Calibrate multimodal ranking weights on benchmarks with known geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRICS = ("silhouette_score", "photometric_score", "thermal_score")


@dataclass(frozen=True)
class CalibrationCase:
    manifest_path: Path
    report_path: Path
    candidate_ids: tuple[str, ...]
    truth: np.ndarray
    scores: np.ndarray


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including deterministic tie handling."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def load_calibration_case(manifest_path: Path, report_path: Path) -> CalibrationCase:
    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth_by_id = {
        item["id"]: float(item["benchmark_only_vertex_rmse"])
        for item in manifest["candidates"]
    }
    rows = []
    for item in report.get("meshes", []):
        candidate_id = Path(item["mesh_path"]).stem
        if candidate_id not in truth_by_id:
            continue
        missing = [metric for metric in METRICS if metric not in item]
        if missing:
            raise ValueError(
                f"{report_path} candidate {candidate_id} lacks {', '.join(missing)}"
            )
        values = [float(item[metric]) for metric in METRICS]
        if not np.isfinite(values).all():
            raise ValueError(f"{report_path} candidate {candidate_id} has non-finite scores")
        rows.append((candidate_id, truth_by_id[candidate_id], values))
    if len(rows) < 3:
        raise ValueError(f"{report_path} needs at least three matched candidates")
    return CalibrationCase(
        manifest_path=manifest_path,
        report_path=report_path,
        candidate_ids=tuple(row[0] for row in rows),
        truth=np.asarray([row[1] for row in rows], dtype=np.float64),
        scores=np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def calibrate_weights(
    cases: list[CalibrationCase],
    *,
    grid_steps: int = 20,
) -> dict:
    """Search non-negative simplex weights after robust per-term scaling."""

    if not cases:
        raise ValueError("at least one calibration case is required")
    if grid_steps < 2:
        raise ValueError("grid_steps must be at least 2")
    pooled = np.vstack([case.scores for case in cases])
    scales = np.median(np.abs(pooled), axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)

    best: tuple[float, float, float, np.ndarray] | None = None
    equal = np.full(3, 1.0 / 3.0)
    for first in range(grid_steps + 1):
        for second in range(grid_steps - first + 1):
            third = grid_steps - first - second
            weights = np.asarray([first, second, third], dtype=np.float64) / grid_steps
            correlations = []
            top1 = []
            for case in cases:
                combined = (case.scores / scales) @ weights
                correlations.append(spearman_rank_correlation(combined, case.truth))
                top1.append(int(np.argmin(combined) == np.argmin(case.truth)))
            mean_correlation = float(np.mean(correlations))
            mean_top1 = float(np.mean(top1))
            simplicity = -float(np.linalg.norm(weights - equal))
            objective = (mean_correlation, mean_top1, simplicity)
            if best is None or objective > best[:3]:
                best = (*objective, weights)

    assert best is not None
    normalized_weights = best[3]
    raw_coefficients = normalized_weights / scales
    if raw_coefficients[0] > 1e-12:
        raw_coefficients /= raw_coefficients[0]
    else:
        raw_coefficients /= max(float(raw_coefficients.max()), 1e-12)

    case_results = []
    for case in cases:
        combined = case.scores @ raw_coefficients
        order = np.argsort(combined)
        case_results.append(
            {
                "manifest": str(case.manifest_path),
                "report": str(case.report_path),
                "spearman": spearman_rank_correlation(combined, case.truth),
                "top1_correct": bool(order[0] == np.argmin(case.truth)),
                "ranking": [case.candidate_ids[index] for index in order],
            }
        )
    return {
        "schema_version": 1,
        "method": "nonnegative_simplex_grid_search_with_median_score_scaling",
        "grid_steps": grid_steps,
        "training_case_count": len(cases),
        "score_scales": dict(zip(METRICS, scales.tolist())),
        "normalized_weights": dict(zip(METRICS, normalized_weights.tolist())),
        "cli_weights": {
            "silhouette_term_weight": float(raw_coefficients[0]),
            "photometric_weight": float(raw_coefficients[1]),
            "thermal_weight": float(raw_coefficients[2]),
        },
        "mean_training_spearman": best[0],
        "mean_training_top1_accuracy": best[1],
        "cases": case_results,
        "warning": (
            "Training metrics are optimistic. Calibrate on multiple benchmarks and "
            "evaluate the saved weights on separate held-out cases. A full matcher "
            "rerun can differ from this post-hoc estimate because the weights also "
            "change which pose wins the final rerank."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        required=True,
        metavar=("MANIFEST", "REPORT"),
        help="ground-truth manifest and matching multiphysics report; repeatable",
    )
    parser.add_argument("--grid-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = [
        load_calibration_case(Path(manifest), Path(report))
        for manifest, report in args.case
    ]
    result = calibrate_weights(cases, grid_steps=args.grid_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    weights = result["cli_weights"]
    print(args.output)
    print(
        "Use: "
        f"--silhouette-term-weight {weights['silhouette_term_weight']:.8g} "
        f"--photometric-weight {weights['photometric_weight']:.8g} "
        f"--thermal-weight {weights['thermal_weight']:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
