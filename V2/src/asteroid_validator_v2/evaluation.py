"""Held-out continuous error metrics and within-object ranking consequences."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    a, b = _ranks(first), _ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def evaluate(records: list[dict], predictions: np.ndarray) -> dict:
    predictions = np.asarray(predictions, dtype=float)
    if not len(records) or predictions.shape != (len(records),) or not np.isfinite(predictions).all():
        raise ValueError("one finite prediction is required per evaluation row")
    truth = np.array([r["geometric_error"] for r in records])
    residual = predictions - truth
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[(row["asteroid_id"], row.get("view_id", "default"))].append(i)
    rankings = []
    for (object_id, view_id), indices in groups.items():
        y, p = truth[indices], predictions[indices]
        selected = int(np.argmin(p))
        rankings.append({"asteroid_id": object_id, "view_id": view_id,
                         "top1_correct": bool(np.isclose(y[selected], y.min(), atol=1e-12, rtol=0)),
                         "selection_regret": float(y[selected] - y.min()),
                         "spearman": spearman(p, y)})
    correlations = [r["spearman"] for r in rankings if r["spearman"] is not None]
    return {"candidate_count": len(records), "independent_object_count": len({r["asteroid_id"] for r in records}),
            "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "bias": float(residual.mean()), "top1_accuracy": float(np.mean([r["top1_correct"] for r in rankings])),
            "mean_selection_regret": float(np.mean([r["selection_regret"] for r in rankings])),
            "mean_spearman": float(np.mean(correlations)) if correlations else None,
            "rankings": rankings,
            "predictions": [{"asteroid_id": r["asteroid_id"], "candidate_id": r["candidate_id"],
                             "predicted_geometric_error": float(p), "measured_geometric_error": float(y)}
                            for r, p, y in zip(records, predictions, truth)]}


def calibration_svg(records: list[dict], predictions: np.ndarray) -> str:
    """Standalone scientific diagnostic; values are percentages of diameter."""
    y, p = np.array([r["geometric_error"] for r in records]) * 100, np.asarray(predictions) * 100
    maximum = max(float(y.max()), float(p.max()), .01) * 1.08
    xcoord = lambda v: 76 + 454 * v / maximum
    ycoord = lambda v: 464 - 414 * v / maximum
    marks = ''.join(f'<circle cx="{xcoord(a):.2f}" cy="{ycoord(b):.2f}" r="4" fill="#176f94" fill-opacity=".75"/>'
                    for a, b in zip(y, p))
    ticks = ''.join(f'<text x="{xcoord(v):.1f}" y="491" text-anchor="middle">{v:.2f}</text>'
                    f'<text x="65" y="{ycoord(v)+5:.1f}" text-anchor="end">{v:.2f}</text>'
                    for v in np.linspace(0, maximum, 5))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 610 550" role="img" '
            'aria-label="Held-out predicted versus measured geometric error">'
            '<rect width="610" height="550" fill="white"/>'
            '<g font-family="Arial,sans-serif" font-size="13" fill="#222">'
            '<text x="76" y="30" font-size="18">Held-out error calibration</text>'
            '<rect x="76" y="50" width="454" height="414" fill="none" stroke="#777"/>'
            '<line x1="76" y1="464" x2="530" y2="50" stroke="#777" stroke-dasharray="5 4"/>'
            f'{marks}{ticks}'
            '<text x="303" y="526" text-anchor="middle">Measured surface error (% of diameter)</text>'
            '<text x="20" y="235" transform="rotate(-90 20 235)" text-anchor="middle">Predicted surface error (% of diameter)</text>'
            '</g></svg>')
