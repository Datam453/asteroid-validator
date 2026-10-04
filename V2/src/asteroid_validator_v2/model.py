"""Small monotone evaluator with explicit chain-rule gradients and CPU Adam."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

FEATURES = ("silhouette_score", "photometric_score", "thermal_score")
FEATURE_VERSION = "v2-frozen-observation-features-1"


def softplus(x: np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0, x)


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


def _features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError("features must be a nonempty Nx3 array")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("features must be finite and non-negative; missing evidence is not zero")
    return x


def _targets(y: np.ndarray, count: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.shape != (count,) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("targets must be one finite non-negative error per row")
    return y


def forward(parameters: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, tuple]:
    logits = parameters[:3] - np.max(parameters[:3])
    weights = np.exp(logits) / np.exp(logits).sum()
    scale = float(softplus(parameters[3]))
    weighted = x @ weights
    z = parameters[4] + scale * weighted
    return softplus(z), (weights, scale, weighted, z)


def loss_gradient(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    prediction, (weights, scale, weighted, z) = forward(parameters, x)
    dz = 2 * (prediction - y) * sigmoid(z) / len(y)
    gradient = np.empty(5)
    gradient[:3] = (dz[:, None] * scale * weights * (x - weighted[:, None])).sum(axis=0)
    gradient[3] = np.sum(dz * weighted) * sigmoid(parameters[3])
    gradient[4] = dz.sum()
    return float(np.mean((prediction - y) ** 2)), gradient


class ErrorEvaluator:
    def __init__(self, parameters: np.ndarray, scales: np.ndarray, metadata: dict | None = None):
        self.parameters = np.asarray(parameters, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self.metadata = metadata or {}
        if self.parameters.shape != (5,) or not np.isfinite(self.parameters).all():
            raise ValueError("model needs five finite parameters")
        if self.scales.shape != (3,) or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError("model needs three finite positive feature scales")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return forward(self.parameters, _features(x) / self.scales)[0]

    def save(self, path: Path) -> None:
        weights, scale, _, _ = forward(self.parameters, np.zeros((1, 3)))[1]
        payload = {"schema_version": 1, "model": "softplus_monotone_weighted_error",
                   "feature_version": FEATURE_VERSION, "features": list(FEATURES),
                   "parameters": self.parameters.tolist(), "normalization_scales": self.scales.tolist(),
                   "relative_weights": dict(zip(FEATURES, weights.tolist())),
                   "positive_scale": scale, "offset": float(self.parameters[4]),
                   "output": "predicted_symmetric_mean_surface_error/reference_diameter",
                   "optimizer": "NumPy Adam with analytic chain-rule gradients", "metadata": self.metadata}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ErrorEvaluator":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if (p.get("schema_version") != 1 or p.get("model") != "softplus_monotone_weighted_error"
                or p.get("features") != list(FEATURES) or p.get("feature_version") != FEATURE_VERSION):
            raise ValueError("incompatible model schema or feature order/version")
        return cls(p["parameters"], p["normalization_scales"], p.get("metadata"))


def train(x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray, *,
          epochs: int = 2500, learning_rate: float = .03, batch_size: int = 64,
          patience: int = 200, seed: int = 0) -> tuple[ErrorEvaluator, list[dict]]:
    x, vx = _features(x), _features(validation_x)
    y, vy = _targets(y, len(x)), _targets(validation_y, len(vx))
    if epochs < 1 or batch_size < 1 or patience < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch size, patience and learning rate must be positive")
    scales = np.median(np.abs(x), axis=0)
    scales = np.where(scales > 1e-12, scales, 1)
    x, vx = x / scales, vx / scales
    mean = max(float(y.mean()), 1e-6)
    inverse_softplus = lambda value: value + np.log(-np.expm1(-value))
    parameters = np.array([0., 0., 0., inverse_softplus(.1), inverse_softplus(mean) - .1])
    best_parameters = parameters.copy()
    best_loss = float(np.mean((forward(parameters, vx)[0] - vy) ** 2))
    history = [{"epoch": 0, "training_mse": float(np.mean((forward(parameters, x)[0] - y) ** 2)),
                "validation_mse": best_loss}]
    moment, velocity = np.zeros(5), np.zeros(5)
    rng, update, since_best, best_epoch = np.random.default_rng(seed), 0, 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            _, gradient = loss_gradient(parameters, x[batch], y[batch])
            update += 1
            moment = .9 * moment + .1 * gradient
            velocity = .999 * velocity + .001 * gradient ** 2
            parameters -= learning_rate * (moment / (1 - .9 ** update)) / (
                np.sqrt(velocity / (1 - .999 ** update)) + 1e-8)
        training_loss = float(np.mean((forward(parameters, x)[0] - y) ** 2))
        validation_loss = float(np.mean((forward(parameters, vx)[0] - vy) ** 2))
        if not np.isfinite(parameters).all() or not np.isfinite(validation_loss):
            raise ValueError("training diverged; reduce learning rate or check input scales")
        if epoch % 10 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "training_mse": training_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss - 1e-15:
            best_loss, best_parameters, since_best, best_epoch = validation_loss, parameters.copy(), 0, epoch
        else:
            since_best += 1
        if since_best >= patience:
            break
    model = ErrorEvaluator(best_parameters, scales, {"seed": seed, "best_epoch": best_epoch,
                            "epochs_completed": epoch, "best_validation_mse": best_loss,
                            "training_rows": len(x), "validation_rows": len(vx)})
    return model, history


def linear_baseline(x: np.ndarray, y: np.ndarray, scales: np.ndarray) -> dict:
    coefficients = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x / scales)), y, rcond=None)[0]
    return {"model": "ordinary_least_squares_with_nonnegative_output_clip",
            "normalization_scales": scales.tolist(), "coefficients": coefficients.tolist()}


def baseline_predict(model: dict, x: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.column_stack((np.ones(len(x)), x / model["normalization_scales"])) @
                      np.asarray(model["coefficients"]))
