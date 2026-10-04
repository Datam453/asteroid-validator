# Asteroid Validator V2

V2 predicts geometric reconstruction error from silhouette, optical and thermal
consistency with observations. It accepts supplied candidate meshes and uses
reference geometry only for training labels and evaluation.

## Installation

From this directory, using Python 3.11 or later:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -e .
```

## Train

Prepare manifests and a case list following the [data contract](docs/data-contract.md).

```powershell
& ./run.ps1 build-dataset --cases cases.json --output runs/study/dataset.json --samples 2048
& ./run.ps1 train --dataset runs/study/dataset.json --output runs/study/experiment --epochs 2500
```

Normalization uses training objects only. Adam minimizes squared prediction
error, and validation error selects the saved checkpoint. The reference mesh,
candidate geometry and physics validators remain fixed during optimization.
The evaluator trains on a CPU with NumPy.

## Score new candidates

```powershell
& ./run.ps1 extract --manifest observations/manifest.json --output features.json
& ./run.ps1 predict --model runs/study/experiment/model.json --features features.json --output ratings.json
```

Prediction requires no reference geometry. Lower scores indicate lower estimated
geometric discrepancy. Multiplying a score by 100 expresses estimated mean
surface distance as a percentage of reference diameter.

## Outputs

Training saves model parameters, normalization, training history, held-out
predictions and baseline comparisons. Evaluation reports MAE, RMSE, bias, rank
correlation and candidate-selection performance. Invalid inputs produce failure
records rather than zero scores.

## Implementation

| Module | Responsibility |
|---|---|
| `geometry.py` | Area-weighted sampling and point-to-triangle error measurement |
| `features.py` | Observation-only silhouette, optical and thermal features |
| `dataset.py` | Feature/label assembly and object-level split validation |
| `model.py` | Differentiable weighted scorer and Adam optimization |
| `pipeline.py` | Model persistence, evaluation and reference-free prediction |
| `physics/` | Mesh geometry and physical observation models |

## Tests

```powershell
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

Tests cover geometric distances, triangulation invariance, coordinate preservation,
analytic gradients, learning, serialization, split leakage and inference without
reference geometry.

## Scope

The current implementation requires all three evidence groups and a consistent
alignment policy. Thermal validation uses instantaneous equilibrium and a
single-wavelength radiance image with a bolometric flux term. Large-mesh distance
calculations require further acceleration.

Available experiments are small controlled evaluations. They do not establish
calibration on independent observational datasets. Missing thermal evidence and
prediction uncertainty require separately validated methods.

See the [methodology](docs/asteroid-validator-v2.md) for the learning objective,
geometric metric and scientific limitations.
