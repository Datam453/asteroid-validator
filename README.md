# Asteroid reconstruction evaluation

Evaluate supplied asteroid meshes against optical and thermal observations.
The system combines silhouette, photometric and thermal consistency measures
to assess reconstruction quality.

## Versions

- **[V2](V2/README.md)** estimates geometric error through supervised regression.
  Training uses reference meshes. Inference requires candidate meshes,
  observations and observing conditions.
- **[V1](v1/README.md)** ranks candidates using validator weights selected by grid
  search against reference-derived rankings.

## Installation

Python 3.11 or later is required.

```powershell
python -m venv V2/.venv
& V2/.venv/Scripts/python.exe -m pip install -e ./V2
```

## Training and evaluation

Prepare observation manifests and a case list with explicit training,
validation and test splits. See the [data contract](V2/docs/data-contract.md).

```powershell
& V2/run.ps1 build-dataset --cases cases.json --output dataset.json --samples 2048
& V2/run.ps1 train --dataset dataset.json --output experiment --epochs 2500
```

Each training example contains the physics-validator responses of one candidate
and its measured discrepancy from a reference surface. All candidates and views
of an asteroid remain in the same split.

See [the methodology](V2/docs/asteroid-validator-v2.md) for the geometric metric,
learning objective, score interpretation and evaluation criteria.

## Validation

```powershell
& V2/.venv/Scripts/python.exe -m unittest discover -s V2/tests -v
```

Current evidence consists of small controlled experiments. Performance on
independent observational datasets remains to be established. Thermal validation
currently assumes instantaneous equilibrium.

Runtime environments, experimental outputs and meeting materials remain outside
version control.
