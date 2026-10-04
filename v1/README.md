# Asteroid Validator V1

V1 ranks supplied candidate meshes by silhouette, photometric and thermal
consistency with observations. Nonnegative validator weights are selected by
grid search to maximize mean training rank correlation.

[V2](../V2/README.md) extends this approach to supervised prediction of geometric
error.

## Installation

Using Python 3.11 or later, from this directory:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -e .
```

## Match supplied meshes

```powershell
& .venv/Scripts/asteroid-outline-match.exe --help
& .venv/Scripts/asteroid-calibrate-weights.exe --help
& .venv/Scripts/asteroid-outline-view.exe --help
```

The outline matcher accepts optical images and NPZ or triangular OBJ meshes. It
fits image-plane alignment before scoring candidates. A reviewed silhouette
mask can replace automatic segmentation. Optical and thermal evidence are
optional inputs to the combined scoring workflow.

Calibration evaluates each weight combination across the training cases and
selects the combination with the highest average Spearman rank correlation.
The saved weights remain fixed for held-out evaluation.

## Tests

```powershell
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Interpretation

V1 scores support candidate ordering without calibrated geometric-error units.
Scale each validator consistently before combining scores. Training rankings do
not establish performance on independent asteroids.

The thermal model uses instantaneous equilibrium. Material uncertainty,
observation mismatch and ambiguous hidden geometry can affect candidate scores.
