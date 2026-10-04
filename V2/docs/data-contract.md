# V2 data contract

## Case list for training

Paths are relative to the case-list file, or absolute. All observations of an
asteroid must remain in one split. Use `view_id` when an object has multiple views.
Train, validation and test each need at least one usable candidate record.

```json
{
  "cases": [
    {"asteroid_id": "asteroid-a", "split": "train", "manifest": "asteroid-a/manifest.json"},
    {"asteroid_id": "asteroid-b", "split": "validation", "manifest": "asteroid-b/manifest.json"},
    {"asteroid_id": "asteroid-c", "split": "test", "manifest": "asteroid-c/manifest.json"}
  ]
}
```

The builder reads reference geometry only to create labels. Features are
computed through a separate path that does not load that geometry. Cross-split
object IDs and identical reference-file hashes are rejected. These checks cannot
detect every near-duplicate or related shape family; dataset curation must also
keep closely related shapes together.

## Observation manifest

Paths within a manifest are relative to that manifest, or absolute. This example
uses a known orthographic camera looking from positive z. Mesh vertices must be
in the declared common frame, with consistent unit scale and face winding.

```json
{
  "schema_version": 1,
  "alignment_policy": "supplied-camera",
  "ground_truth_mesh": "ground_truth/reference.npz",
  "candidates": [
    {"id": "candidate_000", "path": "candidates/candidate_000.obj"},
    {"id": "candidate_001", "path": "candidates/candidate_001.npz"}
  ],
  "observations": {
    "optical": "observations/optical.png",
    "silhouette": "observations/silhouette.png",
    "light_direction": [0.35, 0.42, 0.84],
    "albedo": 0.62,
    "ambient": 0.06,
    "exposure": 0.92,
    "projection": {
      "yaw_deg": 0,
      "pitch_deg": 0,
      "roll_deg": 0,
      "scale_px_per_unit": 16,
      "translation_px": [0, 0]
    },
    "thermal": {
      "radiance_image": "observations/infrared_10um.npy",
      "bolometric_flux_w_m2": 0.3,
      "sun_direction": [0.35, 0.42, 0.84],
      "observer_direction": [0, 0, 1],
      "bond_albedo": 0.08,
      "solar_flux_w_m2_at_1au": 1361,
      "heliocentric_distance_au": 1.2,
      "observer_distance_m": 20000,
      "mesh_unit_m": 500,
      "wavelength_um": 10
    }
  }
}
```

Replace example values with the actual observation parameters. Optical images
are read as normalized grayscale. Thermal images must be finite, non-negative
NumPy arrays matching optical dimensions, in spectral radiance units consistent
with the wavelength (W m^-2 sr^-1 µm^-1 at 10 µm). With wavelength omitted, the
equilibrium model expects bolometric radiance. The flux measurement is
bolometric received flux in W m^-2; it is not interchangeable with a
monochromatic image value. A binary reviewed silhouette is required.

The three current features are a contour/overlap/area silhouette score, mean
Lambertian normal-cone inconsistency, and an equally weighted combination of
relative bolometric flux error and resolved thermal image-pattern error. These
reuse V1 physics. Optical and thermal image comparisons operate on overlapping
valid pixels; the report exposes their coverage and silhouette disagreement.

### Alignment policies

- `supplied-camera` uses declared projection metadata for all candidates. This is
  the supplied-camera policy, appropriate for controlled common-frame experiments.
- `outline-fixed` runs the frozen silhouette-only pose/scale/translation search;
  `projection` is not needed. Optical and thermal scores are then measured at
  that pose. Learned evaluator weights cannot change pose selection. Physical
  thermal flux still uses the candidate's declared physical mesh scale; image
  scale fitting does not silently rescale it.

Do not mix the policies in one dataset/model. Their error distributions differ.
A model trained with supplied-camera alignment cannot be used on outline-fixed features without training
a matching evaluator. Reference-based alignment is not used to extract features.
Label geometry must already share a declared frame and scale; the label code
does not rotate or rescale candidates against the reference. Independently
Supplied meshes require documented coordinate and unit conversion.

## Deployment manifest

Use the same observation/candidate schema, but omit `ground_truth_mesh`. Neither
`extract` nor `predict` requires it. `predict` consumes only the three validated
feature columns, not labels or identifiers as learning inputs.

## Mesh formats

NPZ files contain finite `vertices` (N-by-3 floating point) and integer `faces`
(M-by-3 vertex indices). OBJ polygons are fan-triangulated; positive and negative
indices are supported. Nonconvex OBJ polygons should be triangulated correctly
upstream. Loading preserves translation and scale. Zero-area triangles are
ignored for the surface-distance calculation; meshes with no valid surface fail.

## Saved outputs

- `dataset.json`: fixed features, geometric labels, splits, reference-file hashes,
  sampling details, diagnostics and failed-candidate records.
- `model.json`: frozen model and normalization, feature schema/version, alignment
  policy and training metadata.
- `evaluation.json`: absolute-error metrics, ranking consequences, baseline
  results and failure counts.
- `training_history.json`: training and validation losses; the test set is not
  used for stopping.
- `ratings.json`: deployment predictions sorted by estimated geometric error,
  with feature values and observation alignment.

The prediction is a dimensionless error estimate. Multiplying it by 100 gives
percent of reference diameter under the training metric, not percent correctness.
It is not a probability or certified bound.
