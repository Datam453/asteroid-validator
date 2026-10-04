# Asteroid reconstruction error estimation

## Objective

Estimate geometric discrepancy of a supplied asteroid mesh from its consistency
with observed silhouette, optical brightness and thermal evidence. Reference
geometry supplies supervised training labels and held-out evaluation. It is
unavailable to the deployed evaluator.

The score has a defined geometric interpretation. Sorting predicted errors also
provides a ranking of candidate reconstructions.

## Inputs

| Input | Purpose |
|---|---|
| Candidate mesh | Surface to evaluate |
| Optical image and reviewed silhouette | Brightness and boundary constraints |
| Calibrated thermal radiance and flux | Thermal observation constraints |
| Camera, illumination and physical metadata | Conditions required by validators |
| Reference mesh during training and evaluation | Geometric-error measurement |

The initial model requires all three evidence groups. Substituting zero for
missing evidence would incorrectly imply perfect agreement.

## Geometric-error definition

For candidate surface (M) and reference surface (M^*), define

[
ar d(M	o M^*)=
rac{1}{A(M)}int_{pin M}min_{qin M^*}|p-q|,dA(p).
]

The normalized bidirectional error is

[
D(M,M^*)=
rac{ar d(M	o M^*)+ar d(M^*	o M)}{2d^*},
]

where (d^*) is the reference diameter. Multiplying the dimensionless score by
100 expresses mean surface discrepancy as a percentage of that diameter.

Surface sampling is uniform by area, and distances are measured to triangle
surfaces rather than vertices. Different triangulations of the same surface
preserve the underlying metric. Sampling introduces numerical uncertainty.
Remeshing that changes the surface changes its measured error. Both directions
account for omitted and additional geometry.

Candidate and reference meshes must share coordinates and physical scale.
Reference-assisted alignment must not conceal reconstruction errors or enter
deployment feature extraction. Check convergence at the intended sample count
and mesh resolution.

## Physics features

Each candidate produces silhouette, photometric and thermal mismatch values.
Feature extraction uses observations and observing metadata only. Alignment
follows a fixed observation-only policy independent of learned weights.

Median feature scales use training objects and are saved with the model.
Inference uses the same scales and feature definitions.

## Evaluator

[
w_i=operatorname{softmax}(oldsymbolalpha)_i,
qquad a=operatorname{softplus}(gamma),
]

[
f_	heta(mathbf x)=
operatorname{softplus}left(b+asum_i w_i x_i
ight),
qquad 	heta=(oldsymbolalpha,gamma,b).
]

Weights are nonnegative and sum to one. The learned scale and offset provide
numerical calibration. Relative weights apply after normalization and are not
universal physical constants.

Monotonicity in each mismatch feature is an initial assumption. Compare with
unconstrained regression or nonlinear models if independent evaluation shows
that this assumption limits accuracy.

## Training

[
L(	heta)=rac{1}{N}sum_k
left[f_	heta(mathbf x_k)-D(M_k,M_k^*)
ight]^2.
]

Adam updates evaluator parameters using analytic chain-rule gradients. Physics
responses and geometric labels can be precomputed. Training does not require
backpropagation through alignment, rendering or distance measurement.

All candidates and views of an asteroid remain in one split. Fit normalization
and parameters on training objects, select a checkpoint with validation objects,
then freeze the evaluator before testing. Repeated views do not increase the
number of independent asteroids.

The loss measures prediction error. An inaccurate mesh need not produce a large
training loss if the evaluator predicts its error accurately.

## Evaluation

- MAE: mean absolute difference between predicted and measured geometric error.
- RMSE: root mean squared prediction error, emphasizing large mistakes.
- Bias: mean signed prediction error.
- Rank correlation and best-candidate selection: consequences of the ratings.
- Selection regret: excess measured error over the best available candidate.

Neither MAE nor RMSE allows positive and negative mistakes to cancel. Zero means
perfect predictions rather than necessarily perfect meshes.

Compare with individual validators, constant prediction, simple regression and
the V1-style ranking baseline. Test independent asteroids and changes in
observing conditions, material assumptions and noise. Examine hidden surfaces
and severe local defects in addition to mean surface distance.

## Scope

The score estimates geometric error conditional on available features and the
training distribution. It is not a correctness probability or certified bound.
Different surfaces can explain the same observations and produce similar
validator responses despite different geometric errors. Training cannot restore
information absent from those responses.

Thermal validation currently uses instantaneous equilibrium, a single-wavelength
radiance image and a bolometric flux term. Time-dependent heat transport,
uncertain materials, missing evidence and uncertainty estimation require further
validation.

Current evidence consists of small controlled experiments. Generalization and
calibration on independent observational datasets remain open research questions.
The implementation includes reference-free inference, saved preprocessing,
object-level splits and baseline evaluation.
