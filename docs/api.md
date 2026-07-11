# API reference

The public surface of `rphantom`, grouped by pipeline stage. Every symbol listed
is exported from the top-level package (`from rphantom import ...`). Signatures
are abridged to their key parameters; see each function's docstring for the full
contract, including the exceptions it raises.

Conventions: arrays are `(z, y, x)`, spacing is `(dz, dy, dx)` in mm, feature
tags are the official IBSI strings.

## Phantom generation — `rphantom.phantom`

```python
generate_texture_phantom(size=(64,64,64), spacing=(1,1,1), corr_length=6.0,
                         anisotropy=(1,1,1), hu_mean=40, hu_sd=25,
                         lesion=True, ..., seed=0) -> Phantom
```
Deterministic synthetic 3D texture phantom as an anisotropic Gaussian random
field with an optional embedded ellipsoidal lesion.

- `Phantom` — frozen dataclass: `volume` (float32), `mask` (bool), `spacing`,
  `ground_truth` (dict of the generative parameters), `seed`.
- `measure_correlation_length(volume, spacing, axis=0) -> float` — recover the
  `1/e` correlation length from the power spectrum (verification).
- `gaussian_random_field(shape, spacing, corr_lengths_mm, rng) -> ndarray` — the
  standardised field generator, reused by the noise model.

## Acquisition — `rphantom.acquisition`

```python
simulate_acquisition(phantom, *, psf_fwhm_mm=0, slice_fwhm_mm=0, new_spacing=None,
                     noise_sigma=0, noise_correlation_mm=0, dose=1.0,
                     quantise_step=0, seed=0) -> Acquisition
```
Observe a phantom under one simulated scanner: blur → resample → noise →
quantise. Deterministic given the seed.

- `Acquisition` — frozen dataclass: `volume`, `mask`, `spacing`, `settings`
  (applied parameters), `ground_truth`, `seed`.
- Composable primitives: `apply_blur`, `apply_slice_profile`, `add_noise`,
  `resample`, `quantise`.
- `AcquisitionError` — raised on malformed parameters or a degenerate result.

## Features — `rphantom.features`

```python
extract_features(volume, mask, spacing=(1,1,1), *, bin_width=25.0,
                 include_morphology=True) -> dict[str, float]
```
The full IBSI feature vector (169 features by default), keyed by tag.

Discretisation and the per-family functions, for finer control:

| Function | IBSI | Returns |
|---|---|---|
| `discretise(volume, mask, method="fbs", bin_width=…)` | 2.7 | `Discretisation` |
| `intensity_statistics(volume, mask)` | 3.3 | `IntensityStatistics` |
| `intensity_histogram(disc)` | 3.4 | `IntensityHistogramFeatures` |
| `glcm_features(disc, aggregation="3D_comb")` | 3.6 | `GLCMFeatures` |
| `glrlm_features(disc, aggregation="3D_comb")` | 3.7 | `GLRLMFeatures` |
| `glszm_features(disc, aggregation="3D")` | 3.8 | `GLSZMFeatures` |
| `gldzm_features(disc, aggregation="3D")` | 3.9 | `GLDZMFeatures` |
| `ngtdm_features(disc, aggregation="3D")` | 3.10 | `NGTDMFeatures` |
| `ngldm_features(disc, aggregation="3D", alpha=0)` | 3.11 | `NGLDMFeatures` |
| `morphology_features(volume, mask, spacing)` | 3.1 | `MorphologyFeatures` |
| `local_intensity_features(volume, mask, spacing)` | 3.2 | `LocalIntensityFeatures` |
| `intensity_volume_histogram(disc)` | 3.5 | `IVHFeatures` |

Every feature dataclass has `to_dict(aggregation=None)` returning `{tag: value}`.
Aggregations: `AGGREGATIONS` (six, for GLCM/GLRLM), `ZONE_AGGREGATIONS` (three).
`FeatureError` is raised on a degenerate ROI.

## Stability — `rphantom.stability`

```python
build_stability_atlas(phantoms, conditions, *, condition_labels=None,
                      bin_width=25.0, include_morphology=True) -> StabilityAtlas
```
Sweep phantoms (targets) × acquisition conditions and rate every feature's
reproducibility.

- `StabilityAtlas.ranked(by="icc", ascending=True)` → `list[FeatureReliability]`.
- `FeatureReliability` — `tag`, `icc`, `ccc_min`, `ccc_mean`, `reference_value`.
- `intraclass_correlation(measurements) -> ICCResult` — ICC(2,1) from a
  `(targets, conditions)` table.
- `concordance_correlation(reference, measured) -> ConcordanceResult` — Lin's CCC.
- `StabilityError` — raised on a degenerate or too-small table.

## Normalisation — `rphantom.normalize`

```python
calibrate_response(descriptors, values, *, model="linear"|"power", power=2.0)
    -> CalibrationCurve
normalise_feature(curve, feature, descriptor, *, reference=0.0,
                  require_trustworthy=True) -> float
```
Fit a feature's response to an acquisition descriptor, then invert it to restore
the reference value. Refuses (raises) when the fit is poor unless overridden.

- `CalibrationCurve` — `model`, `r_squared`, `residual_std`, `is_trustworthy`,
  and `normalise(feature, descriptor, reference)`.
- Models: `LinearResponse` (`a + b·d`), `PowerResponse` (`a + b·dᵖ`).
- `NormalizationError` — raised on malformed input or an uninvertible fit.

## Entry points

| Path | What it does |
|---|---|
| `apps/phantom_studio.py` | interactive GUI over phantom → acquisition → features |
| `examples/render_phantom_slice.py` | save a labelled central-slice figure |
| `examples/acquisition_sweep.py` | dose × kernel feature table |
| `examples/run_stability_atlas.py` | full atlas + physics normalisation |
| `scripts/fetch_ibsi_reference.py` | regenerate IBSI test fixtures from source |
