# radiomics-phantom

**IORN-002 — A Physics-Based Stability Atlas for IBSI Radiomics Features Using Synthetic Digital Phantoms**

![CI](https://github.com/Institute-of-One/radiomics-phantom/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![IBSI](https://img.shields.io/badge/IBSI--1-482%2F482%20reference%20values-brightgreen)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21309875.svg)](https://doi.org/10.5281/zenodo.21309875)

Radiomics features are notoriously sensitive to how an image was acquired. Separating that
sensitivity from real biological signal normally requires patient data, which cannot be shared.
This project sidesteps the problem: every volume is **generated from a seed**, its texture
statistics are **known by construction**, and every feature is computed **from first principles**.
The result is a citable research kernel that anyone can rerun end to end.

> Part of *Institute of One* (IORN-002).

> **Scope.** This is a methodological software contribution using synthetic digital phantoms
> only. It involves no human subjects, no patient data, and makes no clinical performance
> claims; institutional review, informed consent, and clinical validation are therefore not
> applicable.

**Documentation:** [Architecture](docs/architecture.md) · [Methodology](docs/methodology.md) ·
[API reference](docs/api.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Design constraints

These are enforced, not aspirational:

- **Pure Python.** Runtime dependencies are only `numpy`, `scipy`, `scikit-image`, `pingouin`,
  `matplotlib`. PyRadiomics is not used. No deep learning.
- **Own IBSI core.** Features are implemented directly against the IBSI reference definitions,
  with fixed bin-size discretisation. MIRP / PySERA are permitted *only* as development-time
  oracles in CI, never as runtime dependencies.
- **Deterministic.** Every generator takes a `seed`; the same seed yields a bit-identical volume.
- **No silent failure.** Degenerate input raises a `ValueError` with a message that says what to
  change. NaN is never swallowed.
- **No patient data.** No DICOM, no scans, nothing derived from a human subject — ever.

## Status

| Module | Purpose | State |
|---|---|---|
| `rphantom/phantom.py` | Deterministic synthetic texture phantoms | **implemented** |
| `rphantom/features.py` | IBSI-compliant feature core | **complete** (11 of 11 families) |
| `rphantom/acquisition.py` | Simulated acquisition degradation | **implemented** |
| `rphantom/stability.py` | ICC / CCC stability atlas | **implemented** |
| `rphantom/normalize.py` | Physics-based normalisation | **implemented** |
| `apps/phantom_studio.py` | Interactive GUI prototype | **implemented** |

Feature families, and how many of IBSI's 482 published digital-phantom reference values
each one accounts for:

| IBSI | Family | Features | Reference values | State |
|---|---|---|---|---|
| 3.3 | Intensity statistics | 18 | 18 | **verified** |
| 3.4 | Intensity histogram | 23 | 23 | **verified** |
| 3.6 | Grey level co-occurrence (GLCM) | 25 | 150 | **verified** |
| 3.7 | Grey level run length (GLRLM) | 16 | 96 | **verified** |
| 3.8 | Grey level size zone (GLSZM) | 16 | 48 | **verified** |
| 3.9 | Grey level distance zone (GLDZM) | 16 | 48 | **verified** |
| 3.10 | Neighbourhood grey tone difference (NGTDM) | 5 | 15 | **verified** |
| 3.11 | Neighbouring grey level dependence (NGLDM) | 17 | 51 | **verified** |
| 3.1 | Morphology | 25 | 25 | **verified** |
| 3.2 | Local intensity | 2 | 2 | **verified** |
| 3.5 | Intensity-volume histogram | 6 | 6 | **verified** |

**All 482** published reference values are reproduced. IBSI leaves four further
morphology features — the OMBB and MVEE bounding-shape densities — unstandardised,
with no reference value; those are the only IBSI-1 features not implemented.

The texture families count more reference values than features because IBSI publishes each
one under every aggregation method. The directional families (GLCM, GLRLM) have six —
`2D_avg`, `2D_comb`, `2_5D_avg`, `2_5D_comb`, `3D_avg`, `3D_comb` — and the zone,
neighbourhood and dependence families have three: `2D`, `2_5D`, `3D`. `rphantom` implements
all of them, because the choice of aggregation is itself a source of feature variability that
this project exists to quantify.

## Install

Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
from rphantom import generate_texture_phantom, measure_correlation_length

phantom = generate_texture_phantom(
    size=(64, 64, 64),
    spacing=(1.0, 1.0, 1.0),   # mm, in (z, y, x) order
    corr_length=6.0,           # 1/e autocorrelation length, mm
    anisotropy=(2.0, 1.0, 1.0),  # z-texture is twice as long-ranged
    hu_mean=40.0,
    hu_sd=25.0,
    lesion=True,
    seed=0,
)

phantom.volume           # float32 (64, 64, 64), HU-like, indexed (z, y, x)
phantom.mask             # bool, the exact ground-truth lesion ellipsoid
phantom.ground_truth     # every generative parameter, echoed back

# The texture really has the correlation length that was asked for:
measure_correlation_length(phantom.volume, phantom.spacing, axis=0)  # ~= 12 mm
```

Render a slice:

```bash
python examples/render_phantom_slice.py --seed 0 --out outputs/phantom_slice.png
```

Extract IBSI features from the lesion:

```python
from rphantom import discretise, glcm_features, glszm_features, intensity_statistics

# Fixed bin size is the IBSI default for calibrated (HU) intensities.
disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

intensity_statistics(phantom.volume, phantom.mask).to_dict()  # {'stat_mean': ..., ...}
glcm_features(disc, "3D_comb").to_dict("3D_comb")             # {'cm_contrast_3D_comb': ..., ...}
glszm_features(disc, "3D").to_dict("3D")                      # {'szm_sze_3D': ..., ...}
```

Every feature set is a frozen dataclass with named fields, and `to_dict()` re-keys it by
official IBSI feature tag so results can be joined against the published tables.

The texture families take a `Discretisation`; morphology, local intensity and the
intensity-volume histogram take the raw volume, mask and voxel `spacing`, because they are
defined on the physical shape and the uncalibrated intensities. Morphology raises on a
constant-intensity ROI — Moran's I and Geary's C are then `0/0` — rather than return `nan`.

Observe the same phantom under a simulated scanner, then extract features from the result:

```python
from rphantom import simulate_acquisition, discretise, glcm_features

acq = simulate_acquisition(
    phantom,
    psf_fwhm_mm=2.0,          # reconstruction-kernel blur (MTF), mm
    slice_fwhm_mm=3.0,        # through-plane slice profile, mm
    new_spacing=(2.0, 1.0, 1.0),  # resample onto a coarser z grid
    noise_sigma=20.0,         # noise at the reference dose, HU-like
    noise_correlation_mm=1.0, # coloured like a reconstruction kernel
    dose=1.0,                 # noise scales as 1 / sqrt(dose)
    quantise_step=1.0,        # HU rounding
    seed=1,
)
acq.volume     # float32, degraded, on the resampled grid
acq.mask       # the ROI, resampled to match
acq.settings   # every parameter applied, to key features against

disc = discretise(acq.volume, acq.mask, method="fbs", bin_width=25.0)
glcm_features(disc, "3D_comb").contrast   # rises with noise, falls with blur
```

Only noise is stochastic; the same phantom, parameters and seed give a bit-identical
acquisition. `examples/acquisition_sweep.py` sweeps dose and kernel and tabulates a feature —
a miniature of the stability atlas to come.

Rate every feature's reproducibility, then undo the drift on one that the physics explains:

```python
from rphantom import build_stability_atlas, calibrate_response, normalise_feature

# Sweep textures (targets) x acquisition settings (conditions); rate each feature.
atlas = build_stability_atlas(phantoms, conditions)
atlas.ranked(by="icc")[:5]          # the five least reproducible features (ICC, CCC)

# Physics-based normalisation: variance grows as var0 + sigma^2 under noise.
curve = calibrate_response(sigmas, variances, model="power", power=2.0)
normalise_feature(curve, measured_variance, sigma)   # -> back to the noiseless value
```

`ICC(2,1)` and Lin's `CCC` are implemented from first principles and validated against
`pingouin` in the test suite. Normalisation refuses (raises) on a feature whose calibration
fit is poor, rather than return a correction the physics does not support.
`examples/run_stability_atlas.py` runs the whole pipeline.

## Interactive prototype

`apps/phantom_studio.py` is a desktop GUI — standard-library tkinter plus matplotlib, no new
dependency — for exploring feature stability by hand. Drag a slider for correlation length,
dose, blur or voxel size and watch the reference and degraded slices, and a live table of IBSI
features, update together:

```bash
python apps/phantom_studio.py
```

Its compute logic lives in `apps/studio_core.py` and is tested headlessly.

## How the phantom is built

The background is a stationary **Gaussian random field**: white noise convolved with an
anisotropic Gaussian kernel in the Fourier domain. A kernel of standard deviation `sigma`
produces an autocovariance `exp(-r^2 / (4 sigma^2))`, so the lag at which correlation falls to
`1/e` is exactly `2 sigma` — that lag, in millimetres, *is* the `corr_length` parameter. Per-axis
`anisotropy` multipliers stretch it independently along `z`, `y` and `x`.

An optional ellipsoidal **lesion** is filled with a second, independent field of different mean,
contrast and correlation length, and blended in. `mask` is always the exact geometric ellipsoid,
even when `lesion_edge_blur_mm` softens the intensity boundary to emulate partial-volume mixing.

Because the convolution is circular (FFT), a correlation length longer than half the field extent
would wrap the field onto itself. Rather than return a plausible-looking volume with the wrong
texture, the generator raises.

## How the features are verified

IBSI publishes a 5×4×4 digital phantom together with reference values for every feature, at a
tolerance of **zero**: a compliant implementation must reproduce each one exactly, to the three
significant digits at which they are published. `tests/test_features_ibsi.py` asserts precisely
that, once per reference value — currently **449 of 482**, being every value for the eight
families implemented so far.

The phantom and the reference table are not transcribed by hand. `scripts/fetch_ibsi_reference.py`
downloads them from their authoritative sources and regenerates `tests/ibsi_reference.py`:

```bash
python scripts/fetch_ibsi_reference.py   # needs network access
```

What the digital phantom *cannot* verify is recorded rather than glossed over. Three examples,
each with a test that says so out loud:

- It does not discriminate the nearest-rank percentile definition from linear interpolation.
  Pinning that requires the IBSI CT benchmark (configurations A–E), which is future work.
- It does not discriminate GLDZM's face-connected border distance from a Chebyshev one, because
  every zone in the phantom touches the ROI border. `tests/test_features_zones.py` pins the
  metric instead, on a volume whose only nearby exit is diagonal.
- It is too lumpy to exercise morphology's geometry, so `tests/test_features_morphology.py`
  checks volume, sphericity and the PCA axes against analytic cubes, spheres and boxes.

## Tests

```bash
python -m pytest
```

The suite pins the contract rather than the implementation: bit-exact reproducibility across
seeds, well-formed output, ground truth that echoes the request, an *empirically measured*
correlation length that recovers the requested one (within 15%) and increases monotonically with
it, the 449 IBSI reference values, run-length and zone decompositions on volumes solvable by
hand, and an explicit exception for every malformed input and every mathematically undefined
feature.

## Preprint

The accompanying manuscript lives in [`paper/`](paper/) as part of the version-controlled
research package: the canonical Markdown, a compiled PDF with embedded figures, a
deterministic figure/number generator, and a medRxiv submission kit. Every quoted value is
reproduced by `paper/make_figures.py`.

## Citing

Archived on Zenodo with a DOI that always resolves to the latest version:
**[10.5281/zenodo.21309875](https://doi.org/10.5281/zenodo.21309875)**. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff); GitHub's *Cite this repository* button renders
a ready-made citation from it.

## Licence

Code is released under the [MIT License](LICENSE). The accompanying prose and figures are
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`tests/ibsi_reference.py` embeds the IBSI-1 digital phantom
([theibsi/data_sets](https://github.com/theibsi/data_sets), CC BY 4.0) and reference values from
the [IBSI-1 submission table](https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx),
reproduced with attribution. It contains no patient data.
