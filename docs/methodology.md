# Methodology

The scientific reasoning behind each stage. This is the material a JOSS/medRxiv
reader would want; it complements the [architecture](architecture.md), which
covers the software structure.

## 1. Synthetic texture phantom

The background parenchyma is a **stationary Gaussian random field (GRF)**: white
noise convolved with an anisotropic Gaussian kernel in the Fourier domain. A
kernel of standard deviation `σ` produces the autocovariance

```
C(r) = exp(-r² / (4σ²))
```

so the lag at which correlation falls to `1/e` is exactly `2σ`. That lag, in
millimetres, *is* the `corr_length` parameter — a controlled, recoverable ground
truth. Per-axis `anisotropy` multipliers stretch it independently along `z, y, x`.

An optional ellipsoidal **lesion** is filled with a second, independent GRF of
different mean, contrast and correlation length, and blended in. The mask is
always the exact geometric ellipsoid, even when an edge blur softens the
intensity boundary to emulate partial-volume mixing.

**Verification.** `measure_correlation_length` estimates the `1/e` lag from the
volume's power spectrum (Wiener–Khinchin). Across seeds it recovers the requested
correlation length within a few percent and monotonically — a self-check that the
generator produces the texture it claims.

## 2. Acquisition simulation

The same texture is observed under many "scanners" by applying deterministic,
physically motivated degradations:

- **Blur (MTF).** An anisotropic Gaussian point spread function; a requested full
  width at half maximum maps to `σ = FWHM / (2√(2 ln 2))`.
- **Slice profile.** The `z`-only special case of the blur.
- **Noise.** Zero-mean Gaussian, optionally coloured by reusing the GRF machinery
  to emulate the correlation a reconstruction kernel imparts. CT noise variance is
  inversely proportional to dose, so the standard deviation scales as `1/√dose`.
- **Resampling.** Spline interpolation onto a new voxel grid — the acquisition
  axis radiomics features are most notoriously sensitive to.
- **Quantisation.** HU rounding onto a uniform grid.

Effects are applied in acquisition order (blur → resample → noise → quantise):
blur band-limits the signal the grid samples, and noise arises on the
reconstructed grid, so it is added after resampling. Only noise is stochastic and
seeded; within a pinned environment the same phantom, parameters and seed give the same acquisition.

## 3. IBSI-aligned feature core

All eleven IBSI feature families are implemented from first principles:
intensity statistics and histogram; the intensity-volume histogram; morphology
and local intensity; and the six texture families GLCM, GLRLM, GLSZM, GLDZM,
NGTDM and NGLDM, each over their IBSI aggregations.

**Discretisation** defaults to fixed bin size for calibrated (HU) scales in this
framework, so a bin keeps the same physical meaning across images; IBSI also
defines and benchmarks fixed-bin-number configurations.

**Benchmarking is the load-bearing claim.** IBSI publishes a 5×4×4 digital phantom
and benchmark values for every feature. `rphantom` matches **all 482** published
digital-phantom values at the reported precision (three significant figures) and
within the applicable IBSI tolerances. Each is asserted once, so a regression
names the feature that broke. This validates the tested feature definitions and
aggregation settings; it does not by itself establish compliance across every
IBSI preprocessing configuration.

Some definitions the tiny phantom cannot discriminate are pinned separately and
documented honestly as such:

- the nearest-rank vs linear-interpolation percentile definition (identical on
  this phantom);
- GLDZM's face-connected vs Chebyshev border distance (identical because every
  phantom zone touches the ROI border) — pinned on a purpose-built volume instead.

Several definitions were *recovered from the reference values themselves*: NGLDM
dependence counts the centre voxel (otherwise low-dependence emphasis divides by
zero on an isolated voxel), and the morphology PCA axes use the sample covariance
(`1/(n-1)`), not the population one — a 0.7 % discrepancy the reference values
exposed.

## 4. Stability atlas

Given phantoms (the *targets*) observed under acquisition conditions, each
feature reduces to two agreement statistics:

- **ICC(2,1)** — the two-way random-effects, single-measurement, absolute-agreement
  intraclass correlation. Across a `targets × conditions` table it is the fraction
  of a feature's variance that is genuine texture-to-texture signal rather than
  acquisition noise. 1 is perfect reproducibility.
- **CCC** — Lin's concordance correlation coefficient of a degraded condition
  against the reference, combining precision (Pearson `r`) and accuracy (bias).

Both are implemented from their definitions and validated against `pingouin`
(where ICC(2,1) is labelled `ICC(A,1)`). The atlas ranks features from least to
most reproducible — the actionable output for choosing robust features.

## 5. Physics-based normalisation

Feature drift under acquisition is not treated as a nuisance to be regressed out
against a cohort, but as a *known physical response* to be inverted. The workflow:

1. **Calibrate** — sweep one descriptor (e.g. noise σ) on a phantom and fit a
   simple invertible model to the feature's response. Intensity variance, for
   instance, follows `var = var₀ + σ²` with `R² = 1.000`.
2. **Correct** — map a feature measured at a known descriptor value back to the
   reference by inverting the model.
3. **Report residual** — a feature whose calibration the model cannot describe
   (`R² < 0.9`) is refused, not silently "corrected". Normalisation you cannot
   trust is worse than none.

Worked end to end in `examples/run_stability_atlas.py`: noise inflates intensity
variance from 625 to 1526 HU², and normalisation returns every value to ~625, the
noiseless truth.

## Reproducibility

Every result in this project is reproducible from a seed and the committed code.
The IBSI fixtures are regenerated from their authoritative sources by
`scripts/fetch_ibsi_reference.py`; the test suite (716 tests) runs offline and
deterministically.

## Key references

- Zwanenburg et al., *The Image Biomarker Standardization Initiative*, Radiology
  295(2):328–338, 2020. <https://doi.org/10.1148/radiol.2020191145>
- Shrout & Fleiss, *Intraclass correlations*, Psychological Bulletin 86(2), 1979.
- Lin, *A concordance correlation coefficient to evaluate reproducibility*,
  Biometrics 45(1), 1989.
- IBSI reference manual and data: <https://ibsi.readthedocs.io>
