# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

No tagged release has been published yet; the versions below are development
milestones. The first public release will coincide with the first Zenodo
deposition.

## [Unreleased]

- Nothing yet.

## [0.6.0] — 2026-07-11

Feature-stability atlas, physics-based normalisation, and an interactive prototype.

### Added
- `rphantom/stability.py`: ICC(2,1) and Lin's concordance correlation coefficient,
  implemented from first principles and validated against `pingouin`; a
  `build_stability_atlas` sweep that rates every feature's reproducibility across
  acquisition conditions.
- `rphantom/normalize.py`: physics-based feature normalisation. Calibrates a
  feature's response to an acquisition descriptor and inverts it to restore the
  reference value; refuses (raises) when the calibration fit is poor.
- `rphantom.features.extract_features`: a one-call full IBSI feature vector.
- `apps/phantom_studio.py` and `apps/studio_core.py`: an interactive desktop GUI
  (tkinter + matplotlib, no new dependency) with orthogonal axial/coronal/sagittal
  views, a slice scrubber, a phantom-specification panel and a live feature table.
- `examples/run_stability_atlas.py` and `examples/acquisition_sweep.py`.

## [0.5.0] — 2026-07-11

Acquisition degradation simulator.

### Added
- `rphantom/acquisition.py`: deterministic, physically motivated degradations —
  Gaussian PSF blur, slice-profile averaging, dose-scaled correlated noise,
  grid resampling and HU quantisation — composed by `simulate_acquisition`.

## [0.4.0] — 2026-07-11

Feature core completed.

### Added
- `rphantom/features.py`: morphology (IBSI 3.1), local intensity (3.2) and the
  intensity-volume histogram (3.5). All **482** published IBSI digital-phantom
  reference values are now reproduced exactly.

### Fixed
- Guard against a planar (zero-thickness) ROI in morphology, and against a
  constant-intensity ROI (Moran's I / Geary's C are then undefined): both raise
  rather than divide by zero.

## [0.3.0] — 2026-07-11

Zone, neighbourhood and dependence texture families.

### Added
- `rphantom/features.py`: GLSZM (3.8), GLDZM (3.9), NGTDM (3.10) and NGLDM (3.11),
  each over the three IBSI zone aggregations. Coverage reaches 449 reference values.

## [0.2.0] — 2026-07-10

IBSI feature core begins.

### Added
- `rphantom/features.py`: fixed-bin-size/number discretisation, intensity
  statistics (3.3), intensity histogram (3.4), GLCM (3.6) and GLRLM (3.7) over all
  six IBSI aggregations. 287 reference values reproduced exactly.
- `scripts/fetch_ibsi_reference.py`: regenerates the test fixtures directly from
  the authoritative IBSI phantom and value table (no hand transcription).

## [0.1.0] — 2026-07-10

Project skeleton and phantom generator.

### Added
- `rphantom/phantom.py`: deterministic synthetic 3D texture phantoms as anisotropic
  Gaussian random fields with an optional embedded ellipsoidal lesion and full
  ground truth; `measure_correlation_length` for verification.
- Repository scaffold: packaging, licence (MIT), `CITATION.cff`, `.zenodo.json`,
  and the first example and tests.

[Unreleased]: https://github.com/Institute-of-One/radiomics-phantom/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.6.0
[0.5.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.5.0
[0.4.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.4.0
[0.3.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.3.0
[0.2.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.2.0
[0.1.0]: https://github.com/Institute-of-One/radiomics-phantom/releases/tag/v0.1.0
