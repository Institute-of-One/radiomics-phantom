---
title: "A Stability Atlas for IBSI Radiomics Features Using Synthetic Digital Phantoms, with Proof-of-Concept Physics-Based Normalisation"
author:
  - Shuji Yamamoto, PhD
date: "Preprint --- 2026-07-11"
abstract: |
  Radiomics features are strongly sensitive to image acquisition, and separating that sensitivity from biological signal usually requires repeated patient scans that cannot be shared. We present an open, fully synthetic framework (radiomics-phantom) that maps and, as a proof of concept, corrects radiomics feature instability without any patient data. Deterministic three-dimensional texture phantoms are generated as anisotropic Gaussian random fields with known ground truth and an optional embedded lesion. An independently implemented, IBSI-aligned feature core covers all eleven IBSI-1 feature families and matched all 482 published digital-phantom benchmark values at the reported precision and within the applicable IBSI tolerances. An image-domain acquisition simulator applies a point-spread blur, through-plane slice-profile averaging, dose-scaled correlated noise, resampling, and quantisation. Per-feature reproducibility across a five-by-five texture-by-acquisition sweep is summarised by the absolute-agreement intraclass correlation ICC(2,1), with Lin's concordance correlation coefficient reported as a secondary metric in the supplement; ICC values span the full range (median 0.02) and are interpreted as exploratory rankings within this acquisition envelope rather than universal reliability estimates. As a proof of concept, intensity variance under additive Gaussian noise was normalised using the analytic variance–noise relationship, with calibration and a held-out evaluation on different noise realisations returning the variance to its noiseless value; features whose response the model cannot describe (for example mean intensity) are refused rather than corrected. All code, the phantom generator, the feature core, and a 716-test suite are released openly (MIT), archived on Zenodo, and checked in continuous integration. The contribution is a reproducible, patient-data-free testbed for radiomics feature stability and a proof-of-concept for physics-grounded normalisation.
keywords: "radiomics; reproducibility; IBSI; digital phantom; feature stability; image acquisition; open source"
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
colorlinks: true
papersize: a4
---

**Author affiliation.** Shuji Yamamoto, PhD — Institute of One (supported by Lisit Co., Ltd., Japan; TexelCraft OÜ, Estonia).

**Corresponding author.** Shuji Yamamoto — yamamoto@lisit.jp · ORCID [0000-0001-9211-1071](https://orcid.org/0000-0001-9211-1071).

**Software.** `radiomics-phantom` v0.6.0 (MIT). Code: <https://github.com/Institute-of-One/radiomics-phantom>. Archive DOI: [10.5281/zenodo.21309875](https://doi.org/10.5281/zenodo.21309875) (all versions).

---

## 1. Background and Motivation

Radiomics extracts large numbers of quantitative features from medical images to characterise tissue phenotype [1]. A persistent obstacle to clinical translation is that many features are highly sensitive to how the image was acquired and reconstructed — dose, reconstruction kernel, slice thickness, and voxel size — so that measured values reflect the scanner as much as the biology; phantom and multi-parameter studies report that a large fraction of CT radiomic features are non-reproducible under acquisition change [7,8]. The Image Biomarker Standardization Initiative (IBSI) has standardised feature definitions and provides a digital reference phantom with benchmark values [2], and widely used toolkits implement these definitions [3]. Standardised *definitions*, however, do not by themselves make feature *values* reproducible across acquisitions, and residual differences persist even among standards-aligned implementations [9].

Studying that reproducibility normally requires repeated patient scans, which are scarce and cannot be shared, or post-hoc harmonisation against a reference cohort (for example ComBat [6,11]), which treats acquisition effects as nuisance to be regressed out rather than as known physics. We take a different route: a fully synthetic testbed in which the texture is known by construction and the acquisition is a controlled, image-domain transformation. Because nothing derives from a human subject, the entire study is shareable and, for a fixed software environment, reproducible.

**Contribution.** We provide (a) an open, deterministic generator of 3D texture phantoms with ground truth, paired with an independently implemented, IBSI-aligned feature core benchmarked against the full set of published digital-phantom values; (b) a *stability atlas* that ranks every feature's reproducibility across a texture-by-acquisition sweep using ICC(2,1) (with the concordance correlation coefficient as a secondary metric); and (c) a proof-of-concept, physics-based normalisation that corrects a feature using its known response to a measurable acquisition descriptor and refuses features whose response it cannot model.

## 2. Methods

### 2.1 Synthetic texture phantom

The background parenchyma is a stationary Gaussian random field: white noise convolved with an anisotropic Gaussian kernel in the Fourier domain. A kernel of standard deviation *s* produces an autocovariance exp(-r^2 / 4s^2), so the lag at which correlation falls to 1/e is 2*s*; that lag, in millimetres, is the prescribed correlation length, controllable per axis to yield anisotropic texture. An optional ellipsoidal lesion is filled with a second, independent field of different mean, contrast, and correlation length. Every generator takes a seed and, within a fixed software environment, returns the same volume together with its ground truth. An empirical estimator recovers the correlation length from the power spectrum, confirming the field carries the requested texture.

### 2.2 IBSI-aligned feature core

All eleven IBSI-1 families are implemented from their definitions using only array primitives: intensity statistics and histogram, the intensity-volume histogram, morphology and local intensity, and the six texture families (grey-level co-occurrence, run length, size zone, distance zone, neighbourhood grey-tone difference, and neighbouring grey-level dependence), each over their IBSI aggregations. Fixed-bin-size discretisation is used as the default for calibrated HU-like intensities in this framework; IBSI also defines and benchmarks fixed-bin-number configurations. Degenerate regions raise explicit errors rather than returning undefined values. This benchmark validates the tested feature definitions and aggregation settings; it does not by itself establish compliance across every IBSI preprocessing configuration.

### 2.3 Acquisition simulator

The same texture is observed under many simulated scanners by applying deterministic, image-domain surrogates for the dominant acquisition effects: a Gaussian point-spread function for in-plane spatial-resolution loss (specified by full width at half maximum), through-plane averaging for the slice profile, correlated Gaussian noise with dose-dependent amplitude (scaled as 1/sqrt(dose)) optionally coloured like a reconstruction kernel, interpolation onto a different voxel grid, and intensity quantisation. It is not a projection-domain CT reconstruction simulator. As implemented, the in-plane point-spread blur and the through-plane slice-profile averaging are applied together as one anisotropic Gaussian, in the order: spatial-resolution and slice-profile blur → resampling → noise → quantisation. Only noise is stochastic and is seeded; for a fixed environment a given phantom, settings, and seed give the same acquisition.

### 2.4 Stability atlas

A set of phantom textures (the *targets*) is observed under a set of acquisition settings (the *conditions*). For each feature, reproducibility is summarised by the two-way random-effects, single-measurement, absolute-agreement intraclass correlation ICC(2,1) [4] across the target-by-condition table, with the choice and interpretation of ICC form following established guidance [10]. ICC(2,1) is used as an absolute-agreement measure across acquisition conditions treated as a sampled acquisition envelope. Lin's concordance correlation coefficient [5] of each degraded condition against the reference is reported as a secondary metric (supplementary table). Both statistics are implemented from first principles and cross-checked against an independent library (pingouin) in the test suite.

### 2.5 Physics-based normalisation (proof of concept)

As a proof of concept, rather than harmonise against a cohort, a feature's response to one acquisition descriptor is calibrated on the phantom by sweeping that descriptor and fitting a simple invertible model (linear or power); the model is then inverted to map an observed feature back to a reference acquisition. Calibration and evaluation use different noise realisations (separate seeds and held-out noise levels). A feature whose calibration the model cannot describe (coefficient of determination below 0.9) is refused rather than silently corrected. This demonstrates the framework's ability to calibrate and invert a known physical response; it does not establish general normalisation for all radiomics features.

## 3. Implementation and Scope

The software is pure Python (numpy, scipy, scikit-image; matplotlib and an optional tkinter GUI) and is deterministic within a pinned software environment for a fixed seed. It is covered by 716 automated tests; continuous-integration tests verify numerical outputs against predefined tolerances across Python 3.10–3.12. It is released under the MIT licence (text and figures under CC BY 4.0) and archived on Zenodo. An interactive desktop tool (Phantom Studio) exposes the phantom, acquisition, and feature stages for exploration. This preprint reports the open core and its reproducible synthetic benchmarks; no clinical or real-data validation is claimed.

## 4. Validation and Results

### 4.1 IBSI validation

Computed on the IBSI digital phantom, the feature core matched all **482** published digital-phantom benchmark values at the reported precision (three significant figures) and within the applicable IBSI tolerances, across every implemented family and aggregation. This benchmark validates the tested feature definitions and aggregation settings; it does not by itself establish compliance across every IBSI preprocessing configuration. The four bounding-shape density features that IBSI leaves unstandardised are the only IBSI-1 features not implemented. Test fixtures are regenerated directly from the authoritative IBSI sources rather than transcribed, and the benchmark assertions run in continuous integration.

The 482 benchmark values span all families and aggregation methods (for example, each texture feature is benchmarked under several aggregations). The stability atlas below uses a defined 136-feature vector: the 3D-merged aggregation for the directional texture families and the 3D aggregation for the zone, neighbourhood, and dependence families, together with intensity statistics and histogram, and excluding morphology, local-intensity, and intensity-volume-histogram features (which require a lesion mask and are not meaningful on the whole-volume region used here). A machine-readable mapping from every benchmark value to its inclusion in the atlas is provided (`paper/supplementary/feature_inventory.csv`).

![**Figure 1.** A synthetic phantom (top) and the same phantom under a simulated acquisition with reconstruction-kernel blur and additive noise (bottom), shown in three orthogonal planes. The red contour is the ground-truth lesion mask. The vertical stretch of the texture in the coronal and sagittal planes reflects the prescribed z-axis anisotropy.](figures/fig1_phantom.png)

### 4.2 Acquisition sensitivity

The degradations move features in the direction physics predicts. For a representative grey-level co-occurrence contrast feature on a fixed texture, adding noise of increasing standard deviation raised contrast monotonically (from 0.26 with no noise to 0.60, 2.28, and 8.27 at noise levels of 10, 25, and 50 HU-like units), while increasing reconstruction-kernel blur lowered it (from 0.26 to 0.24, 0.21, and 0.16 at full widths at half maximum of 2, 4, and 6 mm). Independent noise adds in quadrature to the intensity variance, as expected.

### 4.3 Stability atlas

Sweeping five phantom textures across five acquisition conditions (a noiseless reference, two noise levels, a blur, and a combined noise-plus-blur setting) and rating 136 features, reproducibility spanned the whole range of ICC (median 0.02; minimum -0.20; maximum 1.00) under these deliberately harsh conditions (Figure 2). Three features reached ICC above 0.9, but two of these — minimum intensity and dependence-count percentage — are invariant by construction on these phantoms (constant across all textures and conditions); their ICC of 1.00 is a definitional artefact of the estimator on a zero-variance feature rather than measured agreement, and they are flagged as such in the supplementary table. Excluding these constants, the most reproducible feature was cluster shade (ICC 0.988; median concordance 0.99), followed by median intensity (0.86) and histogram skewness (0.82); the least reproducible were dispersion-type histogram descriptors (interquartile range and quartile coefficient of dispersion, ICC below 0). Because the atlas uses five phantom textures and five deliberately selected acquisition conditions, the ICC values should be interpreted as exploratory rankings within this acquisition envelope rather than universal feature-reliability estimates, and the atlas reports point estimates only; uncertainty estimation (for example bootstrap intervals) is left to future work. Per-feature ICC, concordance, value range, and status are provided (`paper/supplementary/stability_atlas.csv`).

![**Figure 2.** Distribution of ICC(2,1) across 136 features under a five-by-five texture-by-acquisition sweep. The dashed line marks the median; the dotted line marks the conventional robustness threshold of 0.9. Three features exceed 0.9, two of which are constant on these phantoms (ICC forced to 1.0; see text).](figures/fig2_atlas.png)

### 4.4 Physics-based normalisation

As a proof of concept, intensity variance was normalised under additive Gaussian noise using the analytic relationship between signal variance and known noise variance: independent zero-mean noise adds in quadrature, so variance follows var0 + b*sigma^2. Calibrating on noise levels 0–20 (one noise realisation) gave a fitted slope b = 1.00 and intercept equal to the noiseless variance (625 HU-like units squared), at R-squared = 1.000 (Figure 3, left). Evaluated on held-out noise levels (25 and 30) generated with a different seed, the raw variance rose to 1254 and 1529, and inverting the calibration returned it to 628 and 628 — within about three units of the noiseless value of 625 (Figure 3, right). This validates the framework's ability to calibrate and invert a known physical response; it does not establish general normalisation for all radiomics features. Consistently, features whose response the power-2 model cannot describe are refused: for example mean intensity, which has no variance-like response to zero-mean noise (fitted R-squared = 0.19, below the 0.9 acceptance threshold), was correctly rejected rather than corrected.

![**Figure 3.** Proof-of-concept physics-based normalisation of intensity variance. Left: variance versus noise standard deviation, with the fitted var0 + b*sigma^2 model (R-squared = 1.000) from the calibration levels (circles) and two held-out levels generated with a different seed (triangles). Right: raw variance diverges with noise (calibration circles, held-out triangles) while the normalised values (squares) return to the noiseless variance (dotted line).](figures/fig3_normalisation.png)

## 5. Reproducibility

Every result in this preprint is deterministic and, within a pinned software environment, re-runs to identical values from a fixed seed and the released code; continuous-integration tests verify numerical outputs against predefined tolerances across Python 3.10–3.12. The IBSI test fixtures are regenerated from their authoritative sources by a script (`scripts/fetch_ibsi_reference.py`); the figures, the supplementary tables, and every quoted number are produced by `paper/make_figures.py` and `paper/make_supplementary.py` and written to `paper/figures/results.json` and `paper/supplementary/` so that text, tables, and figures cannot diverge. The full suite (716 tests, including the 482 IBSI benchmark assertions) runs offline in continuous integration.

## 6. AI-Use Disclosure

This manuscript and the associated software were produced by a human author (S. Yamamoto), who is solely accountable for their content. AI agents were used as tools: code scaffolding and refactoring of the modules, test drafting, figure and script generation, and manuscript drafting were assisted by a large language model (Claude, Anthropic). The author independently re-executed every numerical result reported here (the IBSI validation, the acquisition sweep, the stability atlas, the normalisation, and the test suite) and verified all figures, equations, and claims against the code. No AI system is an author. This disclosure follows ICMJE and COPE guidance: AI is reported as a tool, not credited with authorship.

## 7. Limitations

Synthetic Gaussian-random-field texture is a controlled idealisation, not a substitute for the full complexity of real tissue; the acquisition model captures dominant effects (blur, noise, slice profile, voxel size, quantisation) but not every scanner-specific behaviour; and the normalisation demonstrated here is exact for intensity variance under additive noise, whereas many features will require richer response models or prove irreducible — which the framework flags explicitly rather than correcting blindly. The stability atlas is reported for one illustrative acquisition envelope; conclusions about a specific feature's robustness are conditional on the swept conditions. No clinical, regulatory, or real-data validation is claimed. External calibration of the phantom's noise and resolution to a specific scanner, additional acquisition effects, per-feature response libraries, and validation against public data are planned. As released, `radiomics-phantom` is a research reference, not a clinical or regulatory-grade tool.

## Declarations

**Data and code availability.** All code, the phantom generator, the IBSI feature core, evaluation scripts, and the test suite are openly available at <https://github.com/Institute-of-One/radiomics-phantom> under the MIT license, archived on Zenodo (concept DOI [10.5281/zenodo.21309875](https://doi.org/10.5281/zenodo.21309875), all versions). No patient, clinical, or client data were used; all data in this study are synthetic and produced by the included reproducible generator. The only external reference data are the public IBSI digital phantom and its published values.

**Ethics.** Not applicable. This study involved no human participants, animal subjects, or patient data; only synthetic data were analyzed.

**Competing interests.** S.Y. is the Representative Director of Lisit Co., Ltd. and Chief Executive Officer of TexelCraft OÜ. These entities support Institute of One, an independent research organization. This relationship is disclosed as a potential competing interest. The work used no client or patient data and presents openly licensed research software. The author declares no other competing interests.

**Funding.** This work received no external grant funding. Computing resources and author time were supported in kind by Lisit Co., Ltd. and TexelCraft OÜ.

**Author contributions.** S.Y. is the sole author and is responsible for conceptualization, methodology, software, validation, formal analysis, visualization, and writing. AI tools were used as disclosed in Section 6.

## References

1. Gillies RJ, Kinahan PE, Hricak H. Radiomics: images are more than pictures, they are data. *Radiology.* 2016;278(2):563–577.
2. Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. *Radiology.* 2020;295(2):328–338.
3. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. *Cancer Research.* 2017;77(21):e104–e107.
4. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin.* 1979;86(2):420–428.
5. Lin LI. A concordance correlation coefficient to evaluate reproducibility. *Biometrics.* 1989;45(1):255–268.
6. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics.* 2007;8(1):118–127.
7. Berenguer R, Pastor-Juan MR, Canales-Vázquez J, et al. Radiomics of CT features may be nonreproducible and redundant: influence of CT acquisition parameters. *Radiology.* 2018;288(2):407–415. doi:10.1148/radiol.2018172361.
8. Mackin D, Fave X, Zhang L, et al. Measuring computed tomography scanner variability of radiomics features. *Investigative Radiology.* 2015;50(11):757–765. doi:10.1097/RLI.0000000000000180.
9. McNitt-Gray M, Napel S, Jaggi A, et al. Standardization in quantitative imaging: a multicenter comparison of radiomic features from different software packages on digital reference objects and patient data sets. *Tomography.* 2020;6(2):118–128. doi:10.18383/j.tom.2019.00031.
10. Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine.* 2016;15(2):155–163. doi:10.1016/j.jcm.2016.02.012.
11. Orlhac F, Boughdad S, Philippe C, et al. A postreconstruction harmonization method for multicenter radiomic studies in PET. *Journal of Nuclear Medicine.* 2018;59(8):1321–1328. doi:10.2967/jnumed.117.199935.
