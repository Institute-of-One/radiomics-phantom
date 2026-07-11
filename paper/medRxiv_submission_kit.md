# medRxiv Submission Kit — IORN-002 (radiomics-phantom)

Copy-paste-ready fields for the medRxiv submission form.
Manuscript file to upload: **`radiomics-phantom_medRxiv.pdf`** (figures embedded).

---

## Article type
**New Results** (methods / open-source software with validation).

## Subject area / category
**Radiology and Imaging** (primary). *Alternate if prompted:* Health Informatics.

> Scope note: medRxiv focuses on health-related research. This is a methods/software
> validation note on synthetic data. If screening questions its clinical relevance,
> the fallback venues are arXiv (eess.IV / physics.med-ph) or TechRxiv.

## Title
A Physics-Based Stability Atlas for IBSI Radiomics Features Using Synthetic Digital Phantoms

## Short / running title
Physics-based radiomics stability atlas on synthetic phantoms

## Authors
Shuji Yamamoto, PhD — Institute of One (supported by Lisit Co., Ltd., Japan; TexelCraft OÜ, Estonia)

- Corresponding author: Shuji Yamamoto
- Email: yamamoto@lisit.jp
- ORCID: 0000-0001-9211-1071

## Keywords
radiomics; reproducibility; IBSI; digital phantom; feature stability; image acquisition; open source

---

## Abstract (plain text)

Radiomics features are strongly sensitive to image acquisition, and separating that sensitivity from biological signal usually requires repeated patient scans that cannot be shared. We present an open, fully synthetic framework (radiomics-phantom) that maps and corrects radiomics feature instability without any patient data. Deterministic three-dimensional texture phantoms are generated as anisotropic Gaussian random fields with known ground truth and an optional embedded lesion. A from-scratch, IBSI-compliant feature core computes all eleven IBSI feature families and reproduces all 482 published IBSI digital-phantom reference values exactly. A physically motivated acquisition simulator applies reconstruction-kernel blur, slice-profile averaging, dose-scaled correlated noise, resampling, and quantisation, deterministically. Per-feature reproducibility is quantified by the intraclass correlation coefficient ICC(2,1) and Lin's concordance correlation coefficient across a grid of phantom textures and acquisition settings; across a five-by-five texture-by-acquisition sweep, reproducibility spans the full range of ICC (median 0.02, minimum -0.20, maximum 1.00), separating fragile from robust features. Finally, each feature's response to a known acquisition descriptor is calibrated and inverted to restore its reference value: intensity variance follows var0 + b*sigma^2 under added noise (fitted slope 1.00, R-squared = 1.000), and physics-based normalisation collapses the noise-induced spread (raw variance 625 to 1526) back onto the noiseless value. All code, the phantom generator, the feature core, evaluation scripts, and a 716-test suite are released openly (MIT), archived on Zenodo, and validated by continuous integration. The contribution is a reproducible, patient-data-free testbed for radiomics feature stability and physics-grounded normalisation.

*(~250 words)*

---

## Required declarations (paste into the matching form fields)

**Competing Interest Statement**
The author conducts commercial imaging-analysis services through Lisit Co., Ltd. (Japan) and TexelCraft OÜ (Estonia), which support the Institute of One research brand. This work used no client or patient data and presents an openly licensed method. The author declares no other competing interests.

**Funding Statement**
This work received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Author Contributions (CRediT)**
Shuji Yamamoto: Conceptualization, Methodology, Software, Validation, Formal analysis, Visualization, Writing – original draft and editing. AI tools were used as disclosed in the manuscript (AI-Use Disclosure).

**Data Availability Statement**
All code, the phantom generator, the IBSI feature core, evaluation scripts, and the test suite are openly available at https://github.com/Institute-of-One/radiomics-phantom under the MIT license, archived on Zenodo (concept DOI 10.5281/zenodo.21309875, all versions). No patient, clinical, or client data were used; all data are synthetic and produced by the included reproducible generator. The only external reference data are the public IBSI digital phantom and its published values.

**Ethics / IRB Statement**
Not applicable. The study involved no human participants, animal subjects, or patient data; only synthetic data were analyzed. No IRB approval was required.

**Clinical Trial Information**
Not applicable (no clinical trial).

**Prior publication / dual submission**
Not previously published and not under consideration at a peer-reviewed journal. The associated software is archived on Zenodo (a data/software archive, not a competing publication).

---

## License (choose on medRxiv)
**Recommended: CC-BY 4.0** — matches the open ethos and the note's text license (text CC BY 4.0 / code MIT). It permits reuse with attribution.

*Alternatives offered by medRxiv:* CC-BY-NC 4.0, CC-BY-ND 4.0, CC-BY-NC-ND 4.0, CC0, or "No reuse."

---

## Optional submission summary (if a free-text field asks "why post this")
This preprint documents and openly releases a reproducible, patient-data-free framework for studying radiomics feature stability: deterministic synthetic phantoms, a from-scratch IBSI feature core that reproduces all 482 published reference values, a physically motivated acquisition simulator, an ICC/CCC stability atlas, and physics-based feature normalisation. It addresses the well-documented sensitivity of radiomics features to acquisition without requiring patient data. Code and archive: github.com/Institute-of-One/radiomics-phantom · DOI 10.5281/zenodo.21309875.

---

## Pre-submission checklist
- [ ] medRxiv account ready (corresponding author = yamamoto@lisit.jp)
- [ ] Upload `radiomics-phantom_medRxiv.pdf` (figures embedded)
- [ ] Paste Title / Abstract / Keywords
- [ ] Select category: Radiology and Imaging · type: New Results
- [ ] Paste Competing Interest / Funding / Contributions / Data Availability / Ethics
- [ ] Choose license: CC-BY 4.0
- [ ] Confirm no patient data / no IRB needed
- [ ] Submit → await medRxiv screening (health-scope check)
