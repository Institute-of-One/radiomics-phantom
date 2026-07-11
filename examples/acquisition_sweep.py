"""Observe one phantom under a sweep of acquisition settings and tabulate a feature.

This is a miniature preview of the stability atlas: a single synthetic texture,
held fixed, is degraded across a grid of doses and reconstruction kernels, and a
representative IBSI feature (GLCM contrast) is printed for each.  It shows the
three modules -- phantom, acquisition, features -- working together.

Usage
-----
    python examples/acquisition_sweep.py [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from rphantom import (
    discretise,
    generate_texture_phantom,
    glcm_features,
    intensity_statistics,
    simulate_acquisition,
)

DOSES = (0.25, 1.0, 4.0)  # relative to reference; lower dose = more noise
PSF_FWHMS = (0.0, 2.0, 4.0)  # reconstruction-kernel blur, mm
NOISE_SIGMA = 30.0  # HU-like, at reference dose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Phantom/noise seed (default: 0).")
    args = parser.parse_args()

    phantom = generate_texture_phantom(
        size=(48, 48, 48), corr_length=6.0, hu_mean=40.0, hu_sd=25.0, lesion=False, seed=args.seed
    )
    roi = np.ones(phantom.shape, dtype=bool)

    print(f"Phantom: {phantom.shape} @ {phantom.spacing} mm, corr_length=6 mm, seed={args.seed}")
    print(f"Noise sigma at reference dose: {NOISE_SIGMA} HU-like\n")

    header = "  dose \\ psf |" + "".join(f"{f:>10.1f} mm" for f in PSF_FWHMS)
    print("GLCM contrast (3D, merged), fixed bin size 25:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dose in DOSES:
        cells = []
        for fwhm in PSF_FWHMS:
            acq = simulate_acquisition(
                phantom,
                psf_fwhm_mm=fwhm,
                noise_sigma=NOISE_SIGMA,
                dose=dose,
                seed=args.seed + 1,
            )
            disc = discretise(acq.volume, roi, method="fbs", bin_width=25.0)
            cells.append(glcm_features(disc, "3D_comb").contrast)
        print(f"  {dose:>9.2f} |" + "".join(f"{c:>13.3f}" for c in cells))

    # The noise-free, blur-free reference for comparison.
    ref = simulate_acquisition(phantom, seed=args.seed + 1)
    ref_contrast = glcm_features(
        discretise(ref.volume, roi, method="fbs", bin_width=25.0), "3D_comb"
    ).contrast
    ref_var = intensity_statistics(ref.volume, roi).variance
    print(f"\nReference (no noise, no blur): contrast={ref_contrast:.3f}, variance={ref_var:.1f}")
    print(
        "Reading: lower dose (more noise) inflates contrast; a broader kernel (more blur)\n"
        "suppresses it. The stability atlas will quantify this per feature, as ICC and CCC."
    )


if __name__ == "__main__":
    main()
