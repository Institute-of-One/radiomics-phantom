"""Build a feature-stability atlas and normalise the least stable feature.

The headline experiment, end to end and reproducible from this file alone:

1. Generate a range of phantom textures (the *targets*).
2. Observe each under several acquisition settings (the *conditions*).
3. Extract every IBSI feature and rate its reproducibility with ICC and CCC.
4. Take the intensity variance -- which noise inflates -- calibrate its physical
   response, and show that normalisation restores it to the noiseless value.

Usage
-----
    python examples/run_stability_atlas.py
"""

from __future__ import annotations

import numpy as np

from rphantom import (
    build_stability_atlas,
    calibrate_response,
    extract_features,
    generate_texture_phantom,
    normalise_feature,
    simulate_acquisition,
)


def main() -> None:
    print("Building phantoms (targets) ...")
    phantoms = [
        generate_texture_phantom(size=(28, 28, 28), corr_length=cl, lesion=False, seed=s)
        for cl, s in [(3.0, 0), (5.0, 1), (7.0, 2), (9.0, 3)]
    ]

    conditions = [
        {},  # reference: noiseless, unblurred
        {"noise_sigma": 15.0, "seed": 1},
        {"noise_sigma": 30.0, "seed": 1},
        {"psf_fwhm_mm": 3.0, "seed": 1},
        {"noise_sigma": 20.0, "psf_fwhm_mm": 2.0, "seed": 1},
    ]
    labels = ["reference", "noise-15", "noise-30", "blur-3", "noise+blur"]

    print(f"Sweeping {len(phantoms)} targets x {len(conditions)} conditions ...")

    def whole_volume(volume, mask, spacing, *, bin_width, include_morphology):
        roi = np.ones(volume.shape, dtype=bool)
        return extract_features(
            volume, roi, spacing, bin_width=bin_width, include_morphology=include_morphology
        )

    atlas = build_stability_atlas(
        phantoms,
        conditions,
        condition_labels=labels,
        feature_extractor=whole_volume,
        include_morphology=False,
    )

    iccs = np.array([r.icc for r in atlas.reliabilities.values()])
    print(f"\nRated {len(atlas.reliabilities)} features.")
    print(
        f"ICC(2,1): min={np.nanmin(iccs):+.2f}  median={np.nanmedian(iccs):+.2f}  "
        f"max={np.nanmax(iccs):+.2f}   ({int(np.sum(iccs > 0.9))} with ICC>0.9)"
    )

    print("\nLeast stable features (lowest ICC across conditions):")
    for r in atlas.ranked(by="icc")[:6]:
        print(f"  {r.tag:26s} ICC={r.icc:+.3f}  CCC_min={r.ccc_min:+.3f}")

    print("\nMost stable features:")
    for r in atlas.ranked(by="icc", ascending=False)[:6]:
        print(f"  {r.tag:26s} ICC={r.icc:+.3f}  CCC_min={r.ccc_min:+.3f}")

    # --- Physics-based normalisation of intensity variance ------------------
    print("\nCalibrating intensity variance against noise, then normalising ...")
    calib_phantom = generate_texture_phantom(size=(28, 28, 28), hu_sd=25.0, lesion=False, seed=0)
    roi = np.ones(calib_phantom.shape, dtype=bool)
    sigmas = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    variances = np.array(
        [
            extract_features(
                simulate_acquisition(calib_phantom, noise_sigma=float(s), seed=1).volume,
                roi,
                calib_phantom.spacing,
                include_morphology=False,
            )["stat_var"]
            for s in sigmas
        ]
    )
    curve = calibrate_response(
        sigmas, variances, tag="stat_var", descriptor_name="noise_sigma", model="power", power=2.0
    )
    print(
        f"  model: var = {curve.model.a:.1f} + {curve.model.b:.3f} * sigma^2   "
        f"(R^2={curve.r_squared:.4f})"
    )

    print(f"\n  {'sigma':>6s} {'raw var':>10s} {'normalised':>11s}")
    for sigma, raw in zip(sigmas, variances, strict=True):
        restored = normalise_feature(curve, raw, sigma, reference=0.0)
        print(f"  {sigma:6.0f} {raw:10.1f} {restored:11.1f}")
    print(f"\n  Raw variance spans {variances.min():.0f}-{variances.max():.0f} HU^2;")
    print(
        f"  after normalisation every value returns to ~{variances[0]:.0f} HU^2 "
        "(the noiseless truth)."
    )


if __name__ == "__main__":
    main()
