"""Generate the preprint figures and numbers from rphantom, deterministically.

Every value quoted in the manuscript is produced here and written to
``paper/figures/results.json`` so text and figures cannot diverge.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rphantom import (  # noqa: E402
    build_stability_atlas,
    calibrate_response,
    extract_features,
    generate_texture_phantom,
    normalise_feature,
    simulate_acquisition,
)
from rphantom.features import discretise, glcm_features  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 200})

results: dict = {}


# --------------------------------------------------------------------------
# Figure 1 — the synthetic phantom in three orthogonal planes, clean vs degraded
# --------------------------------------------------------------------------
def figure_phantom() -> None:
    phantom = generate_texture_phantom(
        size=(48, 48, 48),
        corr_length=6.0,
        anisotropy=(2.0, 1.0, 1.0),
        hu_mean=40.0,
        hu_sd=25.0,
        lesion=True,
        lesion_radii_mm=(11.0, 11.0, 11.0),
        seed=0,
    )
    acq = simulate_acquisition(phantom, psf_fwhm_mm=2.0, noise_sigma=25.0, dose=1.0, seed=1)

    def planes(vol):
        nz, ny, nx = vol.shape
        return {
            "Axial (y,x)": vol[nz // 2],
            "Coronal (z,x)": vol[:, ny // 2, :],
            "Sagittal (z,y)": vol[:, :, nx // 2],
        }

    vmin, vmax = float(phantom.volume.min()), float(phantom.volume.max())
    mplanes = planes(phantom.mask.astype(float))
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.3))
    for row, (label, vol) in enumerate([("Reference", phantom.volume), ("Degraded", acq.volume)]):
        for col, (title, img) in enumerate(planes(vol).items()):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
            m = list(mplanes.values())[col]
            if m.any() and not m.all():
                ax.contour(m, levels=[0.5], colors="#d62728", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig1_phantom.png", bbox_inches="tight")
    plt.close(fig)
    results["fig1"] = {"psf_fwhm_mm": 2.0, "noise_sigma": 25.0}


# --------------------------------------------------------------------------
# Figure 2 — feature-stability atlas: ICC distribution across features
# --------------------------------------------------------------------------
def figure_atlas() -> None:
    phantoms = [
        generate_texture_phantom(size=(24, 24, 24), corr_length=cl, lesion=False, seed=s)
        for cl, s in [(3.0, 0), (4.5, 1), (6.0, 2), (7.5, 3), (9.0, 4)]
    ]
    conditions = [
        {},
        {"noise_sigma": 15.0, "seed": 1},
        {"noise_sigma": 30.0, "seed": 1},
        {"psf_fwhm_mm": 3.0, "seed": 1},
        {"noise_sigma": 20.0, "psf_fwhm_mm": 2.0, "seed": 1},
    ]

    def whole(volume, mask, spacing, *, bin_width, include_morphology):
        roi = np.ones(volume.shape, dtype=bool)
        return extract_features(
            volume, roi, spacing, bin_width=bin_width, include_morphology=include_morphology
        )

    atlas = build_stability_atlas(
        phantoms,
        conditions,
        feature_extractor=whole,
        include_morphology=False,
    )
    iccs = np.array([r.icc for r in atlas.reliabilities.values()], dtype=float)
    iccs = iccs[np.isfinite(iccs)]

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.hist(iccs, bins=np.linspace(-0.4, 1.0, 29), color="#4C78A8", edgecolor="white")
    med = float(np.median(iccs))
    ax.axvline(med, color="#d62728", lw=1.4, ls="--", label=f"median = {med:.2f}")
    ax.axvline(0.9, color="#2ca02c", lw=1.2, ls=":", label="ICC = 0.9 (robust)")
    ax.set_xlabel("ICC(2,1) across acquisition conditions")
    ax.set_ylabel("Number of features")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_atlas.png", bbox_inches="tight")
    plt.close(fig)

    ranked_low = atlas.ranked(by="icc")[:5]
    ranked_high = atlas.ranked(by="icc", ascending=False)[:5]
    results["atlas"] = {
        "n_features": int(len(atlas.reliabilities)),
        "n_targets": atlas.n_targets,
        "n_conditions": atlas.n_conditions,
        "icc_min": float(np.min(iccs)),
        "icc_median": med,
        "icc_max": float(np.max(iccs)),
        "n_icc_gt_0_9": int(np.sum(iccs > 0.9)),
        "least_stable": [(r.tag, round(r.icc, 3)) for r in ranked_low],
        "most_stable": [(r.tag, round(r.icc, 3)) for r in ranked_high],
    }


# --------------------------------------------------------------------------
# Figure 3 — physics-based normalisation of intensity variance
# --------------------------------------------------------------------------
def figure_normalisation() -> None:
    phantom = generate_texture_phantom(size=(28, 28, 28), hu_sd=25.0, lesion=False, seed=0)
    roi = np.ones(phantom.shape, dtype=bool)
    sigmas = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    variances = np.array(
        [
            extract_features(
                simulate_acquisition(phantom, noise_sigma=float(s), seed=1).volume,
                roi,
                phantom.spacing,
                include_morphology=False,
            )["stat_var"]
            for s in sigmas
        ]
    )
    curve = calibrate_response(
        sigmas, variances, tag="stat_var", descriptor_name="noise_sigma", model="power", power=2.0
    )
    normalised = np.array(
        [
            normalise_feature(curve, v, s, reference=0.0)
            for v, s in zip(sigmas * 0 + variances, sigmas, strict=True)
        ]
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    grid = np.linspace(0, 30, 100)
    axL.plot(
        grid,
        curve.model.predict(grid),
        color="#4C78A8",
        lw=1.4,
        label=f"fit: var$_0$+b·$\\sigma^2$\n(R$^2$={curve.r_squared:.4f})",
    )
    axL.scatter(sigmas, variances, color="#d62728", s=18, zorder=3, label="measured")
    axL.set_xlabel("noise $\\sigma$ (HU-like)")
    axL.set_ylabel("intensity variance")
    axL.legend(frameon=False, fontsize=7.5)
    axL.spines[["top", "right"]].set_visible(False)

    axR.scatter(sigmas, variances, color="#d62728", s=18, label="raw")
    axR.scatter(sigmas, normalised, color="#2ca02c", s=18, marker="s", label="normalised")
    axR.axhline(variances[0], color="#888", lw=0.8, ls=":")
    axR.set_xlabel("noise $\\sigma$ (HU-like)")
    axR.set_ylabel("intensity variance")
    axR.legend(frameon=False, fontsize=7.5)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_normalisation.png", bbox_inches="tight")
    plt.close(fig)

    results["normalisation"] = {
        "var0": round(float(variances[0]), 1),
        "b": round(float(curve.model.b), 4),
        "r_squared": round(float(curve.r_squared), 6),
        "raw_max": round(float(variances.max()), 1),
        "normalised_spread": round(float(normalised.max() - normalised.min()), 3),
        "normalised_mean": round(float(normalised.mean()), 1),
    }


# --------------------------------------------------------------------------
# Extra numbers: acquisition sensitivity of one feature (for the text)
# --------------------------------------------------------------------------
def acquisition_numbers() -> None:
    phantom = generate_texture_phantom(size=(48, 48, 48), lesion=False, seed=0)
    roi = np.ones(phantom.shape, dtype=bool)

    def contrast(**kw):
        acq = simulate_acquisition(phantom, seed=1, **kw)
        d = discretise(acq.volume, roi, method="fbs", bin_width=25.0)
        return float(glcm_features(d, "3D_comb").contrast)

    results["acquisition"] = {
        "contrast_noise": {s: round(contrast(noise_sigma=float(s)), 3) for s in (0, 10, 25, 50)},
        "contrast_blur": {f: round(contrast(psf_fwhm_mm=float(f)), 3) for f in (0, 2, 4, 6)},
    }


if __name__ == "__main__":
    print("Figure 1: phantom ...")
    figure_phantom()
    print("Figure 2: stability atlas ...")
    figure_atlas()
    print("Figure 3: normalisation ...")
    figure_normalisation()
    print("Acquisition sensitivity numbers ...")
    acquisition_numbers()
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved figures and results.json to", OUT)
    print(json.dumps(results, indent=2))
