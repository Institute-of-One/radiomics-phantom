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
    # Constant (zero-variance) features are forced to ICC=1.0 by the estimator;
    # that is a definitional artefact, not measured agreement. Separate them.
    constant = {t for t, m in atlas.matrices.items() if float(np.var(m)) == 0.0}
    n_robust = int(np.sum(iccs > 0.9))

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.hist(iccs, bins=np.linspace(-0.4, 1.0, 29), color="#4C78A8", edgecolor="white")
    med = float(np.median(iccs))
    ax.axvline(med, color="#d62728", lw=1.4, ls="--", label=f"median = {med:.2f}")
    ax.axvline(0.9, color="#2ca02c", lw=1.2, ls=":", label="ICC = 0.9 (robust)")
    ax.text(
        0.03,
        0.96,
        f"{len(iccs)} features\nICC > 0.9: {n_robust} ({len(constant)} constant)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.set_xlabel("ICC(2,1) across acquisition conditions")
    ax.set_ylabel("Number of features")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_atlas.png", bbox_inches="tight")
    plt.close(fig)
    non_constant = [r for r in atlas.reliabilities.values() if r.tag not in constant]
    genuine_top = sorted(non_constant, key=lambda r: r.icc, reverse=True)[:5]
    ranked_low = atlas.ranked(by="icc")[:5]
    results["atlas"] = {
        "n_features": int(len(atlas.reliabilities)),
        "n_targets": atlas.n_targets,
        "n_conditions": atlas.n_conditions,
        "icc_min": float(np.min(iccs)),
        "icc_median": med,
        "icc_max": float(np.max(iccs)),
        "n_icc_gt_0_9": int(np.sum(iccs > 0.9)),
        "n_constant_features": len(constant),
        "constant_features": sorted(constant),
        "n_genuine_icc_gt_0_9": int(sum(r.icc > 0.9 for r in non_constant)),
        "least_stable": [(r.tag, round(r.icc, 3)) for r in ranked_low],
        "most_stable_nonconstant": [(r.tag, round(r.icc, 3)) for r in genuine_top],
    }


# --------------------------------------------------------------------------
# Figure 3 — physics-based normalisation of intensity variance
# --------------------------------------------------------------------------
def _variances_at(phantom, roi, sigmas, seed) -> np.ndarray:
    return np.array(
        [
            extract_features(
                simulate_acquisition(phantom, noise_sigma=float(s), seed=seed).volume,
                roi,
                phantom.spacing,
                include_morphology=False,
            )["stat_var"]
            for s in sigmas
        ]
    )


def figure_normalisation() -> None:
    # Proof of concept with an explicit calibration / held-out split and
    # different noise realisations (seed=1 for calibration, seed=2 for the
    # held-out evaluation), so the demonstration is not fit and tested on the
    # same noise pattern.
    phantom = generate_texture_phantom(size=(28, 28, 28), hu_sd=25.0, lesion=False, seed=0)
    roi = np.ones(phantom.shape, dtype=bool)
    sig_cal = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    sig_hld = np.array([25.0, 30.0])
    var_cal = _variances_at(phantom, roi, sig_cal, seed=1)
    var_hld = _variances_at(phantom, roi, sig_hld, seed=2)

    curve = calibrate_response(
        sig_cal, var_cal, tag="stat_var", descriptor_name="noise_sigma", model="power", power=2.0
    )
    norm_cal = np.array(
        [normalise_feature(curve, v, s) for v, s in zip(var_cal, sig_cal, strict=True)]
    )
    norm_hld = np.array(
        [normalise_feature(curve, v, s) for v, s in zip(var_hld, sig_hld, strict=True)]
    )

    # A feature whose response the power-2 model cannot describe -> refused.
    mean_cal = np.array(
        [
            extract_features(
                simulate_acquisition(phantom, noise_sigma=float(s), seed=1).volume,
                roi,
                phantom.spacing,
                include_morphology=False,
            )["stat_mean"]
            for s in sig_cal
        ]
    )
    refused = calibrate_response(sig_cal, mean_cal, tag="stat_mean", model="power", power=2.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    grid = np.linspace(0, 30, 100)
    axL.plot(
        grid,
        curve.model.predict(grid),
        color="#4C78A8",
        lw=1.4,
        label=f"calibration fit\n(R$^2$={curve.r_squared:.4f})",
    )
    axL.scatter(sig_cal, var_cal, color="#d62728", s=20, zorder=3, label="calibration")
    axL.scatter(sig_hld, var_hld, color="#8c564b", s=26, marker="^", zorder=3, label="held-out")
    axL.set_xlabel("noise $\\sigma$ (HU-like)")
    axL.set_ylabel("intensity variance")
    axL.legend(frameon=False, fontsize=7)
    axL.spines[["top", "right"]].set_visible(False)

    axR.scatter(sig_cal, var_cal, color="#d62728", s=20, label="raw (calibration)")
    axR.scatter(sig_hld, var_hld, color="#8c564b", s=26, marker="^", label="raw (held-out)")
    axR.scatter(
        np.r_[sig_cal, sig_hld],
        np.r_[norm_cal, norm_hld],
        color="#2ca02c",
        s=20,
        marker="s",
        label="normalised",
    )
    axR.axhline(var_cal[0], color="#888", lw=0.8, ls=":")
    axR.set_xlabel("noise $\\sigma$ (HU-like)")
    axR.set_ylabel("intensity variance")
    axR.legend(frameon=False, fontsize=7)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_normalisation.png", bbox_inches="tight")
    plt.close(fig)

    all_norm = np.r_[norm_cal, norm_hld]
    results["normalisation"] = {
        "var0": round(float(var_cal[0]), 1),
        "b": round(float(curve.model.b), 4),
        "r_squared": round(float(curve.r_squared), 6),
        "raw_max": round(float(max(var_cal.max(), var_hld.max())), 1),
        "heldout_sigmas": [float(s) for s in sig_hld],
        "heldout_raw": [round(float(v), 1) for v in var_hld],
        "heldout_normalised": [round(float(v), 1) for v in norm_hld],
        "normalised_spread": round(float(all_norm.max() - all_norm.min()), 3),
        "refused_feature": "stat_mean",
        "refused_r_squared": round(float(refused.r_squared), 3),
        "refused_is_trustworthy": bool(refused.is_trustworthy),
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
