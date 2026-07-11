"""Headless compute core for Phantom Studio.

The GUI is a thin view over this module: everything that turns parameters into a
degraded volume and a feature comparison lives here, with no tkinter or
matplotlib import, so it can be tested without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rphantom import (
    extract_features,
    generate_texture_phantom,
    simulate_acquisition,
)

#: A curated, readable subset of IBSI features for the studio's live table.
STUDIO_FEATURES: tuple[str, ...] = (
    "stat_mean",
    "stat_var",
    "stat_skew",
    "ih_entropy",
    "ih_uniformity",
    "cm_contrast_3D_comb",
    "cm_joint_entr_3D_comb",
    "cm_corr_3D_comb",
    "rlm_sre_3D_comb",
    "rlm_rl_entr_3D_comb",
    "szm_sze_3D",
    "ngt_contrast_3D",
    "ngl_dc_entr_3D",
)


@dataclass(frozen=True)
class PhantomParams:
    """Parameters for the synthetic phantom."""

    size: int = 48
    corr_length: float = 6.0
    anisotropy_z: float = 1.0
    hu_mean: float = 40.0
    hu_sd: float = 25.0
    lesion: bool = True
    seed: int = 0

    def build(self):
        return generate_texture_phantom(
            size=(self.size, self.size, self.size),
            corr_length=self.corr_length,
            anisotropy=(self.anisotropy_z, 1.0, 1.0),
            hu_mean=self.hu_mean,
            hu_sd=self.hu_sd,
            lesion=self.lesion,
            seed=self.seed,
        )


@dataclass(frozen=True)
class AcquisitionParams:
    """Parameters for the simulated acquisition."""

    psf_fwhm_mm: float = 0.0
    slice_fwhm_mm: float = 0.0
    noise_sigma: float = 0.0
    noise_correlation_mm: float = 0.0
    dose: float = 1.0
    quantise_step: float = 0.0
    resample_mm: float = 0.0  # 0 means "keep the phantom grid"
    seed: int = 1

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            psf_fwhm_mm=self.psf_fwhm_mm,
            slice_fwhm_mm=self.slice_fwhm_mm,
            noise_sigma=self.noise_sigma,
            noise_correlation_mm=self.noise_correlation_mm,
            dose=self.dose,
            quantise_step=self.quantise_step,
            seed=self.seed,
        )
        if self.resample_mm > 0.0:
            kwargs["new_spacing"] = (self.resample_mm, self.resample_mm, self.resample_mm)
        return kwargs


@dataclass
class StudioResult:
    """Everything the view needs to draw one studio update."""

    reference_volume: np.ndarray
    degraded_volume: np.ndarray
    reference_mask: np.ndarray
    degraded_mask: np.ndarray
    reference_spacing: tuple[float, float, float]
    degraded_spacing: tuple[float, float, float]
    ground_truth: dict[str, Any] = field(default_factory=dict)
    feature_rows: list[tuple[str, float, float, float]] = field(default_factory=list)
    error: str | None = None

    @property
    def reference_slice_index(self) -> int:
        return self.reference_volume.shape[0] // 2

    @property
    def degraded_slice_index(self) -> int:
        return self.degraded_volume.shape[0] // 2

    def spec_lines(self) -> list[tuple[str, str]]:
        """A ``(label, value)`` summary of the phantom and the degraded grid."""
        gt = self.ground_truth
        if not gt:
            return []
        size = gt["size"]
        spacing = gt["spacing"]
        fov = tuple(n * s for n, s in zip(size, spacing, strict=True))
        cz, cy, cx = gt["corr_lengths_mm"]
        lines = [
            ("Volume", f"{size[0]}x{size[1]}x{size[2]} voxels (z, y, x)"),
            ("Voxel spacing", f"{spacing[0]:g} x {spacing[1]:g} x {spacing[2]:g} mm"),
            ("Field of view", f"{fov[0]:g} x {fov[1]:g} x {fov[2]:g} mm"),
            ("Correlation length", f"z={cz:g}  y={cy:g}  x={cx:g} mm"),
            (
                "Anisotropy (z:y:x)",
                f"{gt['anisotropy'][0]:g} : {gt['anisotropy'][1]:g} : {gt['anisotropy'][2]:g}",
            ),
            ("Intensity", f"mean {gt['hu_mean']:g}, SD {gt['hu_sd']:g} HU"),
        ]
        lesion = gt.get("lesion_params")
        if lesion:
            rz, ry, rx = lesion["radii_mm"]
            lines += [
                ("Lesion radii", f"{rz:g} x {ry:g} x {rx:g} mm"),
                ("Lesion volume", f"{lesion['volume_mm3']:.0f} mm3 ({lesion['n_voxels']} voxels)"),
                (
                    "Lesion contrast",
                    f"+{lesion['hu_offset']:g} HU, corr {lesion['corr_length']:g} mm",
                ),
            ]
        else:
            lines.append(("Lesion", "none"))
        lines.append(("Seed", f"{gt['seed']}"))

        deg_shape = self.degraded_volume.shape
        if deg_shape != tuple(size) or self.degraded_spacing != tuple(spacing):
            lines.append(
                (
                    "Degraded grid",
                    f"{deg_shape[0]}x{deg_shape[1]}x{deg_shape[2]} @ "
                    f"{self.degraded_spacing[0]:g}mm (resampled)",
                )
            )
        return lines


def orthogonal_slices(volume: np.ndarray, fraction: float = 0.5) -> dict[str, Any]:
    """Axial, coronal and sagittal cuts of a 3D volume.

    ``fraction`` in ``[0, 1]`` selects the axial (``z``) plane through the stack;
    the coronal and sagittal planes pass through the volume centre.  This is the
    slicing the GUI draws to make the phantom's three-dimensionality visible; it
    is a pure function so it can be tested without a display.

    Returns
    -------
    dict
        ``z, y, x`` indices and the three 2D planes ``axial`` (y, x),
        ``coronal`` (z, x) and ``sagittal`` (z, y).

    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D; got ndim={volume.ndim}.")
    fraction = float(np.clip(fraction, 0.0, 1.0))
    nz, ny, nx = volume.shape
    z = int(round(fraction * (nz - 1)))
    y, x = ny // 2, nx // 2
    return {
        "z": z,
        "y": y,
        "x": x,
        "axial": volume[z, :, :],
        "coronal": volume[:, y, :],
        "sagittal": volume[:, :, x],
    }


def _roi_for_features(mask: np.ndarray) -> np.ndarray:
    """Use the lesion mask if it is non-empty, else the whole volume."""
    return mask if mask.any() else np.ones(mask.shape, dtype=bool)


def compute_studio_result(
    phantom_params: PhantomParams,
    acquisition_params: AcquisitionParams,
    *,
    features: tuple[str, ...] = STUDIO_FEATURES,
    bin_width: float = 25.0,
) -> StudioResult:
    """Build the phantom, degrade it, and compare features to the clean reference.

    The reference is the phantom with no acquisition degradation; the degraded
    volume applies ``acquisition_params``.  Each feature row is
    ``(tag, reference_value, degraded_value, percent_change)``.

    Feature extraction that fails on a degenerate ROI is reported in
    :attr:`StudioResult.error` rather than raised, so the GUI stays responsive.
    """
    phantom = phantom_params.build()
    reference = simulate_acquisition(phantom, seed=acquisition_params.seed)
    degraded = simulate_acquisition(phantom, **acquisition_params.as_kwargs())

    result = StudioResult(
        reference_volume=reference.volume,
        degraded_volume=degraded.volume,
        reference_mask=reference.mask,
        degraded_mask=degraded.mask,
        reference_spacing=reference.spacing,
        degraded_spacing=degraded.spacing,
        ground_truth=phantom.ground_truth,
    )

    try:
        ref_roi = _roi_for_features(reference.mask)
        deg_roi = _roi_for_features(degraded.mask)
        ref_features = extract_features(
            reference.volume,
            ref_roi,
            reference.spacing,
            bin_width=bin_width,
            include_morphology=False,
        )
        deg_features = extract_features(
            degraded.volume,
            deg_roi,
            degraded.spacing,
            bin_width=bin_width,
            include_morphology=False,
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    rows: list[tuple[str, float, float, float]] = []
    for tag in features:
        if tag not in ref_features or tag not in deg_features:
            continue
        ref_value = ref_features[tag]
        deg_value = deg_features[tag]
        change = 100.0 * (deg_value - ref_value) / ref_value if ref_value != 0.0 else float("nan")
        rows.append((tag, ref_value, deg_value, change))
    result.feature_rows = rows
    return result
