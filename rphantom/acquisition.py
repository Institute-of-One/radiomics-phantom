"""Simulated acquisition degradation.

The stability atlas needs the *same* underlying texture observed under many
"scanners".  This module supplies that: physically motivated, parameterised,
deterministic degradations of a :class:`rphantom.phantom.Phantom`.

Modelled effects
----------------
======================  ==========================================  =================
Effect                  Physical origin                             Function
======================  ==========================================  =================
Blur                    finite focal spot + detector + recon kernel :func:`apply_blur`
                        (the modulation transfer function)
Noise                   quantum/electronic noise at a dose level    :func:`add_noise`
Resampling              interpolation onto a different voxel grid   :func:`resample`
Quantisation            HU rounding / reconstruction matrix         :func:`quantise`
======================  ==========================================  =================

The through-plane slice-sensitivity profile is the ``z`` component of
:func:`apply_blur`, and is exposed separately as :func:`apply_slice_profile`.
:func:`simulate_acquisition` composes the whole chain and records what it did.

Determinism
-----------
Only noise is stochastic, and it draws from ``numpy.random.default_rng(seed)``;
every other effect is a deterministic function of its parameters.  The same
phantom, parameters and seed therefore yield a bit-identical acquisition.  As
elsewhere in ``rphantom``, malformed input raises rather than failing silently.

Physical grounding
------------------
* The point spread function is modelled as an anisotropic Gaussian, so a
  requested full width at half maximum (FWHM) maps to a standard deviation of
  ``FWHM / (2 * sqrt(2 * ln 2))``.  Real CT PSFs are not exactly Gaussian, but
  the Gaussian captures the dominant resolution loss with one interpretable
  number per axis.
* CT noise variance is inversely proportional to dose, so relative to a
  reference dose the noise standard deviation scales as ``1 / sqrt(dose)``.
* The reconstruction kernel correlates noise over a short range; that texture is
  reproduced by reusing the Gaussian-random-field machinery of
  :mod:`rphantom.phantom` with a small correlation length.

Runtime dependencies remain ``numpy`` / ``scipy`` / ``scikit-image`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from rphantom.phantom import Phantom, gaussian_random_field

__all__ = [
    "Acquisition",
    "AcquisitionError",
    "add_noise",
    "apply_blur",
    "apply_slice_profile",
    "quantise",
    "resample",
    "simulate_acquisition",
]

#: FWHM = this factor times the Gaussian standard deviation.
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class AcquisitionError(ValueError):
    """Raised when an acquisition cannot be simulated from the given input.

    Subclasses :class:`ValueError`.  Raised for malformed parameters or a
    degenerate result, never returned as a silent ``nan``.
    """


@dataclass(frozen=True)
class Acquisition:
    """A phantom observed under one simulated acquisition setting.

    Attributes
    ----------
    volume:
        ``float32`` degraded volume, shape ``(nz, ny, nx)``.
    mask:
        Boolean ROI mask on the same grid as ``volume``.  It differs from the
        source phantom's mask only when the acquisition resampled the grid.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``, after any resampling.
    settings:
        The acquisition parameters actually applied, for the stability atlas to
        key features against.
    ground_truth:
        The source phantom's ground truth, echoed verbatim.
    seed:
        The noise seed.

    """

    volume: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    settings: dict[str, Any]
    ground_truth: dict[str, Any]
    seed: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """Volume shape ``(nz, ny, nx)``."""
        nz, ny, nx = self.volume.shape
        return (nz, ny, nx)


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3:
        raise AcquisitionError(f"volume must be 3D (z, y, x); got ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise AcquisitionError("volume must be finite.")
    return arr


def _validate_spacing(spacing: Any) -> tuple[float, float, float]:
    arr = np.asarray(spacing, dtype=float)
    if arr.shape != (3,):
        raise AcquisitionError(
            f"spacing must have exactly 3 elements (dz, dy, dx); got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise AcquisitionError(f"spacing must be finite and strictly positive; got {tuple(arr)}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _fwhm_triplet(fwhm_mm: Any, name: str) -> tuple[float, float, float]:
    arr = np.asarray(fwhm_mm, dtype=float)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise AcquisitionError(f"{name} must be a scalar or 3 values (z, y, x); got {fwhm_mm!r}.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise AcquisitionError(f"{name} must be finite and non-negative; got {tuple(arr)}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def apply_blur(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    fwhm_mm: float | tuple[float, float, float],
) -> np.ndarray:
    """Blur ``volume`` with an anisotropic Gaussian point spread function.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    fwhm_mm:
        Full width at half maximum of the PSF, in millimetres.  A scalar applies
        the same blur to all axes; a triple ``(fz, fy, fx)`` blurs each axis
        independently.  An axis with ``0`` is left sharp.

    Returns
    -------
    numpy.ndarray
        ``float64`` blurred volume of the same shape.  The intensity mean is
        preserved up to edge effects (``mode="nearest"``).

    """
    arr = _validate_volume(volume)
    spacing = _validate_spacing(spacing)
    fwhm = _fwhm_triplet(fwhm_mm, "fwhm_mm")

    sigma_vox = [(f / _FWHM_PER_SIGMA) / s for f, s in zip(fwhm, spacing, strict=True)]
    if not any(sigma_vox):
        return arr
    return gaussian_filter(arr, sigma=sigma_vox, mode="nearest")


def apply_slice_profile(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    fwhm_mm: float,
) -> np.ndarray:
    """Average through-plane to emulate a finite slice-sensitivity profile.

    This is the ``z``-only special case of :func:`apply_blur`: a slice thickness
    broader than the reconstruction interval blurs along ``z`` without touching
    in-plane resolution.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    fwhm_mm:
        Effective slice-profile FWHM in millimetres.

    Returns
    -------
    numpy.ndarray
        ``float64`` volume of the same shape.

    """
    fwhm = float(np.asarray(fwhm_mm, dtype=float))
    return apply_blur(volume, spacing, (fwhm, 0.0, 0.0))


def add_noise(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    sigma: float,
    rng: np.random.Generator,
    *,
    correlation_length_mm: float = 0.0,
) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation ``sigma``.

    With ``correlation_length_mm == 0`` the noise is white; with a positive
    correlation length it is coloured by reusing the Gaussian-random-field
    machinery of :mod:`rphantom.phantom`, emulating the way a reconstruction
    kernel correlates noise over a short range.  Either way the field is
    standardised to exactly ``sigma`` before being added, so ``sigma`` is
    reproduced across the whole volume.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    sigma:
        Noise standard deviation in intensity units.  ``0`` returns the volume
        unchanged.
    rng:
        Random generator, consumed in place.
    correlation_length_mm:
        ``1/e`` correlation length of the noise texture, in millimetres.

    Returns
    -------
    numpy.ndarray
        ``float64`` noisy volume of the same shape.

    Raises
    ------
    AcquisitionError
        If ``sigma`` or ``correlation_length_mm`` is negative or non-finite.

    """
    arr = _validate_volume(volume)
    spacing = _validate_spacing(spacing)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise AcquisitionError(f"sigma must be finite and non-negative; got {sigma}.")
    if not np.isfinite(correlation_length_mm) or correlation_length_mm < 0.0:
        raise AcquisitionError(
            f"correlation_length_mm must be finite and non-negative; got {correlation_length_mm}."
        )
    if sigma == 0.0:
        return arr

    if correlation_length_mm > 0.0:
        field = gaussian_random_field(arr.shape, spacing, (correlation_length_mm,) * 3, rng)
    else:
        field = rng.standard_normal(size=arr.shape)
        field = (field - field.mean()) / field.std()

    return arr + sigma * field


def resample(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
    *,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Interpolate ``volume`` and ``mask`` onto a grid of ``new_spacing``.

    Radiomics features are strongly sensitive to voxel size, so this is a
    first-class acquisition axis.  The volume is resampled by a spline of the
    given order; the mask is resampled by nearest-neighbour and re-thresholded so
    that it stays boolean and geometrically faithful.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI mask of the same shape.
    spacing:
        Current voxel size in millimetres, ``(dz, dy, dx)``.
    new_spacing:
        Target voxel size in millimetres.
    order:
        Spline order for the volume, ``0`` to ``5``.  The mask always uses
        order ``0``.

    Returns
    -------
    tuple
        ``(new_volume, new_mask, new_spacing)``.

    Raises
    ------
    AcquisitionError
        On malformed spacing, an out-of-range order, or a target so coarse that
        the ROI would vanish.

    """
    arr = _validate_volume(volume)
    mask = np.asarray(mask)
    if mask.shape != arr.shape:
        raise AcquisitionError(f"mask shape {mask.shape} does not match volume shape {arr.shape}.")
    if mask.dtype != np.bool_:
        raise AcquisitionError(f"mask must be boolean; got dtype {mask.dtype}.")
    if not mask.any():
        raise AcquisitionError("mask is empty; there is no ROI to resample.")
    spacing = _validate_spacing(spacing)
    new_spacing = _validate_spacing(new_spacing)
    if not 0 <= order <= 5:
        raise AcquisitionError(f"order must be between 0 and 5; got {order}.")

    factors = tuple(s / t for s, t in zip(spacing, new_spacing, strict=True))
    new_volume = zoom(arr, factors, order=order, mode="nearest")
    new_mask = zoom(mask.astype(np.float64), factors, order=0, mode="grid-constant") > 0.5

    if not new_mask.any():
        raise AcquisitionError(
            f"resampling from {spacing} to {new_spacing} mm left an empty ROI; "
            "the target grid is too coarse to contain the mask."
        )
    return new_volume, new_mask, new_spacing


def quantise(volume: np.ndarray, step: float) -> np.ndarray:
    """Round intensities onto a uniform grid of width ``step``.

    Parameters
    ----------
    volume:
        3D intensity volume.
    step:
        Quantisation step in intensity units.  ``0`` returns the volume
        unchanged.

    Returns
    -------
    numpy.ndarray
        ``float64`` quantised volume of the same shape.

    Raises
    ------
    AcquisitionError
        If ``step`` is negative or non-finite.

    """
    arr = _validate_volume(volume)
    if not np.isfinite(step) or step < 0.0:
        raise AcquisitionError(f"step must be finite and non-negative; got {step}.")
    if step == 0.0:
        return arr
    return np.round(arr / step) * step


def simulate_acquisition(
    phantom: Phantom,
    *,
    psf_fwhm_mm: float = 0.0,
    slice_fwhm_mm: float = 0.0,
    new_spacing: tuple[float, float, float] | None = None,
    noise_sigma: float = 0.0,
    noise_correlation_mm: float = 0.0,
    dose: float = 1.0,
    quantise_step: float = 0.0,
    resample_order: int = 3,
    seed: int = 0,
) -> Acquisition:
    """Observe ``phantom`` under one simulated acquisition setting.

    The effects are applied in acquisition order: in-plane blur and slice
    profile (both Gaussian), then resampling to the target grid, then noise, then
    quantisation.  Noise is added after resampling because it arises on the
    reconstructed voxel grid; blur is applied before, because it band-limits the
    signal that grid samples.

    Parameters
    ----------
    phantom:
        The source :class:`~rphantom.phantom.Phantom`.
    psf_fwhm_mm:
        In-plane point-spread-function FWHM in millimetres.
    slice_fwhm_mm:
        Through-plane slice-profile FWHM in millimetres.
    new_spacing:
        Target voxel size ``(dz, dy, dx)`` in millimetres, or ``None`` to keep
        the phantom's grid.
    noise_sigma:
        Noise standard deviation in intensity units at the reference dose.
    noise_correlation_mm:
        ``1/e`` correlation length of the noise texture, in millimetres.
    dose:
        Dose relative to the reference; noise scales as ``1 / sqrt(dose)``, so a
        higher dose is quieter.  Must be strictly positive.
    quantise_step:
        Intensity quantisation step; ``0`` disables it.
    resample_order:
        Spline order for volume resampling.
    seed:
        Seed for ``numpy.random.default_rng``; only the noise depends on it.

    Returns
    -------
    Acquisition

    Raises
    ------
    AcquisitionError
        On any malformed parameter, or a non-positive dose.

    Examples
    --------
    >>> from rphantom import generate_texture_phantom
    >>> phantom = generate_texture_phantom(size=(32, 32, 32), seed=0)
    >>> acq = simulate_acquisition(phantom, psf_fwhm_mm=2.0, noise_sigma=10.0, seed=1)
    >>> acq.volume.shape, acq.volume.dtype
    ((32, 32, 32), dtype('float32'))

    """
    if not isinstance(phantom, Phantom):
        raise AcquisitionError(f"phantom must be a Phantom; got {type(phantom).__name__}.")
    if not np.isfinite(dose) or dose <= 0.0:
        raise AcquisitionError(f"dose must be finite and strictly positive; got {dose}.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise AcquisitionError(f"seed must be an int; got {type(seed).__name__}.")

    volume = phantom.volume.astype(np.float64)
    mask = phantom.mask
    spacing = phantom.spacing
    rng = np.random.default_rng(seed)

    volume = apply_blur(volume, spacing, (slice_fwhm_mm, psf_fwhm_mm, psf_fwhm_mm))

    if new_spacing is not None:
        volume, mask, spacing = resample(volume, mask, spacing, new_spacing, order=resample_order)

    effective_sigma = noise_sigma / np.sqrt(dose)
    volume = add_noise(
        volume, spacing, effective_sigma, rng, correlation_length_mm=noise_correlation_mm
    )

    volume = quantise(volume, quantise_step)

    if not np.all(np.isfinite(volume)):
        raise AcquisitionError(
            "simulated volume contains non-finite values; refusing to return it."
        )

    settings = {
        "psf_fwhm_mm": float(psf_fwhm_mm),
        "slice_fwhm_mm": float(slice_fwhm_mm),
        "new_spacing": None if new_spacing is None else _validate_spacing(new_spacing),
        "noise_sigma": float(noise_sigma),
        "noise_correlation_mm": float(noise_correlation_mm),
        "dose": float(dose),
        "effective_noise_sigma": float(effective_sigma),
        "quantise_step": float(quantise_step),
        "resample_order": int(resample_order),
    }

    return Acquisition(
        volume=np.ascontiguousarray(volume, dtype=np.float32),
        mask=mask,
        spacing=spacing,
        settings=settings,
        ground_truth=phantom.ground_truth,
        seed=int(seed),
    )
