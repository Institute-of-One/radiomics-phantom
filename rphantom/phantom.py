"""Synthetic digital texture phantoms for radiomics research.

This module generates fully synthetic, deterministic 3D volumes whose texture
statistics are known by construction.  The texture is a stationary Gaussian
random field (GRF) with a Gaussian autocovariance whose correlation length is
prescribed per axis, produced by filtering white noise in the Fourier domain.

Conventions
-----------
* Arrays are indexed ``(z, y, x)`` and ``spacing`` is given in millimetres in
  the same ``(z, y, x)`` order.
* The **correlation length** ``L`` of an axis is defined as the lag, in
  millimetres, at which the normalised autocovariance of the field falls to
  ``1/e``.  A white-noise field convolved with a Gaussian kernel of standard
  deviation ``sigma`` has autocovariance ``exp(-r**2 / (4 * sigma**2))``, hence
  ``sigma = L / 2`` is used internally.
* Intensities are expressed on a Hounsfield-unit-like scale, but no CT physics
  is modelled here; see :mod:`rphantom.acquisition` for acquisition effects.

No patient data, no DICOM, and no external image inputs are involved: every
voxel is produced from ``numpy.random.default_rng(seed)``, so a given ``seed``
reproduces a bit-identical volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "Phantom",
    "generate_texture_phantom",
    "gaussian_random_field",
    "measure_correlation_length",
]

_INV_E = float(np.exp(-1.0))


@dataclass(frozen=True)
class Phantom:
    """A synthetic phantom volume together with its generative ground truth.

    Attributes
    ----------
    volume:
        ``float32`` array of shape ``(nz, ny, nx)`` on a HU-like scale.
    mask:
        Boolean array of the same shape marking the embedded lesion.  It is
        all-``False`` when the phantom was generated without a lesion.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    ground_truth:
        The generative parameters, echoed verbatim, including the realised
        lesion geometry.  Keys are stable and intended for downstream
        stability/normalisation experiments.
    seed:
        The seed passed to :func:`generate_texture_phantom`.

    """

    volume: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    ground_truth: dict[str, Any]
    seed: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """Volume shape ``(nz, ny, nx)``."""
        nz, ny, nx = self.volume.shape
        return (nz, ny, nx)

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of a single voxel in cubic millimetres."""
        return float(np.prod(self.spacing))


def _validate_triplet(
    value: Any, name: str, *, positive: bool = True
) -> tuple[float, float, float]:
    """Coerce ``value`` to a 3-tuple of floats, raising on malformed input."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 elements (z, y, x); got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; got {tuple(arr)}.")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive; got {tuple(arr)}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_size(size: Any) -> tuple[int, int, int]:
    arr = np.asarray(size)
    if arr.shape != (3,):
        raise ValueError(f"size must have exactly 3 elements (nz, ny, nx); got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"size must contain integers; got dtype {arr.dtype}.")
    if np.any(arr < 4):
        raise ValueError(f"size must be at least 4 along every axis; got {tuple(arr)}.")
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def gaussian_random_field(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    corr_lengths_mm: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a zero-mean, unit-variance Gaussian random field.

    White noise is convolved (circularly, via the FFT) with an anisotropic
    Gaussian kernel of per-axis standard deviation ``corr_lengths_mm / 2``, so
    that the normalised autocovariance of the result falls to ``1/e`` at a lag
    of ``corr_lengths_mm`` along each axis.

    Parameters
    ----------
    shape:
        Field shape ``(nz, ny, nx)``.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    corr_lengths_mm:
        Target ``1/e`` correlation length per axis, in millimetres.
    rng:
        Random generator; consumed in place so that callers can chain draws.

    Returns
    -------
    numpy.ndarray
        ``float64`` field of shape ``shape``, standardised to exactly zero
        empirical mean and unit empirical standard deviation.

    Raises
    ------
    ValueError
        If a requested correlation length exceeds half the field extent along
        its axis.  The FFT convolution is circular, so such a field wraps onto
        itself and its autocovariance is not the one that was requested.
    RuntimeError
        If the filtered field is constant (degenerate) or non-finite, which
        would otherwise propagate silently as NaN into downstream features.

    """
    corr_lengths_mm = _validate_triplet(corr_lengths_mm, "corr_lengths_mm")
    for axis, (length, n, s) in enumerate(zip(corr_lengths_mm, shape, spacing, strict=True)):
        half_extent = 0.5 * n * s
        if length > half_extent:
            raise ValueError(
                f"correlation length {length} mm along axis {axis} exceeds half the field "
                f"extent ({half_extent} mm, from {n} voxels of {s} mm). The field is generated "
                "by circular convolution, so it would wrap onto itself; enlarge the volume or "
                "shorten the correlation length."
            )

    noise = rng.standard_normal(size=shape)

    # Per-axis frequency grids in cycles/mm.  The last axis uses rfftfreq
    # because the transform is real.
    freqs = [
        np.fft.fftfreq(shape[0], d=spacing[0]),
        np.fft.fftfreq(shape[1], d=spacing[1]),
        np.fft.rfftfreq(shape[2], d=spacing[2]),
    ]
    sigmas_mm = np.asarray(corr_lengths_mm, dtype=float) / 2.0

    # Transfer function of a Gaussian kernel: exp(-2 pi^2 sigma^2 f^2).
    exponent = np.zeros(
        (shape[0], shape[1], shape[2] // 2 + 1),
        dtype=np.float64,
    )
    for axis, (f, sigma) in enumerate(zip(freqs, sigmas_mm, strict=True)):
        shaped = f.reshape([-1 if a == axis else 1 for a in range(3)])
        exponent = exponent - 2.0 * np.pi**2 * sigma**2 * shaped**2
    transfer = np.exp(exponent)

    axes = (0, 1, 2)
    field = np.fft.irfftn(np.fft.rfftn(noise, axes=axes) * transfer, s=shape, axes=axes)

    if not np.all(np.isfinite(field)):
        raise RuntimeError(
            "Gaussian random field contains non-finite values; "
            f"check spacing={spacing} and corr_lengths_mm={corr_lengths_mm}."
        )
    sd = float(field.std())
    if sd <= 0.0:
        raise RuntimeError(
            "Gaussian random field is constant (zero variance); the requested "
            f"corr_lengths_mm={corr_lengths_mm} may be too large for shape={shape}."
        )
    return (field - field.mean()) / sd


def measure_correlation_length(
    volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis: int = 0,
) -> float:
    """Estimate the ``1/e`` correlation length of ``volume`` along ``axis``.

    The 1D autocovariance along ``axis`` is obtained from the power spectrum
    (Wiener--Khinchin), averaged over the two remaining axes, and the lag at
    which it first crosses ``1/e`` is linearly interpolated.

    This is the empirical counterpart of the ``corr_length`` parameter of
    :func:`generate_texture_phantom` and is intended for verification.  It
    assumes an approximately stationary field, so it should be applied to
    lesion-free volumes (or to a homogeneous sub-region).

    Parameters
    ----------
    volume:
        Real 3D array.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    axis:
        Axis along which to measure, ``0``, ``1`` or ``2``.

    Returns
    -------
    float
        Correlation length in millimetres.

    Raises
    ------
    ValueError
        If ``volume`` is not a finite 3D array, or if the autocovariance never
        drops to ``1/e`` within half the extent of ``axis`` (the field is too
        correlated to be measured at this size).

    """
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D; got ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("volume must be finite.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2; got {axis}.")
    spacing = _validate_triplet(spacing, "spacing")

    centred = arr - arr.mean()
    power = np.abs(np.fft.fft(centred, axis=axis)) ** 2
    other = tuple(a for a in range(3) if a != axis)
    acov = np.fft.ifft(power.mean(axis=other)).real
    if acov[0] <= 0.0:
        raise ValueError("volume has zero variance; correlation length is undefined.")
    acf = acov / acov[0]

    n_half = arr.shape[axis] // 2
    below = np.flatnonzero(acf[: n_half + 1] <= _INV_E)
    if below.size == 0:
        raise ValueError(
            "autocovariance does not fall to 1/e within half the volume extent "
            f"along axis {axis}; the correlation length exceeds the measurable range."
        )
    i = int(below[0])
    # i >= 1 because acf[0] == 1 > 1/e.
    lo, hi = acf[i - 1], acf[i]
    frac = (lo - _INV_E) / (lo - hi)
    return float((i - 1 + frac) * spacing[axis])


def _ellipsoid_weight(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    center_vox: tuple[float, float, float],
    radii_mm: tuple[float, float, float],
    edge_blur_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hard_mask, blend_weight)`` for an axis-aligned ellipsoid.

    ``hard_mask`` is the exact geometric membership test (the ground-truth
    segmentation); ``blend_weight`` is the same mask optionally smoothed by a
    Gaussian of ``edge_blur_mm`` to emulate partial-volume mixing at the
    boundary.  With ``edge_blur_mm == 0`` the two coincide.
    """
    coords = np.meshgrid(
        *(
            (np.arange(n, dtype=np.float64) - c) * s
            for n, c, s in zip(shape, center_vox, spacing, strict=True)
        ),
        indexing="ij",
    )
    q2 = np.zeros(shape, dtype=np.float64)
    for d, r in zip(coords, radii_mm, strict=True):
        q2 = q2 + (d / r) ** 2
    hard = q2 <= 1.0

    if edge_blur_mm <= 0.0:
        return hard, hard.astype(np.float64)

    sigma_vox = [edge_blur_mm / s for s in spacing]
    weight = gaussian_filter(hard.astype(np.float64), sigma=sigma_vox, mode="nearest")
    return hard, np.clip(weight, 0.0, 1.0)


def generate_texture_phantom(
    size: tuple[int, int, int] = (64, 64, 64),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    corr_length: float = 6.0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_mean: float = 40.0,
    hu_sd: float = 25.0,
    lesion: bool = True,
    lesion_center: tuple[float, float, float] | None = None,
    lesion_radii_mm: tuple[float, float, float] = (10.0, 10.0, 10.0),
    lesion_hu_offset: float = 60.0,
    lesion_hu_sd: float = 15.0,
    lesion_corr_length: float = 3.0,
    lesion_edge_blur_mm: float = 0.0,
    seed: int = 0,
) -> Phantom:
    """Generate a deterministic synthetic texture phantom.

    A stationary Gaussian random field with a prescribed per-axis correlation
    length forms the background parenchyma.  Optionally, an axis-aligned
    ellipsoidal "lesion" filled with a second, independent field of different
    mean, contrast and correlation length is embedded at a known location.

    Parameters
    ----------
    size:
        Volume shape ``(nz, ny, nx)`` in voxels; at least 4 along every axis.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.
    corr_length:
        Base ``1/e`` correlation length of the background texture, in
        millimetres, before anisotropic scaling.
    anisotropy:
        Per-axis multipliers applied to ``corr_length``; the realised
        correlation length along axis ``i`` is ``corr_length * anisotropy[i]``.
    hu_mean, hu_sd:
        Mean and standard deviation of the background intensities.  ``hu_sd``
        may be ``0`` for a homogeneous background.
    lesion:
        Whether to embed a lesion.  When ``False``, ``mask`` is all-``False``.
    lesion_center:
        Lesion centre as a ``(z, y, x)`` voxel index (fractional allowed).
        Defaults to the volume centre.
    lesion_radii_mm:
        Semi-axes of the ellipsoid in millimetres, ``(rz, ry, rx)``.
    lesion_hu_offset:
        Mean intensity of the lesion relative to ``hu_mean``.
    lesion_hu_sd:
        Standard deviation of the lesion texture.
    lesion_corr_length:
        Isotropic ``1/e`` correlation length of the lesion texture, in
        millimetres.
    lesion_edge_blur_mm:
        Gaussian smoothing applied to the lesion blending weight to emulate
        partial-volume mixing.  ``mask`` always remains the exact geometric
        ellipsoid.  Defaults to ``0`` (sharp boundary).
    seed:
        Seed for ``numpy.random.default_rng``.  The same seed and parameters
        always yield a bit-identical volume.

    Returns
    -------
    Phantom
        Volume, lesion mask, spacing, ground truth and seed.

    Raises
    ------
    ValueError
        On malformed geometry, non-positive spacing/correlation lengths,
        negative standard deviations, a lesion centre outside the volume, or a
        correlation length exceeding half the volume extent along any axis.
    RuntimeError
        If the generated field is degenerate or non-finite.

    Examples
    --------
    >>> ph = generate_texture_phantom(size=(32, 32, 32), seed=7)
    >>> ph.volume.shape, ph.volume.dtype
    ((32, 32, 32), dtype('float32'))
    >>> bool(ph.mask.any())
    True

    """
    size = _validate_size(size)
    spacing = _validate_triplet(spacing, "spacing")
    anisotropy = _validate_triplet(anisotropy, "anisotropy")

    if not np.isfinite(corr_length) or corr_length <= 0.0:
        raise ValueError(f"corr_length must be finite and strictly positive; got {corr_length}.")
    if not np.isfinite(hu_mean):
        raise ValueError(f"hu_mean must be finite; got {hu_mean}.")
    if not np.isfinite(hu_sd) or hu_sd < 0.0:
        raise ValueError(f"hu_sd must be finite and non-negative; got {hu_sd}.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {type(seed).__name__}.")

    rng = np.random.default_rng(seed)
    corr_lengths_mm: tuple[float, float, float] = (
        corr_length * anisotropy[0],
        corr_length * anisotropy[1],
        corr_length * anisotropy[2],
    )

    background = hu_mean + hu_sd * gaussian_random_field(size, spacing, corr_lengths_mm, rng)

    mask = np.zeros(size, dtype=bool)
    volume = background
    lesion_gt: dict[str, Any] | None = None

    if lesion:
        radii = _validate_triplet(lesion_radii_mm, "lesion_radii_mm")
        if not np.isfinite(lesion_corr_length) or lesion_corr_length <= 0.0:
            raise ValueError(
                f"lesion_corr_length must be finite and > 0; got {lesion_corr_length}."
            )
        if not np.isfinite(lesion_hu_offset):
            raise ValueError(f"lesion_hu_offset must be finite; got {lesion_hu_offset}.")
        if not np.isfinite(lesion_hu_sd) or lesion_hu_sd < 0.0:
            raise ValueError(f"lesion_hu_sd must be finite and non-negative; got {lesion_hu_sd}.")
        if not np.isfinite(lesion_edge_blur_mm) or lesion_edge_blur_mm < 0.0:
            raise ValueError(
                f"lesion_edge_blur_mm must be finite and non-negative; got {lesion_edge_blur_mm}."
            )

        if lesion_center is None:
            center = ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, (size[2] - 1) / 2.0)
        else:
            center = _validate_triplet(lesion_center, "lesion_center", positive=False)
            if any(not (0.0 <= c <= n - 1) for c, n in zip(center, size, strict=True)):
                raise ValueError(f"lesion_center {center} lies outside the volume of shape {size}.")

        mask, weight = _ellipsoid_weight(size, spacing, center, radii, lesion_edge_blur_mm)
        if not mask.any():
            raise ValueError(
                f"lesion_radii_mm={radii} with spacing={spacing} produced an empty mask; "
                "the ellipsoid is smaller than one voxel."
            )

        lesion_field = (
            hu_mean
            + lesion_hu_offset
            + lesion_hu_sd * gaussian_random_field(size, spacing, (lesion_corr_length,) * 3, rng)
        )
        volume = background * (1.0 - weight) + lesion_field * weight

        lesion_gt = {
            "center_vox": tuple(float(c) for c in center),
            "center_mm": tuple(float(c * s) for c, s in zip(center, spacing, strict=True)),
            "radii_mm": radii,
            "hu_offset": float(lesion_hu_offset),
            "hu_sd": float(lesion_hu_sd),
            "corr_length": float(lesion_corr_length),
            "edge_blur_mm": float(lesion_edge_blur_mm),
            "n_voxels": int(mask.sum()),
            "volume_mm3": float(mask.sum() * np.prod(spacing)),
        }

    if not np.all(np.isfinite(volume)):
        raise RuntimeError("generated volume contains non-finite values; refusing to return it.")

    ground_truth: dict[str, Any] = {
        "size": size,
        "spacing": spacing,
        "corr_length": float(corr_length),
        "anisotropy": anisotropy,
        "corr_lengths_mm": tuple(float(c) for c in corr_lengths_mm),
        "hu_mean": float(hu_mean),
        "hu_sd": float(hu_sd),
        "lesion": bool(lesion),
        "lesion_params": lesion_gt,
        "seed": int(seed),
    }

    return Phantom(
        volume=np.ascontiguousarray(volume, dtype=np.float32),
        mask=mask,
        spacing=spacing,
        ground_truth=ground_truth,
        seed=int(seed),
    )
