"""IBSI-compliant radiomics feature core, implemented from first principles.

This module implements a subset of the feature families defined by the Image
Biomarker Standardisation Initiative (IBSI), directly from their mathematical
definitions, using only ``numpy``.  PyRadiomics is not used.  Every feature is
validated against the IBSI digital phantom reference values in
``tests/test_features_ibsi.py``.

Implemented families
--------------------
=========================================  ========  ==========================
Family                                     Features  Function
=========================================  ========  ==========================
Intensity-based statistics (IBSI 3.3)            18  :func:`intensity_statistics`
Intensity histogram (IBSI 3.4)                   23  :func:`intensity_histogram`
Grey level co-occurrence (IBSI 3.6)              25  :func:`glcm_features`
Grey level run length (IBSI 3.7)                 16  :func:`glrlm_features`
Grey level size zone (IBSI 3.8)                  16  :func:`glszm_features`
Grey level distance zone (IBSI 3.9)              16  :func:`gldzm_features`
Neighbourhood grey tone difference (3.10)         5  :func:`ngtdm_features`
Neighbouring grey level dependence (3.11)        17  :func:`ngldm_features`
Morphology (IBSI 3.1)                            25  :func:`morphology_features`
Local intensity (IBSI 3.2)                        2  :func:`local_intensity_features`
Intensity-volume histogram (IBSI 3.5)             6  :func:`intensity_volume_histogram`
=========================================  ========  ==========================

Texture families are computed on a *discretised* volume; see :func:`discretise`.
The directional families (GLCM, GLRLM) support all six IBSI aggregation methods
(:data:`AGGREGATIONS`); the others, having no direction, support the three of
:data:`ZONE_AGGREGATIONS`.  The choice of aggregation is itself a source of
feature variability that this project exists to quantify.

Conventions
-----------
* Arrays are indexed ``(z, y, x)``.  "2D" means within a constant-``z`` slice.
* Discretised intensities are integers ``1 .. n_levels``.  Levels that do not
  occur in the ROI are still part of the range: the IBSI digital phantom has no
  voxels at levels 2 and 5, yet its co-occurrence matrices are 6x6.
* Entropies use base-2 logarithms, with the convention ``0 * log2(0) = 0``.
* Nothing fails silently.  A degenerate ROI (empty, or a single grey level,
  which makes correlation-type features undefined) raises
  :class:`FeatureError` rather than returning ``nan``.

References
----------
Zwanenburg et al., "The Image Biomarker Standardization Initiative:
Standardized Quantitative Radiomics for High-Throughput Image-based
Phenotyping", Radiology 295(2), 2020.  https://ibsi.readthedocs.io

"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeVar

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage import label as ndimage_label

__all__ = [
    "AGGREGATIONS",
    "ZONE_AGGREGATIONS",
    "Aggregation",
    "Discretisation",
    "FeatureError",
    "GLCMFeatures",
    "GLDZMFeatures",
    "GLRLMFeatures",
    "GLSZMFeatures",
    "IntensityHistogramFeatures",
    "IntensityStatistics",
    "NGLDMFeatures",
    "NGTDMFeatures",
    "ZoneAggregation",
    "discretise",
    "glcm_features",
    "gldzm_features",
    "glrlm_features",
    "glszm_features",
    "intensity_histogram",
    "intensity_statistics",
    "ngldm_features",
    "ngtdm_features",
]


class FeatureError(ValueError):
    """Raised when a feature cannot be computed from the given ROI.

    Subclasses :class:`ValueError`.  Raised instead of returning ``nan`` when
    the ROI is empty, has zero variance, or otherwise makes a feature
    mathematically undefined.
    """


Aggregation = Literal["2D_avg", "2D_comb", "2_5D_avg", "2_5D_comb", "3D_avg", "3D_comb"]

#: The six IBSI texture-matrix aggregation methods, named by their IBSI tag suffix.
#:
#: ``2D_avg``
#:     One matrix per in-plane direction per slice; features averaged over all.
#: ``2D_comb``
#:     Directions merged within each slice; features averaged over slices.
#: ``2_5D_avg``
#:     Slices merged within each in-plane direction; features averaged over directions.
#: ``2_5D_comb``
#:     All slices and in-plane directions merged into one matrix.
#: ``3D_avg``
#:     One matrix per 3D direction; features averaged over the 13 directions.
#: ``3D_comb``
#:     All 13 3D directions merged into one matrix.
AGGREGATIONS: tuple[Aggregation, ...] = (
    "2D_avg",
    "2D_comb",
    "2_5D_avg",
    "2_5D_comb",
    "3D_avg",
    "3D_comb",
)

_LOG2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    """Base-2 Shannon entropy of ``p``, using ``0 * log2(0) = 0``."""
    nz = p[p > 0.0]
    return float(-np.sum(nz * (np.log(nz) / _LOG2)))


def _percentile(sorted_values: np.ndarray, k: float) -> float:
    """The ``k``-th percentile, by the nearest-rank (order statistic) definition.

    ``P_k`` is the smallest value at or below which at least ``k`` percent of
    the sample lies, i.e. element ``ceil(k * n / 100)`` of the sorted sample.
    IBSI's digital phantom does not discriminate between this and linear
    interpolation, but the nearest-rank definition is the one IBSI specifies.
    """
    n = sorted_values.size
    rank = int(np.ceil(k / 100.0 * n))
    return float(sorted_values[min(max(rank, 1), n) - 1])


def _check_roi(mask: np.ndarray, volume: np.ndarray) -> None:
    if volume.ndim != 3:
        raise FeatureError(f"volume must be 3D (z, y, x); got ndim={volume.ndim}.")
    if mask.shape != volume.shape:
        raise FeatureError(f"mask shape {mask.shape} does not match volume shape {volume.shape}.")
    if mask.dtype != np.bool_:
        raise FeatureError(f"mask must be boolean; got dtype {mask.dtype}.")
    if not mask.any():
        raise FeatureError("ROI mask is empty; no features can be computed.")


def _dispersion_stats(values: np.ndarray) -> dict[str, float]:
    """The 18 IBSI dispersion/moment statistics shared by 3.3 and 3.4.

    ``values`` is the 1D sample of ROI intensities (raw for 3.3, discretised
    levels for 3.4).
    """
    x = np.asarray(values, dtype=np.float64)
    mu = float(x.mean())

    central = x - mu
    m2 = float(np.mean(central**2))
    m3 = float(np.mean(central**3))
    m4 = float(np.mean(central**4))

    if m2 <= 0.0:
        raise FeatureError(
            "ROI has zero intensity variance; skewness, kurtosis and the "
            "coefficient of variation are undefined."
        )

    ordered = np.sort(x)
    p10 = _percentile(ordered, 10.0)
    p25 = _percentile(ordered, 25.0)
    p75 = _percentile(ordered, 75.0)
    p90 = _percentile(ordered, 90.0)
    median = float(np.median(x))

    robust = x[(x >= p10) & (x <= p90)]
    robust_mean = float(robust.mean())

    if p75 + p25 == 0.0:
        raise FeatureError("the quartile coefficient of dispersion is undefined: P25 + P75 == 0.")

    return {
        "mean": mu,
        "variance": m2,
        "skewness": m3 / m2**1.5,
        "excess_kurtosis": m4 / m2**2 - 3.0,
        "median": median,
        "minimum": float(x.min()),
        "percentile_10": p10,
        "percentile_90": p90,
        "maximum": float(x.max()),
        "interquartile_range": p75 - p25,
        "intensity_range": float(x.max() - x.min()),
        "mean_absolute_deviation": float(np.abs(central).mean()),
        "robust_mean_absolute_deviation": float(np.abs(robust - robust_mean).mean()),
        "median_absolute_deviation": float(np.abs(x - median).mean()),
        "coefficient_of_variation": np.sqrt(m2) / mu,
        "quartile_coefficient_of_dispersion": (p75 - p25) / (p75 + p25),
        "energy": float(np.sum(x**2)),
        "root_mean_square": float(np.sqrt(np.mean(x**2))),
    }


# ---------------------------------------------------------------------------
# Discretisation (IBSI 2.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Discretisation:
    """A discretised ROI, ready for texture analysis.

    Attributes
    ----------
    levels:
        ``int32`` array shaped like the source volume.  Voxels inside the ROI
        hold an integer in ``1 .. n_levels``; voxels outside hold ``0``.
    mask:
        The ROI intensity mask.
    n_levels:
        ``Ng``, the number of grey levels spanned, ``floor((upper - lower) /
        bin_width) + 1`` for ``"fbs"`` and ``bin_number`` for ``"fbn"``.  Levels
        absent from the ROI still count: they occupy empty rows and columns of
        the texture matrices.
    method:
        ``"fbs"`` (fixed bin size) or ``"fbn"`` (fixed bin number).
    bin_width:
        Bin width for ``"fbs"``, else ``None``.
    bin_number:
        Requested bin count for ``"fbn"``, else ``None``.
    intensity_range:
        The ``(lower, upper)`` intensity bounds used for discretisation.

    """

    levels: np.ndarray
    mask: np.ndarray
    n_levels: int
    method: str
    bin_width: float | None
    bin_number: int | None
    intensity_range: tuple[float, float]

    @property
    def roi_levels(self) -> np.ndarray:
        """The discretised levels inside the ROI, as a flat ``int32`` array."""
        return self.levels[self.mask]


def discretise(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    method: Literal["fbs", "fbn"] = "fbs",
    bin_width: float | None = None,
    bin_number: int | None = None,
    intensity_range: tuple[float, float] | None = None,
) -> Discretisation:
    """Discretise ROI intensities onto integer grey levels (IBSI 2.7).

    Fixed bin size (``"fbs"``) is the IBSI-recommended default for calibrated
    intensity scales such as Hounsfield units, because it preserves the
    physical meaning of a bin across images.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI intensity mask of the same shape.
    method:
        ``"fbs"``: ``level = floor((x - lower) / bin_width) + 1``.
        ``"fbn"``: ``level = floor(bin_number * (x - lower) / (upper - lower)) + 1``,
        with the maximum intensity assigned to the last bin.
    bin_width:
        Required for ``"fbs"``.
    bin_number:
        Required for ``"fbn"``.
    intensity_range:
        Explicit ``(lower, upper)`` bounds, e.g. from re-segmentation.  When
        omitted, the ROI minimum and maximum are used.

    Returns
    -------
    Discretisation

    Raises
    ------
    FeatureError
        On an empty ROI, a mis-specified method, a non-positive bin width or
        bin number, or a constant ROI under ``"fbn"`` (where the bin edges
        would collapse).

    Examples
    --------
    A fixed bin size of 1 leaves integer intensities untouched, which is how
    the IBSI digital phantom is meant to be processed:

    >>> volume = np.array([[[1, 3], [4, 6]]], dtype=float)
    >>> mask = np.ones_like(volume, dtype=bool)
    >>> discretise(volume, mask, method="fbs", bin_width=1.0).roi_levels
    array([1, 3, 4, 6], dtype=int32)

    """
    _check_roi(mask, volume)

    roi = volume[mask].astype(np.float64)
    if not np.all(np.isfinite(roi)):
        raise FeatureError("ROI contains non-finite intensities; refusing to discretise.")

    if intensity_range is None:
        lower, upper = float(roi.min()), float(roi.max())
    else:
        lower, upper = float(intensity_range[0]), float(intensity_range[1])
        if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
            raise FeatureError(
                f"intensity_range must be finite and increasing; got ({lower}, {upper})."
            )
        roi = np.clip(roi, lower, upper)

    levels = np.zeros(volume.shape, dtype=np.int32)

    if method == "fbs":
        if bin_width is None or not np.isfinite(bin_width) or bin_width <= 0.0:
            raise FeatureError(f"method='fbs' requires a positive bin_width; got {bin_width}.")
        if bin_number is not None:
            raise FeatureError("bin_number must not be given when method='fbs'.")
        # Bins are half-open, [lower + (k-1)*w, lower + k*w).  The upper bound
        # therefore opens a bin of its own, which is why the IBSI digital
        # phantom (intensities 1..6, w=1) spans six levels rather than five.
        discrete = np.floor((roi - lower) / bin_width).astype(np.int64) + 1
        n_levels = int(np.floor((upper - lower) / bin_width)) + 1
    elif method == "fbn":
        if bin_number is None or bin_number < 1:
            raise FeatureError(f"method='fbn' requires bin_number >= 1; got {bin_number}.")
        if bin_width is not None:
            raise FeatureError("bin_width must not be given when method='fbn'.")
        if upper <= lower:
            raise FeatureError(
                "method='fbn' is undefined for a constant ROI (upper == lower); "
                "use method='fbs' or supply an intensity_range."
            )
        discrete = np.floor(bin_number * (roi - lower) / (upper - lower)).astype(np.int64) + 1
        discrete = np.minimum(discrete, bin_number)
        n_levels = int(bin_number)
    else:
        raise FeatureError(f"method must be 'fbs' or 'fbn'; got {method!r}.")

    if int(discrete.min()) < 1 or int(discrete.max()) > n_levels:
        raise FeatureError(
            f"discretisation produced levels outside 1..{n_levels} "
            f"(got {int(discrete.min())}..{int(discrete.max())}); this is an internal error."
        )

    levels[mask] = discrete.astype(np.int32)

    return Discretisation(
        levels=levels,
        mask=mask,
        n_levels=n_levels,
        method=method,
        bin_width=None if bin_width is None else float(bin_width),
        bin_number=None if bin_number is None else int(bin_number),
        intensity_range=(lower, upper),
    )


# ---------------------------------------------------------------------------
# Intensity-based statistics (IBSI 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntensityStatistics:
    """The 18 IBSI intensity-based statistical features (IBSI 3.3).

    Computed on the raw (undiscretised) intensities inside the ROI.
    """

    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float
    median: float
    minimum: float
    percentile_10: float
    percentile_90: float
    maximum: float
    interquartile_range: float
    intensity_range: float
    mean_absolute_deviation: float
    robust_mean_absolute_deviation: float
    median_absolute_deviation: float
    coefficient_of_variation: float
    quartile_coefficient_of_dispersion: float
    energy: float
    root_mean_square: float

    #: Field name -> IBSI feature tag.
    TAGS = {
        "mean": "stat_mean",
        "variance": "stat_var",
        "skewness": "stat_skew",
        "excess_kurtosis": "stat_kurt",
        "median": "stat_median",
        "minimum": "stat_min",
        "percentile_10": "stat_p10",
        "percentile_90": "stat_p90",
        "maximum": "stat_max",
        "interquartile_range": "stat_iqr",
        "intensity_range": "stat_range",
        "mean_absolute_deviation": "stat_mad",
        "robust_mean_absolute_deviation": "stat_rmad",
        "median_absolute_deviation": "stat_medad",
        "coefficient_of_variation": "stat_cov",
        "quartile_coefficient_of_dispersion": "stat_qcod",
        "energy": "stat_energy",
        "root_mean_square": "stat_rms",
    }

    def to_dict(self) -> dict[str, float]:
        """Map IBSI feature tags to values."""
        return {self.TAGS[f.name]: getattr(self, f.name) for f in fields(self)}


def intensity_statistics(volume: np.ndarray, mask: np.ndarray) -> IntensityStatistics:
    """Compute the IBSI intensity-based statistical features (IBSI 3.3).

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI intensity mask.

    Returns
    -------
    IntensityStatistics

    Raises
    ------
    FeatureError
        If the ROI is empty, non-finite, or has zero variance.

    """
    _check_roi(mask, volume)
    roi = volume[mask].astype(np.float64)
    if not np.all(np.isfinite(roi)):
        raise FeatureError("ROI contains non-finite intensities.")
    return IntensityStatistics(**_dispersion_stats(roi))


# ---------------------------------------------------------------------------
# Intensity histogram (IBSI 3.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntensityHistogramFeatures:
    """The 23 IBSI intensity-histogram features (IBSI 3.4).

    Computed on discretised grey levels.  Sixteen of them mirror the intensity
    statistics of IBSI 3.3; ``mode`` and the entropy, uniformity and gradient
    features describe the histogram itself.
    """

    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float
    median: float
    minimum: float
    percentile_10: float
    percentile_90: float
    maximum: float
    mode: float
    interquartile_range: float
    intensity_range: float
    mean_absolute_deviation: float
    robust_mean_absolute_deviation: float
    median_absolute_deviation: float
    coefficient_of_variation: float
    quartile_coefficient_of_dispersion: float
    entropy: float
    uniformity: float
    maximum_histogram_gradient: float
    maximum_histogram_gradient_intensity: float
    minimum_histogram_gradient: float
    minimum_histogram_gradient_intensity: float

    TAGS = {
        "mean": "ih_mean",
        "variance": "ih_var",
        "skewness": "ih_skew",
        "excess_kurtosis": "ih_kurt",
        "median": "ih_median",
        "minimum": "ih_min",
        "percentile_10": "ih_p10",
        "percentile_90": "ih_p90",
        "maximum": "ih_max",
        "mode": "ih_mode",
        "interquartile_range": "ih_iqr",
        "intensity_range": "ih_range",
        "mean_absolute_deviation": "ih_mad",
        "robust_mean_absolute_deviation": "ih_rmad",
        "median_absolute_deviation": "ih_medad",
        "coefficient_of_variation": "ih_cov",
        "quartile_coefficient_of_dispersion": "ih_qcod",
        "entropy": "ih_entropy",
        "uniformity": "ih_uniformity",
        "maximum_histogram_gradient": "ih_max_grad",
        "maximum_histogram_gradient_intensity": "ih_max_grad_g",
        "minimum_histogram_gradient": "ih_min_grad",
        "minimum_histogram_gradient_intensity": "ih_min_grad_g",
    }

    def to_dict(self) -> dict[str, float]:
        """Map IBSI feature tags to values."""
        return {self.TAGS[f.name]: getattr(self, f.name) for f in fields(self)}


def intensity_histogram(disc: Discretisation) -> IntensityHistogramFeatures:
    """Compute the IBSI intensity-histogram features (IBSI 3.4).

    The histogram spans every level in ``1 .. n_levels``, including levels that
    no voxel occupies.  Empty bins do not affect entropy or uniformity, but
    they do shift the histogram gradients, which are defined on the bin index.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.

    Returns
    -------
    IntensityHistogramFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or occupies a single grey level.

    """
    levels = disc.roi_levels.astype(np.int64)
    if levels.size == 0:
        raise FeatureError("ROI mask is empty; no features can be computed.")

    # IBSI 3.4 shares 16 statistics with IBSI 3.3, but defines neither energy
    # nor root mean square on the histogram.
    stats = _dispersion_stats(levels)
    del stats["energy"], stats["root_mean_square"]

    counts = np.bincount(levels, minlength=disc.n_levels + 1)[1:].astype(np.float64)
    probabilities = counts / counts.sum()

    # Mode: the most occupied level, lowest wins a tie.
    mode = float(int(np.argmax(counts)) + 1)

    # Histogram gradient (IBSI 3.4.22-3.4.25) on *counts*, by central
    # difference, with forward/backward differences at the two ends.
    if counts.size < 2:
        raise FeatureError(
            "histogram gradients require at least 2 grey levels; "
            f"the discretisation produced n_levels={disc.n_levels}."
        )
    gradient = np.empty_like(counts)
    gradient[0] = counts[1] - counts[0]
    gradient[-1] = counts[-1] - counts[-2]
    if counts.size > 2:
        gradient[1:-1] = (counts[2:] - counts[:-2]) / 2.0

    return IntensityHistogramFeatures(
        mode=mode,
        entropy=_entropy(probabilities),
        uniformity=float(np.sum(probabilities**2)),
        maximum_histogram_gradient=float(gradient.max()),
        maximum_histogram_gradient_intensity=float(int(np.argmax(gradient)) + 1),
        minimum_histogram_gradient=float(gradient.min()),
        minimum_histogram_gradient_intensity=float(int(np.argmin(gradient)) + 1),
        **stats,
    )


# ---------------------------------------------------------------------------
# Directions and texture-matrix machinery
# ---------------------------------------------------------------------------


def _directions_3d() -> list[tuple[int, int, int]]:
    """The 13 unique 3D directions at Chebyshev distance 1.

    One representative of each ``{d, -d}`` pair: the one whose first non-zero
    component is positive.  Texture matrices are symmetrised, so the opposite
    direction is accounted for.
    """
    out: list[tuple[int, int, int]] = []
    for dz, dy, dx in itertools.product((-1, 0, 1), repeat=3):
        if (dz, dy, dx) == (0, 0, 0):
            continue
        first = next(c for c in (dz, dy, dx) if c != 0)
        if first > 0:
            out.append((dz, dy, dx))
    return out


def _directions_2d() -> list[tuple[int, int]]:
    """The 4 unique in-plane directions at Chebyshev distance 1, as ``(dy, dx)``."""
    return [(0, 1), (1, -1), (1, 0), (1, 1)]


def _shift_pair(array: np.ndarray, offset: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return the overlapping views of ``array`` and ``array`` shifted by ``offset``."""
    src, dst = [], []
    for n, d in zip(array.shape, offset, strict=True):
        if d > 0:
            src.append(slice(0, max(n - d, 0)))
            dst.append(slice(min(d, n), n))
        elif d < 0:
            src.append(slice(min(-d, n), n))
            dst.append(slice(0, max(n + d, 0)))
        else:
            src.append(slice(None))
            dst.append(slice(None))
    return array[tuple(src)], array[tuple(dst)]


def _cooccurrence_matrix(
    levels: np.ndarray, mask: np.ndarray, offset: tuple[int, ...], n_levels: int
) -> np.ndarray:
    """Symmetric grey level co-occurrence matrix for one direction."""
    a_src, a_dst = _shift_pair(levels, offset)
    m_src, m_dst = _shift_pair(mask, offset)
    valid = m_src & m_dst
    if not valid.any():
        return np.zeros((n_levels, n_levels), dtype=np.float64)

    i = a_src[valid].astype(np.int64) - 1
    j = a_dst[valid].astype(np.int64) - 1
    flat = np.bincount(i * n_levels + j, minlength=n_levels * n_levels)
    matrix = flat.reshape(n_levels, n_levels).astype(np.float64)
    return matrix + matrix.T


def _run_length_matrix(
    levels: np.ndarray, mask: np.ndarray, offset: tuple[int, ...], n_levels: int, max_run: int
) -> np.ndarray:
    """Grey level run length matrix for one direction.

    Voxels are grouped into the lines parallel to ``offset``.  For every voxel
    the number of backward steps that stay inside the array, ``t``, identifies
    its position along its line, and ``coord - t * offset`` identifies the
    line.  Sorting by ``(line, t)`` lays every line out contiguously, after
    which runs are the maximal stretches of in-ROI voxels sharing a grey level.
    """
    shape = levels.shape
    coords = np.indices(shape)

    steps_back = [
        coords[axis] if d > 0 else (shape[axis] - 1 - coords[axis])
        for axis, d in enumerate(offset)
        if d != 0
    ]
    t = np.min(np.stack(steps_back), axis=0)
    start = tuple(coords[axis] - t * d for axis, d in enumerate(offset))
    line_id = np.ravel_multi_index(start, shape)

    order = np.lexsort((t.ravel(), line_id.ravel()))
    lev = levels.ravel()[order]
    msk = mask.ravel()[order]
    lid = line_id.ravel()[order]

    continues = np.zeros(lev.size, dtype=bool)
    continues[1:] = msk[:-1] & (lid[1:] == lid[:-1]) & (lev[1:] == lev[:-1])
    starts = msk & ~continues

    n_runs = int(starts.sum())
    matrix = np.zeros((n_levels, max_run), dtype=np.float64)
    if n_runs == 0:
        return matrix

    run_index = np.cumsum(starts) - 1
    lengths = np.bincount(run_index[msk], minlength=n_runs)
    run_levels = lev[starts].astype(np.int64)

    flat = np.bincount((run_levels - 1) * max_run + (lengths - 1), minlength=n_levels * max_run)
    return flat.reshape(n_levels, max_run).astype(np.float64)


@dataclass(frozen=True)
class _Matrix:
    """A texture matrix together with the voxel count it was derived from."""

    counts: np.ndarray
    n_voxels: int

    def __add__(self, other: _Matrix) -> _Matrix:
        return _Matrix(self.counts + other.counts, self.n_voxels + other.n_voxels)


def _slice_matrices(
    disc: Discretisation,
    build: Callable[[np.ndarray, np.ndarray, tuple[int, int]], np.ndarray],
    directions_2d: list[tuple[int, int]],
) -> list[list[_Matrix]]:
    """``matrices[z][d]`` for every slice and in-plane direction."""
    out = []
    for z in range(disc.levels.shape[0]):
        lev, msk = disc.levels[z], disc.mask[z]
        n_vox = int(msk.sum())
        out.append([_Matrix(build(lev, msk, d), n_vox) for d in directions_2d])
    return out


def _aggregate(
    disc: Discretisation,
    aggregation: Aggregation,
    build_2d: Callable[[np.ndarray, np.ndarray, tuple[int, int]], np.ndarray],
    build_3d: Callable[[np.ndarray, np.ndarray, tuple[int, int, int]], np.ndarray],
) -> list[_Matrix]:
    """Produce the list of matrices whose features are to be averaged.

    A single-element list means the features are computed once, from one merged
    matrix.  Matrices with no counts (e.g. an empty slice) are dropped.
    """
    if aggregation not in AGGREGATIONS:
        raise FeatureError(f"aggregation must be one of {AGGREGATIONS}; got {aggregation!r}.")

    if aggregation.startswith("3D"):
        n_vox = int(disc.mask.sum())
        per_direction = [
            _Matrix(build_3d(disc.levels, disc.mask, d), n_vox) for d in _directions_3d()
        ]
        matrices = per_direction if aggregation == "3D_avg" else [_sum_matrices(per_direction)]
    else:
        by_slice = _slice_matrices(disc, build_2d, _directions_2d())
        if aggregation == "2D_avg":
            matrices = [m for row in by_slice for m in row]
        elif aggregation == "2D_comb":
            matrices = [_sum_matrices(row) for row in by_slice]
        elif aggregation == "2_5D_avg":
            matrices = [_sum_matrices([row[k] for row in by_slice]) for k in range(4)]
        else:  # 2_5D_comb
            matrices = [_sum_matrices([m for row in by_slice for m in row])]

    kept = [m for m in matrices if m.counts.sum() > 0]
    if not kept:
        raise FeatureError(
            f"aggregation {aggregation!r} produced no non-empty texture matrix; "
            "the ROI is too small to contain a single neighbouring voxel pair."
        )
    return kept


def _sum_matrices(matrices: list[_Matrix]) -> _Matrix:
    """Merge texture matrices by summing counts and the voxels behind them."""
    total = matrices[0]
    for m in matrices[1:]:
        total = total + m
    return total


def _mean_of_features(per_matrix: list[dict[str, float]]) -> dict[str, float]:
    """Average each feature over the matrices it was computed from."""
    keys = per_matrix[0].keys()
    n = len(per_matrix)
    return {k: float(sum(d[k] for d in per_matrix) / n) for k in keys}


# ---------------------------------------------------------------------------
# Grey level co-occurrence matrix (IBSI 3.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLCMFeatures:
    """The 25 IBSI grey level co-occurrence features (IBSI 3.6)."""

    joint_maximum: float
    joint_average: float
    joint_variance: float
    joint_entropy: float
    difference_average: float
    difference_variance: float
    difference_entropy: float
    sum_average: float
    sum_variance: float
    sum_entropy: float
    angular_second_moment: float
    contrast: float
    dissimilarity: float
    inverse_difference: float
    normalised_inverse_difference: float
    inverse_difference_moment: float
    normalised_inverse_difference_moment: float
    inverse_variance: float
    correlation: float
    autocorrelation: float
    cluster_tendency: float
    cluster_shade: float
    cluster_prominence: float
    information_correlation_1: float
    information_correlation_2: float

    #: Field name -> IBSI feature tag stem (the aggregation suffix is appended).
    TAGS = {
        "joint_maximum": "cm_joint_max",
        "joint_average": "cm_joint_avg",
        "joint_variance": "cm_joint_var",
        "joint_entropy": "cm_joint_entr",
        "difference_average": "cm_diff_avg",
        "difference_variance": "cm_diff_var",
        "difference_entropy": "cm_diff_entr",
        "sum_average": "cm_sum_avg",
        "sum_variance": "cm_sum_var",
        "sum_entropy": "cm_sum_entr",
        "angular_second_moment": "cm_energy",
        "contrast": "cm_contrast",
        "dissimilarity": "cm_dissimilarity",
        "inverse_difference": "cm_inv_diff",
        "normalised_inverse_difference": "cm_inv_diff_norm",
        "inverse_difference_moment": "cm_inv_diff_mom",
        "normalised_inverse_difference_moment": "cm_inv_diff_mom_norm",
        "inverse_variance": "cm_inv_var",
        "correlation": "cm_corr",
        "autocorrelation": "cm_auto_corr",
        "cluster_tendency": "cm_clust_tend",
        "cluster_shade": "cm_clust_shade",
        "cluster_prominence": "cm_clust_prom",
        "information_correlation_1": "cm_info_corr1",
        "information_correlation_2": "cm_info_corr2",
    }

    def to_dict(self, aggregation: Aggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def _glcm_from_matrix(counts: np.ndarray) -> dict[str, float]:
    """The 25 GLCM features of one symmetric count matrix.

    Symmetry is required, not merely expected: the marginals ``px`` and ``py``
    are then equal, which is what lets ``correlation`` and the information
    correlations be computed from a single marginal.  It also guarantees
    ``p_ij > 0 => px_i > 0``, so the ``log(px_i * py_j)`` in ``HXY1`` is finite.
    """
    if not np.array_equal(counts, counts.T):
        raise FeatureError("co-occurrence matrix is not symmetric; this is an internal error.")

    n_levels = counts.shape[0]
    p = counts / counts.sum()

    i = np.arange(1, n_levels + 1, dtype=np.float64)
    ii, jj = np.meshgrid(i, i, indexing="ij")

    px = p.sum(axis=1)  # equals py, because p is symmetric
    mu = float(np.sum(i * px))
    sigma_sq = float(np.sum((i - mu) ** 2 * px))

    diff = np.abs(ii - jj)
    total = ii + jj

    # Difference and sum distributions.
    k_diff = np.arange(n_levels, dtype=np.float64)
    p_diff = np.bincount(diff.astype(np.int64).ravel(), weights=p.ravel(), minlength=n_levels)
    diff_avg = float(np.sum(k_diff * p_diff))

    k_sum = np.arange(2, 2 * n_levels + 1, dtype=np.float64)
    p_sum = np.bincount(
        (total.astype(np.int64) - 2).ravel(), weights=p.ravel(), minlength=2 * n_levels - 1
    )
    sum_avg = float(np.sum(k_sum * p_sum))

    if sigma_sq <= 0.0:
        raise FeatureError(
            "the co-occurrence matrix has zero marginal variance (the ROI holds a "
            "single grey level); correlation and information correlation are undefined."
        )

    off = ~np.eye(n_levels, dtype=bool)

    hxy = _entropy(p)
    hx = _entropy(px)
    outer = np.outer(px, px)
    nz = p > 0.0
    hxy1 = float(-np.sum(p[nz] * np.log(outer[nz]) / _LOG2))
    onz = outer > 0.0
    hxy2 = float(-np.sum(outer[onz] * np.log(outer[onz]) / _LOG2))

    if hx <= 0.0:
        raise FeatureError("marginal entropy is zero; information correlation 1 is undefined.")

    return {
        "joint_maximum": float(p.max()),
        "joint_average": mu,
        "joint_variance": float(np.sum((ii - mu) ** 2 * p)),
        "joint_entropy": hxy,
        "difference_average": diff_avg,
        "difference_variance": float(np.sum((k_diff - diff_avg) ** 2 * p_diff)),
        "difference_entropy": _entropy(p_diff),
        "sum_average": sum_avg,
        "sum_variance": float(np.sum((k_sum - sum_avg) ** 2 * p_sum)),
        "sum_entropy": _entropy(p_sum),
        "angular_second_moment": float(np.sum(p**2)),
        "contrast": float(np.sum((ii - jj) ** 2 * p)),
        "dissimilarity": float(np.sum(diff * p)),
        "inverse_difference": float(np.sum(p / (1.0 + diff))),
        "normalised_inverse_difference": float(np.sum(p / (1.0 + diff / n_levels))),
        "inverse_difference_moment": float(np.sum(p / (1.0 + (ii - jj) ** 2))),
        "normalised_inverse_difference_moment": float(
            np.sum(p / (1.0 + (ii - jj) ** 2 / n_levels**2))
        ),
        "inverse_variance": float(np.sum(p[off] / (ii - jj)[off] ** 2)),
        "correlation": float((np.sum(ii * jj * p) - mu**2) / sigma_sq),
        "autocorrelation": float(np.sum(ii * jj * p)),
        "cluster_tendency": float(np.sum((total - 2 * mu) ** 2 * p)),
        "cluster_shade": float(np.sum((total - 2 * mu) ** 3 * p)),
        "cluster_prominence": float(np.sum((total - 2 * mu) ** 4 * p)),
        "information_correlation_1": float((hxy - hxy1) / hx),
        "information_correlation_2": float(np.sqrt(max(1.0 - np.exp(-2.0 * (hxy2 - hxy)), 0.0))),
    }


def glcm_features(disc: Discretisation, aggregation: Aggregation = "3D_comb") -> GLCMFeatures:
    """Compute the IBSI grey level co-occurrence features (IBSI 3.6).

    Co-occurrence is counted between voxel pairs at Chebyshev distance 1.  Each
    direction's matrix is symmetrised, which is equivalent to counting the
    direction and its opposite.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`AGGREGATIONS`.

    Returns
    -------
    GLCMFeatures

    Raises
    ------
    FeatureError
        If the ROI holds a single grey level (correlation-type features are
        then undefined), or contains no neighbouring voxel pair at all.

    """
    matrices = _aggregate(
        disc,
        aggregation,
        build_2d=lambda lev, msk, d: _cooccurrence_matrix(lev, msk, d, disc.n_levels),
        build_3d=lambda lev, msk, d: _cooccurrence_matrix(lev, msk, d, disc.n_levels),
    )
    per_matrix = [_glcm_from_matrix(m.counts) for m in matrices]
    return GLCMFeatures(**_mean_of_features(per_matrix))


# ---------------------------------------------------------------------------
# Grey level run length matrix (IBSI 3.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLRLMFeatures:
    """The 16 IBSI grey level run length features (IBSI 3.7)."""

    short_runs_emphasis: float
    long_runs_emphasis: float
    low_grey_level_run_emphasis: float
    high_grey_level_run_emphasis: float
    short_run_low_grey_level_emphasis: float
    short_run_high_grey_level_emphasis: float
    long_run_low_grey_level_emphasis: float
    long_run_high_grey_level_emphasis: float
    grey_level_non_uniformity: float
    normalised_grey_level_non_uniformity: float
    run_length_non_uniformity: float
    normalised_run_length_non_uniformity: float
    run_percentage: float
    grey_level_variance: float
    run_length_variance: float
    run_entropy: float

    TAGS = {
        "short_runs_emphasis": "rlm_sre",
        "long_runs_emphasis": "rlm_lre",
        "low_grey_level_run_emphasis": "rlm_lgre",
        "high_grey_level_run_emphasis": "rlm_hgre",
        "short_run_low_grey_level_emphasis": "rlm_srlge",
        "short_run_high_grey_level_emphasis": "rlm_srhge",
        "long_run_low_grey_level_emphasis": "rlm_lrlge",
        "long_run_high_grey_level_emphasis": "rlm_lrhge",
        "grey_level_non_uniformity": "rlm_glnu",
        "normalised_grey_level_non_uniformity": "rlm_glnu_norm",
        "run_length_non_uniformity": "rlm_rlnu",
        "normalised_run_length_non_uniformity": "rlm_rlnu_norm",
        "run_percentage": "rlm_r_perc",
        "grey_level_variance": "rlm_gl_var",
        "run_length_variance": "rlm_rl_var",
        "run_entropy": "rlm_rl_entr",
    }

    def to_dict(self, aggregation: Aggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def _glrlm_from_matrix(counts: np.ndarray, n_voxels: int) -> dict[str, float]:
    """The 16 GLRLM features of one count matrix.

    ``n_voxels`` is the number of ROI voxels that fed this matrix; for merged
    matrices it is the sum over the merged directions, so that run percentage
    stays a fraction.
    """
    n_levels, max_run = counts.shape
    n_runs = float(counts.sum())

    i = np.arange(1, n_levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, max_run + 1, dtype=np.float64)[None, :]

    p = counts / n_runs
    marginal_level = counts.sum(axis=1)
    marginal_run = counts.sum(axis=0)

    mu_i = float(np.sum(i * p))
    mu_j = float(np.sum(j * p))

    return {
        "short_runs_emphasis": float(np.sum(counts / j**2) / n_runs),
        "long_runs_emphasis": float(np.sum(counts * j**2) / n_runs),
        "low_grey_level_run_emphasis": float(np.sum(counts / i**2) / n_runs),
        "high_grey_level_run_emphasis": float(np.sum(counts * i**2) / n_runs),
        "short_run_low_grey_level_emphasis": float(np.sum(counts / (i**2 * j**2)) / n_runs),
        "short_run_high_grey_level_emphasis": float(np.sum(counts * i**2 / j**2) / n_runs),
        "long_run_low_grey_level_emphasis": float(np.sum(counts * j**2 / i**2) / n_runs),
        "long_run_high_grey_level_emphasis": float(np.sum(counts * i**2 * j**2) / n_runs),
        "grey_level_non_uniformity": float(np.sum(marginal_level**2) / n_runs),
        "normalised_grey_level_non_uniformity": float(np.sum(marginal_level**2) / n_runs**2),
        "run_length_non_uniformity": float(np.sum(marginal_run**2) / n_runs),
        "normalised_run_length_non_uniformity": float(np.sum(marginal_run**2) / n_runs**2),
        "run_percentage": float(n_runs / n_voxels),
        "grey_level_variance": float(np.sum((i - mu_i) ** 2 * p)),
        "run_length_variance": float(np.sum((j - mu_j) ** 2 * p)),
        "run_entropy": _entropy(p),
    }


def glrlm_features(disc: Discretisation, aggregation: Aggregation = "3D_comb") -> GLRLMFeatures:
    """Compute the IBSI grey level run length features (IBSI 3.7).

    A run is a maximal stretch of consecutive ROI voxels sharing a grey level
    along one direction.  Leaving the ROI ends a run.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`AGGREGATIONS`.

    Returns
    -------
    GLRLMFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or yields no runs.

    """
    max_run = int(max(disc.levels.shape))
    matrices = _aggregate(
        disc,
        aggregation,
        build_2d=lambda lev, msk, d: _run_length_matrix(lev, msk, d, disc.n_levels, max_run),
        build_3d=lambda lev, msk, d: _run_length_matrix(lev, msk, d, disc.n_levels, max_run),
    )
    per_matrix = [_glrlm_from_matrix(m.counts, m.n_voxels) for m in matrices]
    return GLRLMFeatures(**_mean_of_features(per_matrix))


# ---------------------------------------------------------------------------
# Zone-, neighbourhood- and dependence-based families (IBSI 3.8 - 3.11)
# ---------------------------------------------------------------------------

ZoneAggregation = Literal["2D", "2_5D", "3D"]

#: Aggregation methods for the families that have no directionality: the zone,
#: neighbourhood and dependence matrices (IBSI 3.8 - 3.11).
#:
#: ``2D``
#:     One matrix per slice; features averaged over slices.
#: ``2_5D``
#:     Slice matrices merged into one; features computed once.
#: ``3D``
#:     One matrix for the whole volume.
ZONE_AGGREGATIONS: tuple[ZoneAggregation, ...] = ("2D", "2_5D", "3D")


def _offsets(ndim: int) -> list[tuple[int, ...]]:
    """Every neighbour offset at Chebyshev distance 1: 8 in 2D, 26 in 3D."""
    return [d for d in itertools.product((-1, 0, 1), repeat=ndim) if any(d)]


def _neighbour_views(
    levels: np.ndarray, mask: np.ndarray, offset: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """The neighbour level and in-ROI flag at ``offset`` from every voxel.

    Voxels beyond the array edge are treated as outside the ROI, exactly like
    the voxels the mask excludes.
    """
    padded_levels = np.pad(levels, 1)
    padded_mask = np.pad(mask, 1, constant_values=False)
    window = tuple(slice(1 + o, 1 + o + n) for o, n in zip(offset, levels.shape, strict=True))
    return padded_levels[window], padded_mask[window]


@dataclass(frozen=True)
class _Zones:
    """Zones as a list, not a matrix.

    A dense zone matrix is ``n_levels`` by ``max_zone_size``, and the largest
    zone can be the whole ROI, so materialising it costs O(n_levels * n_voxels)
    for no benefit.  Every zone feature is a sum over zones, so the zone list is
    both smaller and closer to the definitions.
    """

    levels: np.ndarray
    values: np.ndarray  # zone size (GLSZM) or zone-to-border distance (GLDZM)
    n_voxels: int

    def __add__(self, other: _Zones) -> _Zones:
        return _Zones(
            np.concatenate([self.levels, other.levels]),
            np.concatenate([self.values, other.values]),
            self.n_voxels + other.n_voxels,
        )


def _label_zones(levels: np.ndarray, mask: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Connected components of equal grey level, one label array per level.

    Connectivity is full: 8-connected within a slice, 26-connected in 3D.
    """
    structure = np.ones((3,) * levels.ndim, dtype=int)
    return [
        ndimage_label((levels == level) & mask, structure=structure)[0]
        for level in range(1, n_levels + 1)
    ]


def _size_zones(levels: np.ndarray, mask: np.ndarray, n_levels: int) -> _Zones:
    """Grey level size zones: one entry per connected component, with its size."""
    zone_levels, zone_sizes = [], []
    for level, labelled in enumerate(_label_zones(levels, mask, n_levels), start=1):
        sizes = np.bincount(labelled.ravel())[1:]
        if sizes.size == 0:
            continue
        zone_levels.append(np.full(sizes.size, level, dtype=np.int64))
        zone_sizes.append(sizes.astype(np.int64))
    return _Zones(
        np.concatenate(zone_levels) if zone_levels else np.empty(0, dtype=np.int64),
        np.concatenate(zone_sizes) if zone_sizes else np.empty(0, dtype=np.int64),
        int(mask.sum()),
    )


def _border_distance(mask: np.ndarray) -> np.ndarray:
    """Steps from each ROI voxel to the nearest voxel outside the ROI.

    Moves are face-connected (4-connectivity in 2D, 6-connectivity in 3D), so a
    voxel touching the ROI border -- or the edge of the array -- has distance 1.
    """
    padded = np.pad(mask, 1, constant_values=False)
    distance = distance_transform_cdt(padded, metric="taxicab")
    return distance[(slice(1, -1),) * mask.ndim]


def _distance_zones(levels: np.ndarray, mask: np.ndarray, n_levels: int) -> _Zones:
    """Grey level distance zones: each zone carries its minimum border distance."""
    distance = _border_distance(mask)
    zone_levels, zone_distances = [], []
    for level, labelled in enumerate(_label_zones(levels, mask, n_levels), start=1):
        n_zones = int(labelled.max())
        if n_zones == 0:
            continue
        inside = labelled > 0
        minima = np.full(n_zones + 1, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(minima, labelled[inside], distance[inside].astype(np.int64))
        zone_levels.append(np.full(n_zones, level, dtype=np.int64))
        zone_distances.append(minima[1:])
    return _Zones(
        np.concatenate(zone_levels) if zone_levels else np.empty(0, dtype=np.int64),
        np.concatenate(zone_distances) if zone_distances else np.empty(0, dtype=np.int64),
        int(mask.sum()),
    )


def _zone_statistics(zones: _Zones) -> dict[str, float]:
    """The 16 zone features, under generic names shared by GLSZM and GLDZM.

    Each feature is an average over zones, because the normalised matrix entry
    ``p_ij`` is exactly the fraction of zones at grey level ``i`` and value
    ``j``.
    """
    n_zones = zones.levels.size
    if n_zones == 0:
        raise FeatureError("the ROI contains no zones; no zone features can be computed.")

    i = zones.levels.astype(np.float64)
    j = zones.values.astype(np.float64)

    level_counts = np.bincount(zones.levels)
    value_counts = np.bincount(zones.values)

    width = int(zones.values.max()) + 1
    joint = np.bincount(zones.levels * width + zones.values)

    return {
        "small_emphasis": float(np.mean(1.0 / j**2)),
        "large_emphasis": float(np.mean(j**2)),
        "low_grey_level_emphasis": float(np.mean(1.0 / i**2)),
        "high_grey_level_emphasis": float(np.mean(i**2)),
        "small_low_grey_level_emphasis": float(np.mean(1.0 / (i**2 * j**2))),
        "small_high_grey_level_emphasis": float(np.mean(i**2 / j**2)),
        "large_low_grey_level_emphasis": float(np.mean(j**2 / i**2)),
        "large_high_grey_level_emphasis": float(np.mean(i**2 * j**2)),
        "grey_level_non_uniformity": float(np.sum(level_counts**2) / n_zones),
        "normalised_grey_level_non_uniformity": float(np.sum(level_counts**2) / n_zones**2),
        "value_non_uniformity": float(np.sum(value_counts**2) / n_zones),
        "normalised_value_non_uniformity": float(np.sum(value_counts**2) / n_zones**2),
        "zone_percentage": float(n_zones / zones.n_voxels),
        "grey_level_variance": float(np.mean((i - i.mean()) ** 2)),
        "value_variance": float(np.mean((j - j.mean()) ** 2)),
        "value_entropy": _entropy(joint[joint > 0] / n_zones),
    }


def _zone_aggregate(
    disc: Discretisation,
    aggregation: ZoneAggregation,
    build: Callable[[np.ndarray, np.ndarray], _Zones],
) -> list[_Zones]:
    """Slice up the ROI -- or not -- according to ``aggregation``."""
    if aggregation not in ZONE_AGGREGATIONS:
        raise FeatureError(f"aggregation must be one of {ZONE_AGGREGATIONS}; got {aggregation!r}.")
    if aggregation == "3D":
        return [build(disc.levels, disc.mask)]

    per_slice = [
        build(disc.levels[z], disc.mask[z])
        for z in range(disc.levels.shape[0])
        if disc.mask[z].any()
    ]
    if not per_slice:
        raise FeatureError("ROI mask is empty; no features can be computed.")
    if aggregation == "2D":
        return per_slice

    merged = per_slice[0]
    for zones in per_slice[1:]:
        merged = merged + zones
    return [merged]


#: The order in which :func:`_zone_statistics` keys map onto the GLSZM/GLDZM
#: dataclass fields, which are declared in the same order.
_ZONE_FIELD_ORDER = (
    "small_emphasis",
    "large_emphasis",
    "low_grey_level_emphasis",
    "high_grey_level_emphasis",
    "small_low_grey_level_emphasis",
    "small_high_grey_level_emphasis",
    "large_low_grey_level_emphasis",
    "large_high_grey_level_emphasis",
    "grey_level_non_uniformity",
    "normalised_grey_level_non_uniformity",
    "value_non_uniformity",
    "normalised_value_non_uniformity",
    "zone_percentage",
    "grey_level_variance",
    "value_variance",
    "value_entropy",
)


_ZoneFeaturesT = TypeVar("_ZoneFeaturesT")


def _zone_features(
    disc: Discretisation,
    aggregation: ZoneAggregation,
    build: Callable[[np.ndarray, np.ndarray], _Zones],
    cls: type[_ZoneFeaturesT],
) -> _ZoneFeaturesT:
    zones = _zone_aggregate(disc, aggregation, build)
    generic = _mean_of_features([_zone_statistics(z) for z in zones])
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    return cls(**{name: generic[key] for name, key in zip(names, _ZONE_FIELD_ORDER, strict=True)})


# --- Grey level size zone matrix (IBSI 3.8) --------------------------------


@dataclass(frozen=True)
class GLSZMFeatures:
    """The 16 IBSI grey level size zone features (IBSI 3.8)."""

    small_zone_emphasis: float
    large_zone_emphasis: float
    low_grey_level_zone_emphasis: float
    high_grey_level_zone_emphasis: float
    small_zone_low_grey_level_emphasis: float
    small_zone_high_grey_level_emphasis: float
    large_zone_low_grey_level_emphasis: float
    large_zone_high_grey_level_emphasis: float
    grey_level_non_uniformity: float
    normalised_grey_level_non_uniformity: float
    zone_size_non_uniformity: float
    normalised_zone_size_non_uniformity: float
    zone_percentage: float
    grey_level_variance: float
    zone_size_variance: float
    zone_size_entropy: float

    TAGS = {
        "small_zone_emphasis": "szm_sze",
        "large_zone_emphasis": "szm_lze",
        "low_grey_level_zone_emphasis": "szm_lgze",
        "high_grey_level_zone_emphasis": "szm_hgze",
        "small_zone_low_grey_level_emphasis": "szm_szlge",
        "small_zone_high_grey_level_emphasis": "szm_szhge",
        "large_zone_low_grey_level_emphasis": "szm_lzlge",
        "large_zone_high_grey_level_emphasis": "szm_lzhge",
        "grey_level_non_uniformity": "szm_glnu",
        "normalised_grey_level_non_uniformity": "szm_glnu_norm",
        "zone_size_non_uniformity": "szm_zsnu",
        "normalised_zone_size_non_uniformity": "szm_zsnu_norm",
        "zone_percentage": "szm_z_perc",
        "grey_level_variance": "szm_gl_var",
        "zone_size_variance": "szm_zs_var",
        "zone_size_entropy": "szm_zs_entr",
    }

    def to_dict(self, aggregation: ZoneAggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def glszm_features(disc: Discretisation, aggregation: ZoneAggregation = "3D") -> GLSZMFeatures:
    """Compute the IBSI grey level size zone features (IBSI 3.8).

    A zone is a maximal connected set of ROI voxels sharing a grey level, under
    full connectivity: 8-connected within a slice, 26-connected in 3D.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`ZONE_AGGREGATIONS`.

    Returns
    -------
    GLSZMFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or contains no zones.

    """
    return _zone_features(
        disc,
        aggregation,
        lambda levels, mask: _size_zones(levels, mask, disc.n_levels),
        GLSZMFeatures,
    )


# --- Grey level distance zone matrix (IBSI 3.9) ----------------------------


@dataclass(frozen=True)
class GLDZMFeatures:
    """The 16 IBSI grey level distance zone features (IBSI 3.9)."""

    small_distance_emphasis: float
    large_distance_emphasis: float
    low_grey_level_zone_emphasis: float
    high_grey_level_zone_emphasis: float
    small_distance_low_grey_level_emphasis: float
    small_distance_high_grey_level_emphasis: float
    large_distance_low_grey_level_emphasis: float
    large_distance_high_grey_level_emphasis: float
    grey_level_non_uniformity: float
    normalised_grey_level_non_uniformity: float
    zone_distance_non_uniformity: float
    normalised_zone_distance_non_uniformity: float
    zone_percentage: float
    grey_level_variance: float
    zone_distance_variance: float
    zone_distance_entropy: float

    TAGS = {
        "small_distance_emphasis": "dzm_sde",
        "large_distance_emphasis": "dzm_lde",
        "low_grey_level_zone_emphasis": "dzm_lgze",
        "high_grey_level_zone_emphasis": "dzm_hgze",
        "small_distance_low_grey_level_emphasis": "dzm_sdlge",
        "small_distance_high_grey_level_emphasis": "dzm_sdhge",
        "large_distance_low_grey_level_emphasis": "dzm_ldlge",
        "large_distance_high_grey_level_emphasis": "dzm_ldhge",
        "grey_level_non_uniformity": "dzm_glnu",
        "normalised_grey_level_non_uniformity": "dzm_glnu_norm",
        "zone_distance_non_uniformity": "dzm_zdnu",
        "normalised_zone_distance_non_uniformity": "dzm_zdnu_norm",
        "zone_percentage": "dzm_z_perc",
        "grey_level_variance": "dzm_gl_var",
        "zone_distance_variance": "dzm_zd_var",
        "zone_distance_entropy": "dzm_zd_entr",
    }

    def to_dict(self, aggregation: ZoneAggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def gldzm_features(disc: Discretisation, aggregation: ZoneAggregation = "3D") -> GLDZMFeatures:
    """Compute the IBSI grey level distance zone features (IBSI 3.9).

    Zones are those of :func:`glszm_features`; each carries the smallest number
    of face-connected steps from any of its voxels to the outside of the ROI, so
    a zone touching the ROI border has distance 1.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`ZONE_AGGREGATIONS`.

    Returns
    -------
    GLDZMFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or contains no zones.

    """
    return _zone_features(
        disc,
        aggregation,
        lambda levels, mask: _distance_zones(levels, mask, disc.n_levels),
        GLDZMFeatures,
    )


# --- Neighbourhood grey tone difference matrix (IBSI 3.10) -----------------


@dataclass(frozen=True)
class _NGTDMCounts:
    """Per-level voxel counts ``n_i`` and absolute neighbourhood differences ``s_i``."""

    counts: np.ndarray
    differences: np.ndarray
    n_valid: int

    def __add__(self, other: _NGTDMCounts) -> _NGTDMCounts:
        return _NGTDMCounts(
            self.counts + other.counts,
            self.differences + other.differences,
            self.n_valid + other.n_valid,
        )


def _ngtdm_counts(levels: np.ndarray, mask: np.ndarray, n_levels: int) -> _NGTDMCounts:
    """Accumulate ``n_i`` and ``s_i`` over the ROI voxels that have a neighbour."""
    total = np.zeros(levels.shape, dtype=np.float64)
    n_neighbours = np.zeros(levels.shape, dtype=np.int64)
    for offset in _offsets(levels.ndim):
        neighbour_levels, neighbour_mask = _neighbour_views(levels, mask, offset)
        total += np.where(neighbour_mask, neighbour_levels, 0)
        n_neighbours += neighbour_mask

    valid = mask & (n_neighbours > 0)
    if not valid.any():
        raise FeatureError(
            "no ROI voxel has a neighbour inside the ROI; the neighbourhood grey tone "
            "difference matrix is empty."
        )
    centre = levels[valid].astype(np.float64)
    average = total[valid] / n_neighbours[valid]

    counts = np.bincount(levels[valid], minlength=n_levels + 1)[1:].astype(np.float64)
    differences = np.bincount(
        levels[valid], weights=np.abs(centre - average), minlength=n_levels + 1
    )[1:]
    return _NGTDMCounts(counts, differences, int(valid.sum()))


@dataclass(frozen=True)
class NGTDMFeatures:
    """The 5 IBSI neighbourhood grey tone difference features (IBSI 3.10)."""

    coarseness: float
    contrast: float
    busyness: float
    complexity: float
    strength: float

    TAGS = {
        "coarseness": "ngt_coarseness",
        "contrast": "ngt_contrast",
        "busyness": "ngt_busyness",
        "complexity": "ngt_complexity",
        "strength": "ngt_strength",
    }

    def to_dict(self, aggregation: ZoneAggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def _ngtdm_from_counts(data: _NGTDMCounts) -> dict[str, float]:
    occupied = data.counts > 0
    if int(occupied.sum()) < 2:
        raise FeatureError(
            "the neighbourhood grey tone difference matrix holds a single grey level; "
            "contrast, busyness, complexity and strength are undefined."
        )

    i = np.flatnonzero(occupied).astype(np.float64) + 1.0
    p = data.counts[occupied] / data.n_valid
    s = data.differences[occupied]
    n_gl = i.size

    total_difference = float(s.sum())
    weighted_difference = float(np.sum(p * s))
    if total_difference <= 0.0:
        raise FeatureError(
            "every ROI voxel equals its neighbourhood average; coarseness is infinite "
            "and strength is undefined."
        )

    difference = i[:, None] - i[None, :]
    sum_p = p[:, None] + p[None, :]

    busyness_denominator = float(np.sum(np.abs(i[:, None] * p[:, None] - i[None, :] * p[None, :])))
    if busyness_denominator <= 0.0:
        raise FeatureError("busyness is undefined: the grey level weights are degenerate.")

    contrast_term = float(np.sum(p[:, None] * p[None, :] * difference**2)) / (n_gl * (n_gl - 1))
    complexity = float(
        np.sum(np.abs(difference) * (p[:, None] * s[:, None] + p[None, :] * s[None, :]) / sum_p)
    )

    return {
        "coarseness": 1.0 / weighted_difference,
        "contrast": contrast_term * total_difference / data.n_valid,
        "busyness": weighted_difference / busyness_denominator,
        "complexity": complexity / data.n_valid,
        "strength": float(np.sum(sum_p * difference**2)) / total_difference,
    }


def ngtdm_features(disc: Discretisation, aggregation: ZoneAggregation = "3D") -> NGTDMFeatures:
    """Compute the IBSI neighbourhood grey tone difference features (IBSI 3.10).

    For every ROI voxel that has at least one ROI neighbour at Chebyshev
    distance 1, the absolute difference between its grey level and the mean grey
    level of those neighbours is accumulated per grey level.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`ZONE_AGGREGATIONS`.

    Returns
    -------
    NGTDMFeatures

    Raises
    ------
    FeatureError
        If the ROI holds a single grey level, or every voxel equals its
        neighbourhood average, which leaves these features undefined.

    """
    if aggregation not in ZONE_AGGREGATIONS:
        raise FeatureError(f"aggregation must be one of {ZONE_AGGREGATIONS}; got {aggregation!r}.")

    if aggregation == "3D":
        per_matrix = [_ngtdm_from_counts(_ngtdm_counts(disc.levels, disc.mask, disc.n_levels))]
    else:
        per_slice = [
            _ngtdm_counts(disc.levels[z], disc.mask[z], disc.n_levels)
            for z in range(disc.levels.shape[0])
            if disc.mask[z].any()
        ]
        if not per_slice:
            raise FeatureError("ROI mask is empty; no features can be computed.")
        if aggregation == "2D":
            per_matrix = [_ngtdm_from_counts(counts) for counts in per_slice]
        else:
            merged = per_slice[0]
            for counts in per_slice[1:]:
                merged = merged + counts
            per_matrix = [_ngtdm_from_counts(merged)]

    return NGTDMFeatures(**_mean_of_features(per_matrix))


# --- Neighbouring grey level dependence matrix (IBSI 3.11) -----------------


@dataclass(frozen=True)
class NGLDMFeatures:
    """The 17 IBSI neighbouring grey level dependence features (IBSI 3.11)."""

    low_dependence_emphasis: float
    high_dependence_emphasis: float
    low_grey_level_count_emphasis: float
    high_grey_level_count_emphasis: float
    low_dependence_low_grey_level_emphasis: float
    low_dependence_high_grey_level_emphasis: float
    high_dependence_low_grey_level_emphasis: float
    high_dependence_high_grey_level_emphasis: float
    grey_level_non_uniformity: float
    normalised_grey_level_non_uniformity: float
    dependence_count_non_uniformity: float
    normalised_dependence_count_non_uniformity: float
    dependence_count_percentage: float
    grey_level_variance: float
    dependence_count_variance: float
    dependence_count_entropy: float
    dependence_count_energy: float

    TAGS = {
        "low_dependence_emphasis": "ngl_lde",
        "high_dependence_emphasis": "ngl_hde",
        "low_grey_level_count_emphasis": "ngl_lgce",
        "high_grey_level_count_emphasis": "ngl_hgce",
        "low_dependence_low_grey_level_emphasis": "ngl_ldlge",
        "low_dependence_high_grey_level_emphasis": "ngl_ldhge",
        "high_dependence_low_grey_level_emphasis": "ngl_hdlge",
        "high_dependence_high_grey_level_emphasis": "ngl_hdhge",
        "grey_level_non_uniformity": "ngl_glnu",
        "normalised_grey_level_non_uniformity": "ngl_glnu_norm",
        "dependence_count_non_uniformity": "ngl_dcnu",
        "normalised_dependence_count_non_uniformity": "ngl_dcnu_norm",
        "dependence_count_percentage": "ngl_dc_perc",
        "grey_level_variance": "ngl_gl_var",
        "dependence_count_variance": "ngl_dc_var",
        "dependence_count_entropy": "ngl_dc_entr",
        "dependence_count_energy": "ngl_dc_energy",
    }

    def to_dict(self, aggregation: ZoneAggregation | None = None) -> dict[str, float]:
        """Map IBSI feature tags to values, optionally suffixed by ``aggregation``."""
        suffix = f"_{aggregation}" if aggregation else ""
        return {self.TAGS[f.name] + suffix: getattr(self, f.name) for f in fields(self)}


def _ngldm_matrix(
    levels: np.ndarray, mask: np.ndarray, n_levels: int, alpha: int, max_dependence: int
) -> _Matrix:
    """Dependence counts per (grey level, dependence).

    The dependence of a voxel is the number of its Chebyshev-distance-1
    neighbours inside the ROI whose grey level differs by at most ``alpha``,
    **plus one for the voxel itself**.  Counting the centre keeps the dependence
    at or above 1, which is what makes low dependence emphasis -- a sum of
    ``1 / j**2`` -- finite for a voxel that resembles none of its neighbours.
    """
    dependence = np.ones(levels.shape, dtype=np.int64)
    for offset in _offsets(levels.ndim):
        neighbour_levels, neighbour_mask = _neighbour_views(levels, mask, offset)
        dependence += neighbour_mask & (np.abs(neighbour_levels - levels) <= alpha)

    i = levels[mask].astype(np.int64) - 1
    j = dependence[mask] - 1
    flat = np.bincount(i * max_dependence + j, minlength=n_levels * max_dependence)
    return _Matrix(flat.reshape(n_levels, max_dependence).astype(np.float64), int(mask.sum()))


def _ngldm_from_matrix(counts: np.ndarray, n_voxels: int) -> dict[str, float]:
    n_levels, max_dependence = counts.shape
    n_total = float(counts.sum())

    i = np.arange(1, n_levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, max_dependence + 1, dtype=np.float64)[None, :]

    p = counts / n_total
    marginal_level = counts.sum(axis=1)
    marginal_dependence = counts.sum(axis=0)

    mu_i = float(np.sum(i * p))
    mu_j = float(np.sum(j * p))

    return {
        "low_dependence_emphasis": float(np.sum(counts / j**2) / n_total),
        "high_dependence_emphasis": float(np.sum(counts * j**2) / n_total),
        "low_grey_level_count_emphasis": float(np.sum(counts / i**2) / n_total),
        "high_grey_level_count_emphasis": float(np.sum(counts * i**2) / n_total),
        "low_dependence_low_grey_level_emphasis": float(np.sum(counts / (i**2 * j**2)) / n_total),
        "low_dependence_high_grey_level_emphasis": float(np.sum(counts * i**2 / j**2) / n_total),
        "high_dependence_low_grey_level_emphasis": float(np.sum(counts * j**2 / i**2) / n_total),
        "high_dependence_high_grey_level_emphasis": float(np.sum(counts * i**2 * j**2) / n_total),
        "grey_level_non_uniformity": float(np.sum(marginal_level**2) / n_total),
        "normalised_grey_level_non_uniformity": float(np.sum(marginal_level**2) / n_total**2),
        "dependence_count_non_uniformity": float(np.sum(marginal_dependence**2) / n_total),
        "normalised_dependence_count_non_uniformity": float(
            np.sum(marginal_dependence**2) / n_total**2
        ),
        "dependence_count_percentage": float(n_total / n_voxels),
        "grey_level_variance": float(np.sum((i - mu_i) ** 2 * p)),
        "dependence_count_variance": float(np.sum((j - mu_j) ** 2 * p)),
        "dependence_count_entropy": _entropy(p),
        "dependence_count_energy": float(np.sum(p**2)),
    }


def ngldm_features(
    disc: Discretisation, aggregation: ZoneAggregation = "3D", alpha: int = 0
) -> NGLDMFeatures:
    """Compute the IBSI neighbouring grey level dependence features (IBSI 3.11).

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.
    aggregation:
        One of :data:`ZONE_AGGREGATIONS`.
    alpha:
        Coarseness parameter: two voxels are *dependent* when their grey levels
        differ by at most ``alpha``.  IBSI's benchmarks use ``0``.

    Returns
    -------
    NGLDMFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty, or ``alpha`` is negative.

    """
    if aggregation not in ZONE_AGGREGATIONS:
        raise FeatureError(f"aggregation must be one of {ZONE_AGGREGATIONS}; got {aggregation!r}.")
    if alpha < 0:
        raise FeatureError(f"alpha must be non-negative; got {alpha}.")

    # Dependence counts the centre voxel, so it spans 1 .. 1 + n_neighbours.
    max_dependence = 1 + len(_offsets(3 if aggregation == "3D" else 2))

    if aggregation == "3D":
        matrices = [_ngldm_matrix(disc.levels, disc.mask, disc.n_levels, alpha, max_dependence)]
    else:
        per_slice = [
            _ngldm_matrix(disc.levels[z], disc.mask[z], disc.n_levels, alpha, max_dependence)
            for z in range(disc.levels.shape[0])
            if disc.mask[z].any()
        ]
        if not per_slice:
            raise FeatureError("ROI mask is empty; no features can be computed.")
        if aggregation == "2D":
            matrices = per_slice
        else:
            merged = per_slice[0]
            for matrix in per_slice[1:]:
                merged = merged + matrix
            matrices = [merged]

    per_matrix = [_ngldm_from_matrix(m.counts, m.n_voxels) for m in matrices]
    return NGLDMFeatures(**_mean_of_features(per_matrix))


# ---------------------------------------------------------------------------
# Morphology, local intensity, intensity-volume histogram (IBSI 3.1, 3.2, 3.5)
# ---------------------------------------------------------------------------
#
# Unlike the texture families, these are computed on the *raw* intensities and
# depend on the physical voxel spacing.  Morphology works from a triangular
# surface mesh of the ROI (marching cubes) as IBSI prescribes, so its volume and
# area differ from naive voxel counting.


def _validate_triplet_features(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise FeatureError(
            f"spacing must have exactly 3 elements (dz, dy, dx); got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise FeatureError(f"spacing must be finite and strictly positive; got {tuple(arr)}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class _Mesh:
    """A triangular surface mesh of the ROI in physical (z, y, x) mm coordinates."""

    vertices: np.ndarray  # (n_vertices, 3)
    faces: np.ndarray  # (n_faces, 3) int indices into vertices

    @property
    def volume(self) -> float:
        """Enclosed volume via the signed-tetrahedron sum (IBSI 3.1.1)."""
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        return float(abs(signed.sum()))

    @property
    def area(self) -> float:
        """Surface area as the sum of triangle areas (IBSI 3.1.3)."""
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _roi_mesh(mask: np.ndarray, spacing: tuple[float, float, float]) -> _Mesh:
    """Marching-cubes surface of the ROI, at the voxel-centre iso-level.

    The mask is padded so that a ROI touching the array edge is still closed,
    and vertices are scaled to millimetres.
    """
    from skimage.measure import marching_cubes

    padded = np.pad(mask.astype(np.float64), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    except (RuntimeError, ValueError) as exc:
        raise FeatureError(f"could not build a surface mesh of the ROI: {exc}") from exc
    # Undo the pad: subtract one voxel in each direction, in mm.
    verts = verts - np.asarray(spacing, dtype=float)
    return _Mesh(verts, faces.astype(np.int64))


@dataclass(frozen=True)
class MorphologyFeatures:
    """The 25 standardised IBSI morphological features (IBSI 3.1).

    The four OMBB/MVEE density features are not standardised by IBSI and are not
    computed here.
    """

    volume: float
    approximate_volume: float
    surface_area: float
    surface_to_volume_ratio: float
    compactness_1: float
    compactness_2: float
    spherical_disproportion: float
    sphericity: float
    asphericity: float
    centre_of_mass_shift: float
    maximum_3d_diameter: float
    major_axis_length: float
    minor_axis_length: float
    least_axis_length: float
    elongation: float
    flatness: float
    volume_density_aabb: float
    area_density_aabb: float
    volume_density_aee: float
    area_density_aee: float
    volume_density_convex_hull: float
    area_density_convex_hull: float
    integrated_intensity: float
    morans_i: float
    gearys_c: float

    TAGS = {
        "volume": "morph_volume",
        "approximate_volume": "morph_vol_approx",
        "surface_area": "morph_area_mesh",
        "surface_to_volume_ratio": "morph_av",
        "compactness_1": "morph_comp_1",
        "compactness_2": "morph_comp_2",
        "spherical_disproportion": "morph_sph_dispr",
        "sphericity": "morph_sphericity",
        "asphericity": "morph_asphericity",
        "centre_of_mass_shift": "morph_com",
        "maximum_3d_diameter": "morph_diam",
        "major_axis_length": "morph_pca_maj_axis",
        "minor_axis_length": "morph_pca_min_axis",
        "least_axis_length": "morph_pca_least_axis",
        "elongation": "morph_pca_elongation",
        "flatness": "morph_pca_flatness",
        "volume_density_aabb": "morph_vol_dens_aabb",
        "area_density_aabb": "morph_area_dens_aabb",
        "volume_density_aee": "morph_vol_dens_aee",
        "area_density_aee": "morph_area_dens_aee",
        "volume_density_convex_hull": "morph_vol_dens_conv_hull",
        "area_density_convex_hull": "morph_area_dens_conv_hull",
        "integrated_intensity": "morph_integ_int",
        "morans_i": "morph_moran_i",
        "gearys_c": "morph_geary_c",
    }

    def to_dict(self) -> dict[str, float]:
        """Map IBSI feature tags to values."""
        return {self.TAGS[f.name]: getattr(self, f.name) for f in fields(self)}


def _ellipsoid_surface_area(a: float, b: float, c: float) -> float:
    """Surface area of an ellipsoid with semi-axes a >= b >= c (IBSI 3.1.20).

    Uses the Legendre-series approximation that IBSI specifies, truncated at 20
    terms, which is far more than enough for three significant digits.
    """
    from scipy.special import eval_legendre

    if c <= 0.0:
        return 0.0
    # IBSI parameterisation.
    alpha = np.sqrt(1.0 - (b / a) ** 2) if a > 0 else 0.0
    beta = np.sqrt(1.0 - (c / a) ** 2) if a > 0 else 0.0

    if alpha == 0.0 and beta == 0.0:
        return 4.0 * np.pi * a * a

    total = 0.0
    for nu in range(20):
        if alpha * beta > 0:
            legendre = eval_legendre(nu, (alpha**2 + beta**2) / (2.0 * alpha * beta))
        else:
            legendre = 1.0 if nu == 0 else 0.0
        total += (alpha * beta) ** nu / (1.0 - 4.0 * nu**2) * legendre
    return float(4.0 * np.pi * a * b * total)


def morphology_features(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MorphologyFeatures:
    """Compute the 25 standardised IBSI morphological features (IBSI 3.1).

    The ROI shape is meshed with marching cubes; volume and surface area come
    from that mesh, so they differ from voxel counting.  Spatial statistics
    (Moran's I, Geary's C) and intensity-weighted features use the raw
    intensities.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI mask.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.

    Returns
    -------
    MorphologyFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty, non-finite, too small or too thin to mesh, planar
        (no extent along one axis), or of constant intensity.  The last case
        leaves Moran's I and Geary's C undefined (a ``0 / 0``); because those are
        part of the returned set, the whole call raises rather than return two
        ``nan`` fields.  Pass a textured ROI, or the mask itself as the volume,
        if only the shape features are wanted.

    """
    _check_roi(mask, volume)
    spacing = _validate_triplet_features(spacing)
    intensities = volume[mask].astype(np.float64)
    if not np.all(np.isfinite(intensities)):
        raise FeatureError("ROI contains non-finite intensities.")

    voxel_volume = float(np.prod(spacing))
    n_voxels = int(mask.sum())

    mesh = _roi_mesh(mask, spacing)
    V = mesh.volume
    A = mesh.area
    if V <= 0.0 or A <= 0.0:
        raise FeatureError(
            "the ROI mesh is degenerate (zero volume or area); the ROI is too small "
            "or too thin to have a surface."
        )

    approx_volume = n_voxels * voxel_volume

    # Voxel-centre coordinates in mm, (z, y, x).
    coords = np.argwhere(mask).astype(np.float64) * np.asarray(spacing)

    # Principal component axes from the voxel-position covariance.  IBSI uses the
    # sample covariance (Bessel's 1 / (n - 1) correction).
    centred = coords - coords.mean(axis=0)
    cov = centred.T @ centred / (n_voxels - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]  # major, minor, least
    lam_major, lam_minor, lam_least = (float(e) for e in eigenvalues)
    major = 4.0 * np.sqrt(lam_major)
    minor = 4.0 * np.sqrt(lam_minor)
    least = 4.0 * np.sqrt(max(lam_least, 0.0))

    # Centre of mass shift: geometric vs intensity-weighted centroid.
    com_geom = coords.mean(axis=0)
    weight_sum = float(intensities.sum())
    if weight_sum == 0.0:
        raise FeatureError("integrated intensity is zero; centre of mass shift is undefined.")
    com_gl = (coords * intensities[:, None]).sum(axis=0) / weight_sum
    com_shift = float(np.linalg.norm(com_geom - com_gl))

    # Maximum 3D diameter over mesh vertices, via the convex hull to stay cheap.
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist

    hull = ConvexHull(mesh.vertices)
    hull_points = mesh.vertices[np.unique(hull.simplices)]
    max_diameter = float(pdist(hull_points).max())

    # Densities.
    aabb_extent = coords.max(axis=0) - coords.min(axis=0) + np.asarray(spacing)
    v_aabb = float(np.prod(aabb_extent))
    a_aabb = 2.0 * float(
        aabb_extent[0] * aabb_extent[1]
        + aabb_extent[1] * aabb_extent[2]
        + aabb_extent[0] * aabb_extent[2]
    )

    if least <= 0.0:
        raise FeatureError(
            "the ROI is planar (its least principal axis has zero length); the "
            "enclosing-ellipsoid density and related features are degenerate. A "
            "morphological ROI needs extent in all three dimensions."
        )

    semi = np.array([major, minor, least]) / 2.0
    v_aee = 4.0 / 3.0 * np.pi * float(np.prod(semi))
    a_aee = _ellipsoid_surface_area(*semi)

    v_convex = float(hull.volume)
    a_convex = float(hull.area)

    return MorphologyFeatures(
        volume=V,
        approximate_volume=approx_volume,
        surface_area=A,
        surface_to_volume_ratio=A / V,
        compactness_1=V / (np.sqrt(np.pi) * A**1.5),
        compactness_2=36.0 * np.pi * V**2 / A**3,
        spherical_disproportion=A / (36.0 * np.pi * V**2) ** (1.0 / 3.0),
        sphericity=(36.0 * np.pi * V**2) ** (1.0 / 3.0) / A,
        asphericity=(A**3 / (36.0 * np.pi * V**2)) ** (1.0 / 3.0) - 1.0,
        centre_of_mass_shift=com_shift,
        maximum_3d_diameter=max_diameter,
        major_axis_length=major,
        minor_axis_length=minor,
        least_axis_length=least,
        elongation=np.sqrt(lam_minor / lam_major),
        flatness=np.sqrt(max(lam_least, 0.0) / lam_major),
        volume_density_aabb=V / v_aabb,
        area_density_aabb=A / a_aabb,
        volume_density_aee=V / v_aee,
        area_density_aee=A / a_aee,
        volume_density_convex_hull=V / v_convex,
        area_density_convex_hull=A / a_convex,
        integrated_intensity=V * float(intensities.mean()),
        morans_i=_morans_i(coords, intensities),
        gearys_c=_gearys_c(coords, intensities),
    )


def _spatial_weights(coords: np.ndarray) -> np.ndarray:
    """Inverse Euclidean distance weights between every pair of ROI voxels."""
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    np.fill_diagonal(weights, 0.0)
    return weights


def _morans_i(coords: np.ndarray, intensities: np.ndarray) -> float:
    n = intensities.size
    weights = _spatial_weights(coords)
    centred = intensities - intensities.mean()
    denom = float(np.sum(centred**2))
    w_sum = float(weights.sum())
    if denom == 0.0 or w_sum == 0.0:
        raise FeatureError("Moran's I is undefined for a constant ROI.")
    numer = float(centred @ weights @ centred)
    return n / w_sum * numer / denom


def _gearys_c(coords: np.ndarray, intensities: np.ndarray) -> float:
    n = intensities.size
    weights = _spatial_weights(coords)
    centred = intensities - intensities.mean()
    denom = float(np.sum(centred**2))
    w_sum = float(weights.sum())
    if denom == 0.0 or w_sum == 0.0:
        raise FeatureError("Geary's C is undefined for a constant ROI.")
    diff_sq = (intensities[:, None] - intensities[None, :]) ** 2
    numer = float(np.sum(weights * diff_sq))
    return (n - 1) / (2.0 * w_sum) * numer / denom


# --- Local intensity (IBSI 3.2) --------------------------------------------


@dataclass(frozen=True)
class LocalIntensityFeatures:
    """The 2 IBSI local intensity features (IBSI 3.2)."""

    local_intensity_peak: float
    global_intensity_peak: float

    TAGS = {
        "local_intensity_peak": "loc_peak_loc",
        "global_intensity_peak": "loc_peak_glob",
    }

    def to_dict(self) -> dict[str, float]:
        """Map IBSI feature tags to values."""
        return {self.TAGS[f.name]: getattr(self, f.name) for f in fields(self)}


def _peak_neighbourhood_offsets(spacing: tuple[float, float, float]) -> np.ndarray:
    """Voxel offsets whose centres lie within the 1 cm^3-equivalent sphere.

    The sphere has radius ``(3 / (4 pi))^(1/3)`` cm ~= 6.2 mm, so that its
    volume is 1 cm^3.
    """
    radius_mm = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0
    reach = [int(np.floor(radius_mm / s)) for s in spacing]
    grid = np.stack(
        np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    distances = np.linalg.norm(grid * np.asarray(spacing), axis=1)
    return grid[distances <= radius_mm + 1e-9]


def local_intensity_features(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> LocalIntensityFeatures:
    """Compute the IBSI local intensity peak features (IBSI 3.2).

    Around each voxel, the mean intensity over a 1 cm^3 sphere (whether or not
    its neighbours are in the ROI) defines a local peak.  The *local* peak is
    that mean at the ROI's maximum-intensity voxels; the *global* peak is the
    largest such mean over the whole ROI.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI mask.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``.

    Returns
    -------
    LocalIntensityFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or non-finite.

    """
    _check_roi(mask, volume)
    spacing = _validate_triplet_features(spacing)
    volume = volume.astype(np.float64)
    if not np.all(np.isfinite(volume[mask])):
        raise FeatureError("ROI contains non-finite intensities.")

    offsets = _peak_neighbourhood_offsets(spacing)
    shape = volume.shape

    def sphere_mean(z: int, y: int, x: int) -> float:
        pts = offsets + np.array([z, y, x])
        inside = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
        pts = pts[inside]
        return float(volume[pts[:, 0], pts[:, 1], pts[:, 2]].mean())

    roi_coords = np.argwhere(mask)
    roi_values = volume[mask]

    # Local peak: at the ROI voxels of maximum intensity.
    peak_voxels = roi_coords[roi_values == roi_values.max()]
    local_peak = max(sphere_mean(*coord) for coord in peak_voxels)

    # Global peak: the maximum sphere mean over all ROI voxels.
    global_peak = max(sphere_mean(*coord) for coord in roi_coords)

    return LocalIntensityFeatures(
        local_intensity_peak=local_peak,
        global_intensity_peak=global_peak,
    )


# --- Intensity-volume histogram (IBSI 3.5) ---------------------------------


@dataclass(frozen=True)
class IVHFeatures:
    """The 6 standardised IBSI intensity-volume histogram features (IBSI 3.5).

    The area under the IVH curve is not standardised by IBSI and is omitted.
    """

    volume_fraction_at_10pct: float
    volume_fraction_at_90pct: float
    intensity_at_10pct_volume: float
    intensity_at_90pct_volume: float
    volume_fraction_difference: float
    intensity_difference: float

    TAGS = {
        "volume_fraction_at_10pct": "ivh_v10",
        "volume_fraction_at_90pct": "ivh_v90",
        "intensity_at_10pct_volume": "ivh_i10",
        "intensity_at_90pct_volume": "ivh_i90",
        "volume_fraction_difference": "ivh_diff_v10_v90",
        "intensity_difference": "ivh_diff_i10_i90",
    }

    def to_dict(self) -> dict[str, float]:
        """Map IBSI feature tags to values."""
        return {self.TAGS[f.name]: getattr(self, f.name) for f in fields(self)}


def intensity_volume_histogram(disc: Discretisation) -> IVHFeatures:
    """Compute the IBSI intensity-volume histogram features (IBSI 3.5).

    For each discrete grey level ``g``, the intensity fraction is
    ``(g - 1) / (Ng - 1)`` and the volume fraction is the fraction of ROI voxels
    with level at least ``g``.  ``V_x`` is the largest volume fraction whose
    intensity fraction is at least ``x``; ``I_x`` is the least grey level whose
    volume fraction is at most ``x``.

    Parameters
    ----------
    disc:
        A discretised ROI from :func:`discretise`.  For a calibrated scale the
        IBSI digital phantom is discretised with a fixed bin size of 1.

    Returns
    -------
    IVHFeatures

    Raises
    ------
    FeatureError
        If the ROI is empty or spans a single grey level.

    """
    levels = disc.roi_levels.astype(np.int64)
    if levels.size == 0:
        raise FeatureError("ROI mask is empty; no features can be computed.")
    if disc.n_levels < 2:
        raise FeatureError("the intensity-volume histogram needs at least 2 grey levels.")

    n = levels.size
    grey = np.arange(1, disc.n_levels + 1)
    intensity_fraction = (grey - 1) / (disc.n_levels - 1)

    # Volume fraction nu[g] = fraction of voxels with level >= g.
    at_least = np.array([float(np.count_nonzero(levels >= g)) / n for g in grey])

    def volume_at_intensity(x: float) -> float:
        eligible = at_least[intensity_fraction >= x - 1e-12]
        return float(eligible.max()) if eligible.size else 0.0

    def intensity_at_volume(x: float) -> float:
        eligible = grey[at_least <= x + 1e-12]
        return float(eligible.min()) if eligible.size else float(grey[-1])

    v10 = volume_at_intensity(0.10)
    v90 = volume_at_intensity(0.90)
    i10 = intensity_at_volume(0.10)
    i90 = intensity_at_volume(0.90)

    return IVHFeatures(
        volume_fraction_at_10pct=v10,
        volume_fraction_at_90pct=v90,
        intensity_at_10pct_volume=i10,
        intensity_at_90pct_volume=i90,
        volume_fraction_difference=v10 - v90,
        intensity_difference=i10 - i90,
    )


# ---------------------------------------------------------------------------
# Convenience: extract a full feature vector in one call
# ---------------------------------------------------------------------------


def extract_features(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    bin_width: float = 25.0,
    include_morphology: bool = True,
) -> dict[str, float]:
    """Compute a full IBSI feature vector, keyed by feature tag.

    A single call over every implemented family, using the 3D-merged texture
    aggregations (``3D_comb`` for GLCM/GLRLM, ``3D`` for the zone, neighbourhood
    and dependence families).  This is the vector the stability atlas and the GUI
    consume.

    Parameters
    ----------
    volume:
        3D intensity volume ``(z, y, x)``.
    mask:
        Boolean ROI mask.
    spacing:
        Voxel size in millimetres, ``(dz, dy, dx)``; used by the morphology and
        local-intensity families.
    bin_width:
        Fixed bin size for the texture and intensity-volume-histogram families.
    include_morphology:
        Whether to add the morphology, local-intensity and intensity-volume
        histogram families.  These need at least two grey levels and a
        three-dimensional, non-constant ROI; set ``False`` to skip them for
        degenerate ROIs.

    Returns
    -------
    dict
        Feature tag -> value, e.g. ``{"stat_mean": ..., "cm_contrast_3D_comb": ...}``.

    Raises
    ------
    FeatureError
        If the ROI is degenerate for a requested family (propagated from the
        underlying feature functions).

    """
    disc = discretise(volume, mask, method="fbs", bin_width=bin_width)

    features: dict[str, float] = {}
    features.update(intensity_statistics(volume, mask).to_dict())
    features.update(intensity_histogram(disc).to_dict())
    features.update(glcm_features(disc, "3D_comb").to_dict("3D_comb"))
    features.update(glrlm_features(disc, "3D_comb").to_dict("3D_comb"))
    features.update(glszm_features(disc, "3D").to_dict("3D"))
    features.update(gldzm_features(disc, "3D").to_dict("3D"))
    features.update(ngtdm_features(disc, "3D").to_dict("3D"))
    features.update(ngldm_features(disc, "3D").to_dict("3D"))

    if include_morphology:
        features.update(morphology_features(volume, mask, spacing).to_dict())
        features.update(local_intensity_features(volume, mask, spacing).to_dict())
        features.update(intensity_volume_histogram(disc).to_dict())

    return features
