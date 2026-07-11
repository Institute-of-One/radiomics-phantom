"""Feature-stability atlas: rating each feature's reproducibility under acquisition change.

Given one texture observed under many simulated scanners, a *stable* feature
returns nearly the same value every time; an *unstable* one swings with dose,
kernel or voxel size.  This module quantifies that with two agreement statistics,
implemented from first principles (``numpy`` only) and validated against
``pingouin`` in the test suite:

* **ICC(2,1)** -- the two-way random-effects, single-measurement, absolute-agreement
  intraclass correlation.  Across a table of ``targets x conditions`` it measures
  how much of the feature's variance is the real texture-to-texture signal versus
  acquisition noise.  1 is perfect reproducibility, 0 none.
* **CCC** -- Lin's concordance correlation coefficient, the agreement between a
  degraded acquisition's features and the reference acquisition's, across targets.

The atlas builder ties the three modules together: it sweeps a set of phantoms
(the *targets*) across a set of acquisition settings (the *conditions*), extracts
every IBSI feature, and reduces each feature to its reliability.

Determinism and honesty are preserved throughout: acquisitions are seeded, and a
statistic that is undefined (a constant column, too few targets) raises a
:class:`StabilityError` rather than returning ``nan``.

References
----------
Shrout & Fleiss, "Intraclass correlations: uses in assessing rater reliability",
Psychological Bulletin 86(2), 1979.
Lin, "A concordance correlation coefficient to evaluate reproducibility",
Biometrics 45(1), 1989.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rphantom.acquisition import simulate_acquisition
from rphantom.features import extract_features
from rphantom.phantom import Phantom

__all__ = [
    "ICCResult",
    "ConcordanceResult",
    "FeatureReliability",
    "StabilityAtlas",
    "StabilityError",
    "concordance_correlation",
    "intraclass_correlation",
    "build_stability_atlas",
]


class StabilityError(ValueError):
    """Raised when a reliability statistic cannot be computed.

    Subclasses :class:`ValueError`.  Raised instead of returning ``nan`` when a
    measurement table is malformed or degenerate (a single condition, a constant
    feature, mismatched lengths).
    """


@dataclass(frozen=True)
class ICCResult:
    """A two-way random-effects intraclass correlation and its ANOVA parts.

    Attributes
    ----------
    icc:
        ICC(2,1), absolute agreement, single measurement.
    n_targets, n_conditions:
        The measurement table's dimensions.
    ms_between_targets, ms_between_conditions, ms_error:
        The ANOVA mean squares the estimate is built from.

    """

    icc: float
    n_targets: int
    n_conditions: int
    ms_between_targets: float
    ms_between_conditions: float
    ms_error: float


def intraclass_correlation(measurements: np.ndarray) -> ICCResult:
    """ICC(2,1): two-way random effects, single rater, absolute agreement.

    Parameters
    ----------
    measurements:
        A ``(n_targets, n_conditions)`` array: one row per target (e.g. a phantom
        texture), one column per condition (e.g. an acquisition setting).

    Returns
    -------
    ICCResult

    Raises
    ------
    StabilityError
        If there are fewer than two targets or two conditions, or the table is
        not finite.

    Notes
    -----
    With row (target) mean squares ``MSR``, column (condition) mean squares
    ``MSC`` and residual ``MSE`` from the two-way ANOVA of the table,

    ``ICC(2,1) = (MSR - MSE) / (MSR + (k-1) MSE + (k / n)(MSC - MSE))``

    for ``n`` targets and ``k`` conditions.  The estimate can fall slightly below
    zero when the condition effect dominates; that is a real (if unhelpful)
    outcome and is returned as-is rather than clipped.

    """
    table = np.asarray(measurements, dtype=np.float64)
    if table.ndim != 2:
        raise StabilityError(f"measurements must be a 2D table; got ndim={table.ndim}.")
    n, k = table.shape
    if n < 2 or k < 2:
        raise StabilityError(f"ICC needs at least 2 targets and 2 conditions; got {n} x {k}.")
    if not np.all(np.isfinite(table)):
        raise StabilityError("measurements must be finite.")

    grand_mean = float(table.mean())
    row_means = table.mean(axis=1)
    col_means = table.mean(axis=0)

    ss_rows = k * float(np.sum((row_means - grand_mean) ** 2))
    ss_cols = n * float(np.sum((col_means - grand_mean) ** 2))
    ss_total = float(np.sum((table - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    if denominator == 0.0:
        raise StabilityError(
            "ICC is undefined: the total variance of the measurement table is zero "
            "(every value is identical)."
        )
    icc = (ms_rows - ms_error) / denominator

    return ICCResult(
        icc=float(icc),
        n_targets=int(n),
        n_conditions=int(k),
        ms_between_targets=float(ms_rows),
        ms_between_conditions=float(ms_cols),
        ms_error=float(ms_error),
    )


@dataclass(frozen=True)
class ConcordanceResult:
    """Lin's concordance correlation coefficient and its parts."""

    ccc: float
    pearson_r: float
    bias_correction: float
    n: int


def concordance_correlation(reference: np.ndarray, measured: np.ndarray) -> ConcordanceResult:
    """Lin's concordance correlation coefficient between paired vectors.

    Parameters
    ----------
    reference:
        Feature values under the reference acquisition, one per target.
    measured:
        Feature values under the degraded acquisition, same targets, same order.

    Returns
    -------
    ConcordanceResult
        ``ccc`` in ``[-1, 1]``; ``pearson_r`` (precision) and ``bias_correction``
        (accuracy) factor such that ``ccc = pearson_r * bias_correction``.

    Raises
    ------
    StabilityError
        If the vectors differ in length, have fewer than two elements, are not
        finite, or both have zero variance (leaving the CCC undefined).

    """
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise StabilityError(
            f"reference and measured must be equal-length 1D vectors; got {x.shape}, {y.shape}."
        )
    if x.size < 2:
        raise StabilityError("concordance needs at least 2 paired values.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise StabilityError("reference and measured must be finite.")

    mx, my = float(x.mean()), float(y.mean())
    vx = float(np.mean((x - mx) ** 2))
    vy = float(np.mean((y - my) ** 2))
    sxy = float(np.mean((x - mx) * (y - my)))

    denominator = vx + vy + (mx - my) ** 2
    if denominator == 0.0:
        raise StabilityError(
            "concordance is undefined: reference and measured are both constant and equal."
        )
    ccc = 2.0 * sxy / denominator

    if vx == 0.0 or vy == 0.0:
        pearson = 0.0
    else:
        pearson = sxy / np.sqrt(vx * vy)
    bias_correction = ccc / pearson if pearson != 0.0 else 0.0

    return ConcordanceResult(
        ccc=float(ccc),
        pearson_r=float(pearson),
        bias_correction=float(bias_correction),
        n=int(x.size),
    )


@dataclass(frozen=True)
class FeatureReliability:
    """The stability of one feature across acquisition conditions.

    Attributes
    ----------
    tag:
        IBSI feature tag.
    icc:
        ICC(2,1) across all conditions.
    ccc_min, ccc_mean:
        The worst-case and mean concordance of a degraded condition against the
        reference condition, over the non-reference conditions.
    reference_value:
        The feature's mean over targets under the reference condition, for scale.

    """

    tag: str
    icc: float
    ccc_min: float
    ccc_mean: float
    reference_value: float


@dataclass(frozen=True)
class StabilityAtlas:
    """A table of per-feature reliabilities, plus the raw measurement matrices.

    Attributes
    ----------
    reliabilities:
        Feature tag -> :class:`FeatureReliability`.
    matrices:
        Feature tag -> ``(n_targets, n_conditions)`` measurement array.
    condition_labels:
        A human-readable label per condition; index 0 is the reference.
    n_targets, n_conditions:
        The sweep dimensions.

    """

    reliabilities: dict[str, FeatureReliability]
    matrices: dict[str, np.ndarray]
    condition_labels: list[str]
    n_targets: int
    n_conditions: int

    def ranked(self, *, by: str = "icc", ascending: bool = True) -> list[FeatureReliability]:
        """Feature reliabilities sorted by ``icc`` or ``ccc_min``.

        Ascending order (the default) puts the *least* stable features first,
        which is what an atlas reader usually wants to see.
        """
        if by not in ("icc", "ccc_min", "ccc_mean"):
            raise StabilityError(f"cannot rank by {by!r}; use 'icc', 'ccc_min' or 'ccc_mean'.")
        return sorted(
            self.reliabilities.values(), key=lambda r: getattr(r, by), reverse=not ascending
        )


def _feature_matrix(
    feature_rows: list[dict[str, float]], n_targets: int, n_conditions: int
) -> dict[str, np.ndarray]:
    """Reshape a flat list of feature dicts into per-tag ``(targets, conditions)`` arrays.

    ``feature_rows`` is ordered target-major: target 0 under every condition,
    then target 1, and so on.  Only tags present for *every* cell are kept.
    """
    common = set(feature_rows[0])
    for row in feature_rows[1:]:
        common &= set(row)

    matrices: dict[str, np.ndarray] = {}
    for tag in sorted(common):
        flat = np.array([row[tag] for row in feature_rows], dtype=np.float64)
        matrices[tag] = flat.reshape(n_targets, n_conditions)
    return matrices


def build_stability_atlas(
    phantoms: list[Phantom],
    conditions: list[dict[str, Any]],
    *,
    condition_labels: list[str] | None = None,
    bin_width: float = 25.0,
    include_morphology: bool = True,
    feature_extractor: Callable[..., dict[str, float]] | None = None,
) -> StabilityAtlas:
    """Sweep phantoms across acquisition conditions and rate every feature.

    Each phantom is a *target*; each acquisition condition is a *condition*.  For
    every (target, condition) the phantom is degraded with
    :func:`~rphantom.acquisition.simulate_acquisition` and its full feature vector
    extracted.  Per feature, the atlas reports ICC(2,1) across all conditions and
    the concordance of each degraded condition against the reference (condition 0).

    Parameters
    ----------
    phantoms:
        At least two phantoms, ideally spanning a range of textures so the ICC has
        genuine target-to-target signal to detect.
    conditions:
        At least two acquisition settings, as keyword-argument dicts for
        :func:`~rphantom.acquisition.simulate_acquisition`.  Condition 0 is the
        reference the concordances are measured against -- make it the mildest
        (e.g. ``{}`` for a noiseless, unblurred acquisition).
    condition_labels:
        Optional labels; defaults to ``"cond0" .. "condN"``.
    bin_width:
        Fixed bin size passed to feature extraction.
    include_morphology:
        Whether to include the morphology/local/IVH families.
    feature_extractor:
        Override for the feature function, taking ``(volume, mask, spacing,
        bin_width=..., include_morphology=...)``.  Defaults to
        :func:`rphantom.features.extract_features`.

    Returns
    -------
    StabilityAtlas

    Raises
    ------
    StabilityError
        If fewer than two phantoms or conditions are given, or if the swept
        feature tables are too degenerate for any statistic.

    """
    if len(phantoms) < 2:
        raise StabilityError(f"need at least 2 phantoms; got {len(phantoms)}.")
    if len(conditions) < 2:
        raise StabilityError(f"need at least 2 conditions; got {len(conditions)}.")
    if condition_labels is not None and len(condition_labels) != len(conditions):
        raise StabilityError("condition_labels must match the number of conditions.")

    extractor = feature_extractor or extract_features
    labels = condition_labels or [f"cond{i}" for i in range(len(conditions))]
    n_targets, n_conditions = len(phantoms), len(conditions)

    feature_rows: list[dict[str, float]] = []
    for phantom in phantoms:
        for condition in conditions:
            acquisition = simulate_acquisition(phantom, **condition)
            feature_rows.append(
                extractor(
                    acquisition.volume,
                    acquisition.mask,
                    acquisition.spacing,
                    bin_width=bin_width,
                    include_morphology=include_morphology,
                )
            )

    matrices = _feature_matrix(feature_rows, n_targets, n_conditions)
    if not matrices:
        raise StabilityError(
            "no feature was computable across every (phantom, condition) cell; "
            "the sweep produced no common feature tags."
        )

    reliabilities: dict[str, FeatureReliability] = {}
    for tag, matrix in matrices.items():
        reference_column = matrix[:, 0]

        # A feature that never varies between targets carries no reliability
        # signal; report it explicitly rather than dividing by zero.
        if float(np.var(matrix)) == 0.0:
            reliabilities[tag] = FeatureReliability(
                tag=tag,
                icc=1.0,
                ccc_min=1.0,
                ccc_mean=1.0,
                reference_value=float(reference_column.mean()),
            )
            continue

        try:
            icc = intraclass_correlation(matrix).icc
        except StabilityError:
            icc = float("nan")

        cccs = []
        for j in range(1, n_conditions):
            try:
                cccs.append(concordance_correlation(reference_column, matrix[:, j]).ccc)
            except StabilityError:
                cccs.append(float("nan"))

        reliabilities[tag] = FeatureReliability(
            tag=tag,
            icc=float(icc),
            ccc_min=float(np.nanmin(cccs)) if cccs else float("nan"),
            ccc_mean=float(np.nanmean(cccs)) if cccs else float("nan"),
            reference_value=float(reference_column.mean()),
        )

    return StabilityAtlas(
        reliabilities=reliabilities,
        matrices=matrices,
        condition_labels=list(labels),
        n_targets=n_targets,
        n_conditions=n_conditions,
    )
