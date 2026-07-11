"""Physics-based feature normalisation.

Rather than harmonise features after the fact against a reference cohort (ComBat
and its relatives), this module corrects a feature using the *known* response of
that feature to a measurable acquisition descriptor -- noise level, blur, voxel
size -- learned on the synthetic phantom where the descriptor is known exactly.

The workflow is:

1. **Calibrate.**  On a phantom, sweep one acquisition descriptor (e.g. noise
   standard deviation) and record how a feature responds.  Fit a simple,
   interpretable model to that calibration curve.
2. **Correct.**  Given a feature measured under a known descriptor value, map it
   back to the value it would have taken at a reference descriptor by inverting
   the model.
3. **Report residual.**  A feature whose calibration curve the model cannot
   describe is flagged by its fit residual, rather than silently "corrected".
   Normalisation you cannot trust is worse than none.

Two model families are provided, both invertible in closed form and chosen to
match the physics observed in :mod:`rphantom.acquisition`:

* :class:`LinearResponse` -- ``feature = a + b * descriptor``.  Fits, for
  instance, intensity variance against noise *variance* (``var = var0 + sigma^2``).
* :class:`PowerResponse` -- ``feature = a + b * descriptor**p`` for a fixed ``p``.
  A power of 2 fits features that grow with noise amplitude.

Everything is deterministic and, as elsewhere, degeneracy raises a
:class:`NormalizationError` rather than returning ``nan``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LinearResponse",
    "PowerResponse",
    "NormalizationError",
    "CalibrationCurve",
    "calibrate_response",
    "normalise_feature",
]


class NormalizationError(ValueError):
    """Raised when a calibration or correction is not well defined.

    Subclasses :class:`ValueError`.  Raised instead of returning ``nan`` when a
    curve cannot be fit, a model cannot be inverted, or inputs are malformed.
    """


@dataclass(frozen=True)
class _Response:
    """Base for invertible one-descriptor response models.

    ``coefficients`` are ``(a, b)`` such that ``feature = a + b * basis(descriptor)``.
    """

    a: float
    b: float

    def predict(self, descriptor: np.ndarray) -> np.ndarray:
        """Predicted feature value(s) at ``descriptor``."""
        raise NotImplementedError

    def invert(self, feature: float, descriptor: float, reference: float = 0.0) -> float:
        """Map ``feature`` observed at ``descriptor`` back to ``reference``."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearResponse(_Response):
    """``feature = a + b * descriptor``.

    Correction to a reference descriptor ``d_ref`` is a rigid shift along the
    fitted line, independent of the feature's own scatter.
    """

    def predict(self, descriptor: np.ndarray) -> np.ndarray:
        """Predicted feature value(s) at ``descriptor``."""
        return self.a + self.b * np.asarray(descriptor, dtype=np.float64)

    def invert(self, feature: float, descriptor: float, reference: float = 0.0) -> float:
        """Map ``feature`` observed at ``descriptor`` back to ``reference``.

        The model says the descriptor added ``b * (descriptor - reference)``;
        subtract it.
        """
        return float(feature - self.b * (descriptor - reference))


@dataclass(frozen=True)
class PowerResponse(_Response):
    """``feature = a + b * descriptor ** power`` for a fixed, known ``power``."""

    power: float = 2.0

    def predict(self, descriptor: np.ndarray) -> np.ndarray:
        """Predicted feature value(s) at ``descriptor``."""
        return self.a + self.b * np.asarray(descriptor, dtype=np.float64) ** self.power

    def invert(self, feature: float, descriptor: float, reference: float = 0.0) -> float:
        """Remove the modelled descriptor term and restore it at ``reference``."""
        return float(feature - self.b * (descriptor**self.power - reference**self.power))


@dataclass(frozen=True)
class CalibrationCurve:
    """A fitted response model plus the goodness of fit that qualifies its use.

    Attributes
    ----------
    tag:
        The feature this curve describes.
    model:
        The fitted :class:`LinearResponse` or :class:`PowerResponse`.
    descriptor_name:
        Name of the acquisition descriptor, e.g. ``"noise_sigma"``.
    r_squared:
        Coefficient of determination of the fit; 1 is perfect.
    residual_std:
        Standard deviation of the fit residual, in the feature's own units.
    descriptors, values:
        The calibration samples the curve was fit to.

    """

    tag: str
    model: _Response
    descriptor_name: str
    r_squared: float
    residual_std: float
    descriptors: np.ndarray
    values: np.ndarray

    @property
    def is_trustworthy(self) -> bool:
        """Whether the model explains the calibration curve well (``R^2 >= 0.9``)."""
        return self.r_squared >= 0.9

    def normalise(self, feature: float, descriptor: float, reference: float = 0.0) -> float:
        """Correct ``feature`` measured at ``descriptor`` to ``reference``."""
        return self.model.invert(feature, descriptor, reference)


def _basis(descriptors: np.ndarray, model: str, power: float) -> np.ndarray:
    if model == "linear":
        return descriptors
    if model == "power":
        return descriptors**power
    raise NormalizationError(f"model must be 'linear' or 'power'; got {model!r}.")


def calibrate_response(
    descriptors: np.ndarray,
    values: np.ndarray,
    *,
    tag: str = "",
    descriptor_name: str = "descriptor",
    model: str = "linear",
    power: float = 2.0,
) -> CalibrationCurve:
    """Fit a feature's response to one acquisition descriptor.

    Parameters
    ----------
    descriptors:
        The descriptor values swept, e.g. an array of noise sigmas.
    values:
        The feature value measured at each descriptor, same length.
    tag:
        The feature tag, carried through for reporting.
    descriptor_name:
        Human-readable descriptor name.
    model:
        ``"linear"`` (``a + b*d``) or ``"power"`` (``a + b*d**power``).
    power:
        The exponent for ``model="power"``.

    Returns
    -------
    CalibrationCurve

    Raises
    ------
    NormalizationError
        On mismatched lengths, too few samples, non-finite data, or a descriptor
        column with no variation (the slope would be undefined).

    """
    d = np.asarray(descriptors, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if d.shape != v.shape or d.ndim != 1:
        raise NormalizationError(
            f"descriptors and values must be 1D of equal length; got {d.shape} and {v.shape}."
        )
    if d.size < 3:
        raise NormalizationError("calibration needs at least 3 samples.")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(v))):
        raise NormalizationError("descriptors and values must be finite.")

    basis = _basis(d, model, power)
    if float(np.var(basis)) == 0.0:
        raise NormalizationError(
            "the descriptor does not vary; a response slope cannot be estimated."
        )

    # Ordinary least squares for [a, b] in v = a + b * basis.
    design = np.column_stack([np.ones_like(basis), basis])
    (a, b), *_ = np.linalg.lstsq(design, v, rcond=None)

    predicted = a + b * basis
    residual = v - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    fitted: _Response
    if model == "linear":
        fitted = LinearResponse(a=float(a), b=float(b))
    else:
        fitted = PowerResponse(a=float(a), b=float(b), power=float(power))

    return CalibrationCurve(
        tag=tag,
        model=fitted,
        descriptor_name=descriptor_name,
        r_squared=float(r_squared),
        residual_std=float(residual.std()),
        descriptors=d,
        values=v,
    )


def normalise_feature(
    curve: CalibrationCurve,
    feature: float,
    descriptor: float,
    *,
    reference: float = 0.0,
    require_trustworthy: bool = True,
) -> float:
    """Correct one measured feature back to a reference acquisition.

    Parameters
    ----------
    curve:
        A :class:`CalibrationCurve` for this feature and descriptor.
    feature:
        The feature value observed under the acquisition.
    descriptor:
        The descriptor value at which it was observed (e.g. the actual noise
        sigma of that scan).
    reference:
        The descriptor value to normalise to; ``0`` means the ideal (noiseless,
        unblurred) acquisition.
    require_trustworthy:
        If ``True`` (the default), refuse to normalise a feature whose
        calibration fit is poor (``R^2 < 0.9``), raising instead of returning a
        correction that the physics does not support.

    Returns
    -------
    float
        The normalised feature value.

    Raises
    ------
    NormalizationError
        If inputs are non-finite, or the calibration is untrustworthy and
        ``require_trustworthy`` is set.

    """
    if not (np.isfinite(feature) and np.isfinite(descriptor) and np.isfinite(reference)):
        raise NormalizationError("feature, descriptor and reference must be finite.")
    if require_trustworthy and not curve.is_trustworthy:
        raise NormalizationError(
            f"refusing to normalise {curve.tag or 'this feature'}: the calibration fit is poor "
            f"(R^2={curve.r_squared:.3f} < 0.9). This feature's response to "
            f"{curve.descriptor_name} is not captured by the model; pass "
            "require_trustworthy=False to override."
        )
    return curve.normalise(feature, descriptor, reference)
