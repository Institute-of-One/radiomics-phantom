"""Tests for :mod:`rphantom.normalize`.

The headline claim -- that a feature's acquisition-induced drift can be undone by
inverting its known physical response -- is checked end to end: intensity
variance really does follow ``var0 + sigma^2`` under added noise, and the
normaliser really does collapse the noisy measurements back onto ``var0``.
Analytic curves pin the algebra; a poorly-fit feature is refused, not corrected.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom import extract_features, generate_texture_phantom, simulate_acquisition
from rphantom.normalize import (
    LinearResponse,
    NormalizationError,
    PowerResponse,
    calibrate_response,
    normalise_feature,
)

# --------------------------------------------------------------------------
# The response models, on exact synthetic curves
# --------------------------------------------------------------------------


def test_linear_response_recovers_known_coefficients() -> None:
    d = np.linspace(0.0, 10.0, 11)
    v = 3.0 + 2.5 * d
    curve = calibrate_response(d, v, model="linear")

    assert isinstance(curve.model, LinearResponse)
    assert curve.model.a == pytest.approx(3.0)
    assert curve.model.b == pytest.approx(2.5)
    assert curve.r_squared == pytest.approx(1.0)
    assert curve.is_trustworthy


def test_power_response_recovers_known_coefficients() -> None:
    d = np.linspace(0.0, 6.0, 13)
    v = 100.0 + 0.5 * d**2
    curve = calibrate_response(d, v, model="power", power=2.0)

    assert isinstance(curve.model, PowerResponse)
    assert curve.model.a == pytest.approx(100.0)
    assert curve.model.b == pytest.approx(0.5)
    assert curve.r_squared == pytest.approx(1.0)


def test_linear_inversion_shifts_along_the_line() -> None:
    curve = calibrate_response(np.arange(5.0), 1.0 + 2.0 * np.arange(5.0), model="linear")
    # A value observed at descriptor 4 maps back to descriptor 0 by removing 2*4.
    assert curve.normalise(feature=9.0, descriptor=4.0, reference=0.0) == pytest.approx(1.0)


def test_power_inversion_removes_the_power_term() -> None:
    curve = calibrate_response(np.arange(6.0), 10.0 + 3.0 * np.arange(6.0) ** 2, model="power")
    # Observed at descriptor 5 (value 10 + 3*25 = 85) -> back to 10 at reference 0.
    assert curve.normalise(feature=85.0, descriptor=5.0, reference=0.0) == pytest.approx(10.0)


def test_normalise_to_a_nonzero_reference() -> None:
    curve = calibrate_response(np.arange(5.0), 2.0 * np.arange(5.0), model="linear")
    # From descriptor 4 to reference 1: remove b*(4-1) = 2*3 = 6, from value 8 -> 2.
    assert curve.normalise(feature=8.0, descriptor=4.0, reference=1.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# End to end: real acquisition physics
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def variance_calibration():
    phantom = generate_texture_phantom(size=(28, 28, 28), hu_sd=25.0, lesion=False, seed=0)
    roi = np.ones(phantom.shape, dtype=bool)
    sigmas = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    values = np.array(
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
        sigmas, values, tag="stat_var", descriptor_name="noise_sigma", model="power", power=2.0
    )
    return curve, sigmas, values


def test_intensity_variance_follows_var0_plus_sigma_squared(variance_calibration) -> None:
    curve, _, values = variance_calibration
    # var = var0 + sigma^2, so the fitted slope is ~1 and the intercept is var0.
    assert curve.model.b == pytest.approx(1.0, abs=0.02)
    assert curve.model.a == pytest.approx(values[0], rel=0.01)
    assert curve.r_squared > 0.999


def test_normalisation_collapses_the_noise_spread(variance_calibration) -> None:
    curve, sigmas, values = variance_calibration
    true_reference = values[0]

    normalised = [
        normalise_feature(curve, v, s, reference=0.0) for v, s in zip(values, sigmas, strict=True)
    ]

    # Raw values span hundreds of HU^2; normalised values sit on var0 within ~1.
    assert max(normalised) - min(normalised) < 1.0
    assert all(abs(n - true_reference) < 1.0 for n in normalised)
    assert np.std(normalised) < 0.1 * np.std(values)


# --------------------------------------------------------------------------
# Refusing to normalise what the physics does not support
# --------------------------------------------------------------------------


def test_poorly_fit_feature_is_refused() -> None:
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, 30.0, 8)
    noise = rng.normal(size=8) * 5.0  # no relationship to d
    curve = calibrate_response(d, noise, tag="junk", model="power")

    assert not curve.is_trustworthy
    with pytest.raises(NormalizationError, match="calibration fit is poor"):
        normalise_feature(curve, 5.0, 20.0)


def test_override_allows_normalising_an_untrustworthy_curve() -> None:
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, 30.0, 8)
    curve = calibrate_response(d, rng.normal(size=8), model="linear")

    # It returns a number rather than raising; the caller has taken responsibility.
    assert np.isfinite(normalise_feature(curve, 5.0, 20.0, require_trustworthy=False))


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("descriptors", "values", "match"),
    [
        (np.arange(3.0), np.arange(4.0), "equal length"),
        (np.arange(2.0), np.arange(2.0), "at least 3 samples"),
        (np.full(4, np.nan), np.arange(4.0), "must be finite"),
        (np.zeros(4), np.arange(4.0), "does not vary"),
    ],
)
def test_calibrate_rejects_bad_input(descriptors, values, match) -> None:
    with pytest.raises(NormalizationError, match=match):
        calibrate_response(descriptors, values)


def test_unknown_model_raises() -> None:
    with pytest.raises(NormalizationError, match="model must be"):
        calibrate_response(np.arange(4.0), np.arange(4.0), model="quadratic")


def test_non_finite_normalisation_input_raises() -> None:
    curve = calibrate_response(np.arange(4.0), np.arange(4.0), model="linear")
    with pytest.raises(NormalizationError, match="must be finite"):
        normalise_feature(curve, np.nan, 1.0)
