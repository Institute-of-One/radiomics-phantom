"""Tests for :mod:`rphantom.acquisition`.

The contract: each degradation is a deterministic, physically sensible function
of its parameters, nothing fails silently, and -- because this module exists to
feed the stability atlas -- the degradations move IBSI features in the direction
physics predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom import discretise, generate_texture_phantom, glcm_features, intensity_statistics
from rphantom.acquisition import (
    Acquisition,
    AcquisitionError,
    add_noise,
    apply_blur,
    apply_slice_profile,
    quantise,
    resample,
    simulate_acquisition,
)
from rphantom.phantom import measure_correlation_length


@pytest.fixture
def phantom():
    return generate_texture_phantom(size=(48, 48, 48), corr_length=6.0, seed=0)


@pytest.fixture
def flat():
    return np.full((40, 40, 40), 40.0), (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [5.0, 10.0, 25.0])
def test_white_noise_reproduces_sigma_and_zero_mean(flat, sigma) -> None:
    volume, spacing = flat
    noisy = add_noise(volume, spacing, sigma, np.random.default_rng(1))

    residual = noisy - volume
    assert residual.std() == pytest.approx(sigma, rel=1e-6)
    assert residual.mean() == pytest.approx(0.0, abs=1e-9)


def test_correlated_noise_has_the_requested_correlation_length(flat) -> None:
    volume, spacing = flat
    noisy = add_noise(volume, spacing, 20.0, np.random.default_rng(2), correlation_length_mm=3.0)

    residual = noisy - volume
    assert residual.std() == pytest.approx(20.0, rel=1e-6)
    assert measure_correlation_length(residual, spacing, axis=0) == pytest.approx(3.0, rel=0.2)


def test_noise_is_deterministic_and_seed_dependent(flat) -> None:
    volume, spacing = flat
    a = add_noise(volume, spacing, 10.0, np.random.default_rng(3))
    b = add_noise(volume, spacing, 10.0, np.random.default_rng(3))
    c = add_noise(volume, spacing, 10.0, np.random.default_rng(4))

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_zero_sigma_is_a_no_op(flat) -> None:
    volume, spacing = flat
    assert np.array_equal(add_noise(volume, spacing, 0.0, np.random.default_rng(0)), volume)


# --------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------


def test_blur_preserves_mean_and_lowers_variance(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    sharp_std = volume.std()

    for fwhm in (2.0, 4.0):
        blurred = apply_blur(volume, phantom.spacing, fwhm)
        assert blurred.mean() == pytest.approx(volume.mean(), rel=1e-3)
        assert blurred.std() < sharp_std


def test_blur_increases_the_correlation_length(phantom) -> None:
    sharp = measure_correlation_length(phantom.volume.astype(np.float64), phantom.spacing, 0)
    blurred_volume = apply_blur(phantom.volume.astype(np.float64), phantom.spacing, 4.0)
    blurred = measure_correlation_length(blurred_volume, phantom.spacing, 0)
    assert blurred > sharp


def test_slice_profile_blurs_z_far_more_than_in_plane(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    out = apply_slice_profile(volume, phantom.spacing, 4.0)

    z_before = measure_correlation_length(volume, phantom.spacing, 0)
    z_after = measure_correlation_length(out, phantom.spacing, 0)
    x_before = measure_correlation_length(volume, phantom.spacing, 2)
    x_after = measure_correlation_length(out, phantom.spacing, 2)

    # z correlation grows; the in-plane measurement barely moves (it shifts only
    # because it averages over the now-smoother z axis).
    assert z_after - z_before > 0.5
    assert abs(x_after - x_before) < 0.25 * (z_after - z_before)


def test_zero_fwhm_blur_is_a_no_op(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    assert np.array_equal(apply_blur(volume, phantom.spacing, 0.0), volume)


def test_anisotropic_blur_accepts_a_triple(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    out = apply_blur(volume, phantom.spacing, (4.0, 0.0, 0.0))
    assert measure_correlation_length(out, phantom.spacing, 0) > measure_correlation_length(
        volume, phantom.spacing, 0
    )


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new_spacing", "expected_shape"),
    [((2.0, 2.0, 2.0), (16, 16, 16)), ((0.5, 0.5, 0.5), (64, 64, 64))],
)
def test_resample_changes_the_grid_by_the_spacing_ratio(new_spacing, expected_shape) -> None:
    phantom = generate_texture_phantom(size=(32, 32, 32), seed=0)
    volume, mask, spacing = resample(
        phantom.volume.astype(np.float64), phantom.mask, phantom.spacing, new_spacing
    )

    assert volume.shape == expected_shape
    assert mask.shape == expected_shape
    assert mask.dtype == np.bool_
    assert mask.any()
    assert spacing == new_spacing


def test_finer_resampling_preserves_the_roi_volume_better() -> None:
    """Coarsening a lesion loses boundary voxels -- a real radiomics instability.

    The finer the target grid, the closer the resampled ROI volume stays to the
    original, which is exactly the sensitivity the stability atlas will chart.
    """
    phantom = generate_texture_phantom(size=(32, 32, 32), lesion_radii_mm=(8.0, 8.0, 8.0), seed=0)
    fine = phantom.mask.sum() * float(np.prod(phantom.spacing))

    ratios = []
    for target in ((1.5, 1.5, 1.5), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)):
        _, mask, spacing = resample(
            phantom.volume.astype(np.float64), phantom.mask, phantom.spacing, target
        )
        ratios.append(mask.sum() * float(np.prod(spacing)) / fine)

    assert ratios == sorted(ratios, reverse=True)  # finer grid -> ratio nearer 1
    assert 0.6 < ratios[0] < 1.2  # the finest target stays within a sensible band


def test_resample_round_trip_is_close(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    down, mask, mid = resample(volume, phantom.mask, phantom.spacing, (2.0, 2.0, 2.0))
    up, _, back = resample(down, mask, mid, phantom.spacing)

    assert back == phantom.spacing
    # A coarse round trip loses fine detail but keeps the gross intensity level.
    assert up.mean() == pytest.approx(volume.mean(), rel=0.05)


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------


def test_quantise_places_values_on_the_grid(phantom) -> None:
    q = quantise(phantom.volume.astype(np.float64), 10.0)
    assert np.allclose(q / 10.0, np.round(q / 10.0))
    assert np.abs(q - phantom.volume).max() <= 5.0 + 1e-6


def test_zero_step_is_a_no_op(phantom) -> None:
    volume = phantom.volume.astype(np.float64)
    assert np.array_equal(quantise(volume, 0.0), volume)


# --------------------------------------------------------------------------
# simulate_acquisition
# --------------------------------------------------------------------------


def test_identity_acquisition_returns_the_phantom(phantom) -> None:
    acq = simulate_acquisition(phantom, seed=0)
    assert isinstance(acq, Acquisition)
    assert np.array_equal(acq.volume, phantom.volume)
    assert np.array_equal(acq.mask, phantom.mask)
    assert acq.spacing == phantom.spacing


def test_acquisition_is_bit_identical_for_the_same_seed(phantom) -> None:
    kwargs = dict(psf_fwhm_mm=2.0, noise_sigma=15.0, noise_correlation_mm=1.5, dose=2.0, seed=7)
    a = simulate_acquisition(phantom, **kwargs)
    b = simulate_acquisition(phantom, **kwargs)

    assert np.array_equal(a.volume, b.volume)
    assert a.settings == b.settings


def test_acquisition_differs_by_seed_only_through_noise(phantom) -> None:
    a = simulate_acquisition(phantom, noise_sigma=15.0, seed=1)
    b = simulate_acquisition(phantom, noise_sigma=15.0, seed=2)
    noiseless_1 = simulate_acquisition(phantom, noise_sigma=0.0, seed=1)
    noiseless_2 = simulate_acquisition(phantom, noise_sigma=0.0, seed=2)

    assert not np.array_equal(a.volume, b.volume)
    assert np.array_equal(noiseless_1.volume, noiseless_2.volume)


def test_dose_scales_noise_as_inverse_sqrt(phantom) -> None:
    baseline = phantom.volume.astype(np.float64)
    stds = []
    for dose in (1.0, 4.0, 16.0):
        acq = simulate_acquisition(phantom, noise_sigma=20.0, dose=dose, seed=3)
        stds.append((acq.volume.astype(np.float64) - baseline).std())

    assert stds[0] == pytest.approx(20.0, rel=0.02)
    assert stds[1] == pytest.approx(10.0, rel=0.02)
    assert stds[2] == pytest.approx(5.0, rel=0.02)


def test_settings_and_ground_truth_are_recorded(phantom) -> None:
    acq = simulate_acquisition(
        phantom,
        psf_fwhm_mm=1.5,
        slice_fwhm_mm=3.0,
        new_spacing=(2.0, 1.0, 1.0),
        noise_sigma=12.0,
        dose=2.0,
        quantise_step=5.0,
        seed=4,
    )
    assert acq.spacing == (2.0, 1.0, 1.0)
    assert acq.shape == (24, 48, 48)
    assert acq.settings["effective_noise_sigma"] == pytest.approx(12.0 / np.sqrt(2.0))
    assert acq.settings["new_spacing"] == (2.0, 1.0, 1.0)
    assert acq.ground_truth == phantom.ground_truth
    assert np.allclose(acq.volume / 5.0, np.round(acq.volume / 5.0))


# --------------------------------------------------------------------------
# Integration: degradations move IBSI features as physics predicts
# --------------------------------------------------------------------------


def _contrast(acq) -> float:
    full = np.ones(acq.volume.shape, dtype=bool)
    disc = discretise(acq.volume, full, method="fbs", bin_width=25.0)
    return glcm_features(disc, "3D_comb").contrast


def test_noise_raises_glcm_contrast(phantom) -> None:
    contrasts = [
        _contrast(simulate_acquisition(phantom, noise_sigma=s, seed=1)) for s in (0.0, 10.0, 30.0)
    ]
    assert contrasts == sorted(contrasts)
    assert contrasts[-1] > contrasts[0]


def test_blur_lowers_glcm_contrast(phantom) -> None:
    contrasts = [
        _contrast(simulate_acquisition(phantom, psf_fwhm_mm=f, seed=1)) for f in (0.0, 3.0, 6.0)
    ]
    assert contrasts == sorted(contrasts, reverse=True)


def test_noise_raises_intensity_variance(phantom) -> None:
    full = np.ones(phantom.volume.shape, dtype=bool)
    quiet = simulate_acquisition(phantom, noise_sigma=0.0, seed=1)
    loud = simulate_acquisition(phantom, noise_sigma=40.0, seed=1)

    var_quiet = intensity_statistics(quiet.volume, full).variance
    var_loud = intensity_statistics(loud.volume, full).variance
    # Independent noise adds in variance: ~ sigma^2 on top of the texture.
    assert var_loud == pytest.approx(var_quiet + 40.0**2, rel=0.1)


# --------------------------------------------------------------------------
# Explicit failure, never silent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda p: add_noise(p.volume, p.spacing, -1.0, np.random.default_rng(0)), "sigma must"),
        (
            lambda p: add_noise(
                p.volume, p.spacing, 1.0, np.random.default_rng(0), correlation_length_mm=-1.0
            ),
            "correlation_length_mm must",
        ),
        (lambda p: apply_blur(p.volume, p.spacing, -1.0), "must be finite and non-negative"),
        (lambda p: apply_blur(p.volume, (1.0, 0.0, 1.0), 1.0), "spacing must"),
        (lambda p: quantise(p.volume, -1.0), "step must"),
        (lambda p: simulate_acquisition(p, dose=0.0), "dose must"),
        (lambda p: simulate_acquisition(p, seed=1.5), "seed must be an int"),
        (lambda p: resample(p.volume, p.mask, p.spacing, (0.0, 1.0, 1.0)), "spacing must"),
        (lambda p: resample(p.volume, p.mask, p.spacing, p.spacing, order=7), "order must"),
    ],
)
def test_malformed_input_raises(call, match) -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=0)
    with pytest.raises(AcquisitionError, match=match):
        call(phantom)


def test_resample_rejects_an_empty_mask() -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), lesion=False, seed=0)
    with pytest.raises(AcquisitionError, match="mask is empty"):
        resample(phantom.volume, phantom.mask, phantom.spacing, (2.0, 2.0, 2.0))


def test_simulate_rejects_a_non_phantom() -> None:
    with pytest.raises(AcquisitionError, match="phantom must be a Phantom"):
        simulate_acquisition(np.zeros((8, 8, 8)))  # type: ignore[arg-type]
