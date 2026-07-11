"""Tests for :mod:`rphantom.phantom`.

The contract under test: phantoms are *deterministic* (a seed pins every bit),
*well-formed* (shape, dtype, finiteness, non-empty mask), *self-describing*
(ground truth echoes the request), and *honest* (the texture really has the
correlation length that was asked for).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rphantom.phantom import (
    Phantom,
    generate_texture_phantom,
    measure_correlation_length,
)

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_is_bit_identical() -> None:
    kwargs = dict(size=(24, 28, 32), spacing=(1.5, 0.8, 0.8), corr_length=5.0, seed=1234)
    a = generate_texture_phantom(**kwargs)
    b = generate_texture_phantom(**kwargs)

    assert np.array_equal(a.volume, b.volume)
    assert np.array_equal(a.mask, b.mask)
    assert a.ground_truth == b.ground_truth
    assert a.seed == b.seed == 1234


def test_different_seed_gives_different_texture_but_same_mask() -> None:
    a = generate_texture_phantom(size=(32, 32, 32), seed=0)
    b = generate_texture_phantom(size=(32, 32, 32), seed=1)

    assert not np.array_equal(a.volume, b.volume)
    # Geometry is not stochastic: only the texture is.
    assert np.array_equal(a.mask, b.mask)


def test_lesion_field_does_not_perturb_background_draw() -> None:
    """The background is drawn first, so it must not depend on lesion options."""
    with_lesion = generate_texture_phantom(size=(32, 32, 32), lesion=True, seed=5)
    without = generate_texture_phantom(size=(32, 32, 32), lesion=False, seed=5)

    outside = ~with_lesion.mask
    assert np.array_equal(with_lesion.volume[outside], without.volume[outside])


# --------------------------------------------------------------------------
# Structure and well-formedness
# --------------------------------------------------------------------------


def test_shape_dtype_spacing_and_finiteness() -> None:
    ph = generate_texture_phantom(size=(16, 20, 24), spacing=(2.0, 1.0, 0.5), seed=3)

    assert isinstance(ph, Phantom)
    assert ph.volume.shape == (16, 20, 24)
    assert ph.mask.shape == (16, 20, 24)
    assert ph.volume.dtype == np.float32
    assert ph.mask.dtype == np.bool_
    assert ph.spacing == (2.0, 1.0, 0.5)
    assert ph.shape == (16, 20, 24)
    assert ph.voxel_volume_mm3 == pytest.approx(1.0)
    assert np.all(np.isfinite(ph.volume))


def test_mask_is_non_empty_and_matches_ellipsoid_volume() -> None:
    radii = (8.0, 6.0, 6.0)
    ph = generate_texture_phantom(
        size=(48, 48, 48), spacing=(1.0, 1.0, 1.0), lesion_radii_mm=radii, seed=0
    )

    assert ph.mask.any()
    analytic_mm3 = 4.0 / 3.0 * np.pi * radii[0] * radii[1] * radii[2]
    assert ph.mask.sum() * ph.voxel_volume_mm3 == pytest.approx(analytic_mm3, rel=0.03)
    assert ph.ground_truth["lesion_params"]["n_voxels"] == int(ph.mask.sum())


def test_no_lesion_yields_empty_mask_and_null_lesion_params() -> None:
    ph = generate_texture_phantom(size=(16, 16, 16), lesion=False, seed=0)

    assert ph.mask.shape == ph.volume.shape
    assert not ph.mask.any()
    assert ph.ground_truth["lesion"] is False
    assert ph.ground_truth["lesion_params"] is None


def test_frozen_dataclass() -> None:
    ph = generate_texture_phantom(size=(16, 16, 16), seed=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ph.seed = 99  # type: ignore[misc]


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def test_ground_truth_echoes_the_request() -> None:
    ph = generate_texture_phantom(
        size=(32, 32, 32),
        spacing=(1.0, 0.7, 0.7),
        corr_length=5.0,
        anisotropy=(2.0, 1.0, 1.0),
        hu_mean=-30.0,
        hu_sd=12.0,
        lesion=True,
        lesion_center=(16.0, 14.0, 18.0),
        lesion_radii_mm=(5.0, 4.0, 3.0),
        lesion_hu_offset=80.0,
        lesion_hu_sd=9.0,
        lesion_corr_length=2.0,
        seed=42,
    )
    gt = ph.ground_truth

    assert gt["size"] == (32, 32, 32)
    assert gt["spacing"] == (1.0, 0.7, 0.7)
    assert gt["corr_length"] == 5.0
    assert gt["anisotropy"] == (2.0, 1.0, 1.0)
    assert gt["corr_lengths_mm"] == (10.0, 5.0, 5.0)
    assert gt["hu_mean"] == -30.0
    assert gt["hu_sd"] == 12.0
    assert gt["lesion"] is True
    assert gt["seed"] == 42

    les = gt["lesion_params"]
    assert les["center_vox"] == (16.0, 14.0, 18.0)
    assert les["center_mm"] == pytest.approx((16.0, 9.8, 12.6))
    assert les["radii_mm"] == (5.0, 4.0, 3.0)
    assert les["hu_offset"] == 80.0
    assert les["hu_sd"] == 9.0
    assert les["corr_length"] == 2.0


# --------------------------------------------------------------------------
# Intensity statistics
# --------------------------------------------------------------------------


def test_background_intensity_statistics_match_request() -> None:
    ph = generate_texture_phantom(
        size=(64, 64, 64), hu_mean=40.0, hu_sd=25.0, lesion=False, seed=11
    )
    # The field is standardised exactly, so mean/sd are pinned, not merely close.
    assert ph.volume.mean() == pytest.approx(40.0, abs=0.05)
    assert ph.volume.std() == pytest.approx(25.0, rel=1e-3)


def test_homogeneous_background_when_hu_sd_is_zero() -> None:
    ph = generate_texture_phantom(size=(16, 16, 16), hu_sd=0.0, lesion=False, seed=0)
    assert np.allclose(ph.volume, 40.0)


def test_lesion_is_brighter_than_background_by_the_requested_offset() -> None:
    ph = generate_texture_phantom(
        size=(48, 48, 48), hu_mean=40.0, hu_sd=25.0, lesion_hu_offset=60.0, seed=0
    )
    inside = ph.volume[ph.mask].mean()
    outside = ph.volume[~ph.mask].mean()
    # Both fields are standardised over the whole volume, so the in-mask sample
    # mean scatters around the target by a fraction of lesion_hu_sd.
    assert inside - outside == pytest.approx(60.0, abs=12.0)


def test_edge_blur_softens_intensities_but_not_the_mask() -> None:
    common = dict(size=(48, 48, 48), lesion_radii_mm=(8.0, 8.0, 8.0), seed=2)
    sharp = generate_texture_phantom(lesion_edge_blur_mm=0.0, **common)
    blurred = generate_texture_phantom(lesion_edge_blur_mm=2.0, **common)

    assert np.array_equal(sharp.mask, blurred.mask)
    assert not np.array_equal(sharp.volume, blurred.volume)
    # Blurring bleeds lesion signal outward, raising the immediate surroundings.
    shell = (~sharp.mask) & (blurred.volume != sharp.volume)
    assert shell.any()
    assert blurred.volume[shell].mean() > sharp.volume[shell].mean()


# --------------------------------------------------------------------------
# Texture: does the field really have the requested correlation length?
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_measured_correlation_length_increases_monotonically(seed: int) -> None:
    measured = [
        measure_correlation_length(
            generate_texture_phantom(
                size=(64, 64, 64), corr_length=length, lesion=False, seed=seed
            ).volume,
            (1.0, 1.0, 1.0),
            axis=0,
        )
        for length in (2.0, 4.0, 6.0, 8.0)
    ]
    assert measured == sorted(measured)
    assert len(set(measured)) == len(measured)


@pytest.mark.parametrize("requested", [2.0, 4.0, 8.0])
def test_measured_correlation_length_recovers_the_request(requested: float) -> None:
    ph = generate_texture_phantom(size=(64, 64, 64), corr_length=requested, lesion=False, seed=7)
    for axis in (0, 1, 2):
        measured = measure_correlation_length(ph.volume, ph.spacing, axis=axis)
        assert measured == pytest.approx(requested, rel=0.15)


def test_anisotropy_stretches_the_texture_along_the_requested_axis() -> None:
    ph = generate_texture_phantom(
        size=(64, 64, 64), corr_length=4.0, anisotropy=(2.0, 1.0, 1.0), lesion=False, seed=0
    )
    lz, ly, lx = (measure_correlation_length(ph.volume, ph.spacing, a) for a in (0, 1, 2))

    assert lz == pytest.approx(8.0, rel=0.15)
    assert ly == pytest.approx(4.0, rel=0.15)
    assert lx == pytest.approx(4.0, rel=0.15)
    assert lz / ly == pytest.approx(2.0, rel=0.2)


def test_correlation_length_is_measured_in_millimetres_not_voxels() -> None:
    ph = generate_texture_phantom(
        size=(48, 48, 48), spacing=(2.0, 2.0, 2.0), corr_length=8.0, lesion=False, seed=1
    )
    assert measure_correlation_length(ph.volume, ph.spacing, axis=0) == pytest.approx(8.0, rel=0.15)


# --------------------------------------------------------------------------
# Explicit failure, never silent NaN
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(size=(8, 8)), "size must have exactly 3"),
        (dict(size=(2, 8, 8)), "at least 4"),
        (dict(size=(8.0, 8.0, 8.0)), "must contain integers"),
        (dict(spacing=(1.0, 0.0, 1.0)), "spacing must be strictly positive"),
        (dict(spacing=(1.0, 1.0)), "spacing must have exactly 3"),
        (dict(corr_length=0.0), "corr_length must be finite and strictly positive"),
        (dict(corr_length=float("nan")), "corr_length must be finite"),
        (dict(anisotropy=(1.0, -1.0, 1.0)), "anisotropy must be strictly positive"),
        (dict(hu_sd=-1.0), "hu_sd must be finite and non-negative"),
        (dict(hu_mean=float("inf")), "hu_mean must be finite"),
        (dict(seed=1.5), "seed must be an int"),
        (dict(lesion_radii_mm=(0.0, 1.0, 1.0)), "lesion_radii_mm must be strictly positive"),
        (dict(lesion_corr_length=-1.0), "lesion_corr_length must be finite"),
        (dict(lesion_hu_sd=-1.0), "lesion_hu_sd must be finite and non-negative"),
        (dict(lesion_edge_blur_mm=-1.0), "lesion_edge_blur_mm must be finite"),
        (dict(lesion_center=(0.0, 0.0, 99.0)), "lies outside the volume"),
    ],
)
def test_malformed_input_raises_value_error(kwargs: dict, match: str) -> None:
    base = dict(size=(16, 16, 16), seed=0)
    with pytest.raises(ValueError, match=match):
        generate_texture_phantom(**{**base, **kwargs})


def test_subvoxel_lesion_raises_rather_than_returning_an_empty_mask() -> None:
    """An ellipsoid that falls between sample points must not silently vanish."""
    with pytest.raises(ValueError, match="empty mask"):
        generate_texture_phantom(
            size=(16, 16, 16),
            spacing=(4.0, 4.0, 4.0),
            lesion_center=(1.5, 1.5, 1.5),  # midway between voxels, 2 mm from each
            lesion_radii_mm=(0.5, 0.5, 0.5),
            seed=0,
        )


def test_correlation_length_beyond_half_the_volume_is_rejected() -> None:
    """Circular convolution wraps: such a field cannot have the requested texture."""
    with pytest.raises(ValueError, match="exceeds half the field extent"):
        generate_texture_phantom(size=(32, 32, 32), corr_length=40.0, lesion=False, seed=0)

    # Anisotropy is what actually pushes an otherwise-legal request over the edge.
    with pytest.raises(ValueError, match="along axis 0"):
        generate_texture_phantom(
            size=(32, 32, 32), corr_length=10.0, anisotropy=(4.0, 1.0, 1.0), lesion=False, seed=0
        )

    # Exactly at half the extent is still admissible.
    generate_texture_phantom(size=(32, 32, 32), corr_length=16.0, lesion=False, seed=0)


def test_spacing_not_voxel_count_sets_the_extent_limit() -> None:
    """16 voxels of 4 mm is a 64 mm field: a 30 mm correlation length fits."""
    generate_texture_phantom(
        size=(16, 16, 16), spacing=(4.0, 4.0, 4.0), corr_length=30.0, lesion=False, seed=0
    )
    with pytest.raises(ValueError, match="exceeds half the field extent"):
        generate_texture_phantom(
            size=(16, 16, 16), spacing=(1.0, 1.0, 1.0), corr_length=30.0, lesion=False, seed=0
        )


def test_measure_correlation_length_rejects_bad_input() -> None:
    ph = generate_texture_phantom(size=(16, 16, 16), lesion=False, seed=0)

    with pytest.raises(ValueError, match="volume must be 3D"):
        measure_correlation_length(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="volume must be finite"):
        measure_correlation_length(np.full((8, 8, 8), np.nan))
    with pytest.raises(ValueError, match="axis must be 0, 1 or 2"):
        measure_correlation_length(ph.volume, ph.spacing, axis=3)
    with pytest.raises(ValueError, match="zero variance"):
        measure_correlation_length(np.zeros((8, 8, 8)))


def test_measure_correlation_length_refuses_to_extrapolate_beyond_the_volume() -> None:
    """A field with no variation along the axis is unmeasurable -- say so, don't guess."""
    plane = np.random.default_rng(0).standard_normal((16, 16))
    constant_along_z = np.broadcast_to(plane, (16, 16, 16)).copy()

    with pytest.raises(ValueError, match="does not fall to 1/e"):
        measure_correlation_length(constant_along_z, (1.0, 1.0, 1.0), axis=0)

    # ...but the in-plane axes of the very same volume are perfectly measurable.
    assert measure_correlation_length(constant_along_z, (1.0, 1.0, 1.0), axis=1) > 0.0
