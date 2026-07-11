"""Contract tests for morphology, local intensity and the intensity-volume histogram.

The IBSI digital phantom pins the *values* of these families, but on a shape too
small and lumpy to exercise their geometry.  These tests do that on shapes whose
answer is analytic -- a solid cube, a sphere, an anisotropic box -- plus the
usual input-validation and degenerate-ROI contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom.features import (
    FeatureError,
    discretise,
    intensity_volume_histogram,
    local_intensity_features,
    morphology_features,
)
from rphantom.phantom import generate_texture_phantom


def _textured(shape: tuple[int, int, int]) -> np.ndarray:
    """A non-constant intensity field, so Moran's I and Geary's C are defined."""
    return np.fromfunction(lambda z, y, x: 10.0 + z + 2.0 * y + 3.0 * x, shape, dtype=float)


def _solid_cube(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    return _textured((n, n, n)), np.ones((n, n, n), dtype=bool)


def _solid_ball(n: int = 21, radius: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    c = (n - 1) / 2.0
    zz, yy, xx = np.ogrid[:n, :n, :n]
    mask = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= radius**2
    return _textured((n, n, n)), mask


# --------------------------------------------------------------------------
# Morphology on analytic shapes
# --------------------------------------------------------------------------


def test_cube_volume_and_area_are_close_to_the_analytic_values() -> None:
    volume, mask = _solid_cube(n=12)
    morph = morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))

    # Marching cubes rounds the corners, so the mesh volume/area sit just under
    # the 12^3 cube; voxel counting gives the exact cube.
    assert morph.approximate_volume == pytest.approx(12**3)
    assert morph.volume == pytest.approx(12**3, rel=0.05)
    assert morph.surface_area == pytest.approx(6 * 12**2, rel=0.05)


def test_sphere_is_nearly_spherical() -> None:
    volume, mask = _solid_ball(n=25, radius=10.0)
    morph = morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))

    # Marching cubes faceting keeps sphericity a little under 1; it is still the
    # roundest shape these tests build.
    assert morph.sphericity == pytest.approx(0.91, abs=0.03)
    assert morph.elongation == pytest.approx(1.0, abs=0.05)
    assert morph.flatness == pytest.approx(1.0, abs=0.05)
    assert morph.surface_to_volume_ratio == pytest.approx(3.0 / 10.0, rel=0.12)


def test_pca_axes_follow_the_longest_physical_direction() -> None:
    """A box elongated in z must have its major PCA axis along z."""
    mask = np.zeros((20, 6, 6), dtype=bool)
    mask[2:18, 1:5, 1:5] = True
    volume = _textured((20, 6, 6))

    morph = morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))
    assert morph.major_axis_length > morph.minor_axis_length
    assert morph.minor_axis_length == pytest.approx(morph.least_axis_length, rel=0.1)
    assert morph.elongation < 0.5  # clearly not isotropic


def test_spacing_scales_the_volume_cubically() -> None:
    volume, mask = _solid_cube(n=8)
    unit = morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))
    coarse = morphology_features(volume, mask, spacing=(2.0, 2.0, 2.0))

    assert coarse.approximate_volume == pytest.approx(8.0 * unit.approximate_volume)
    assert coarse.volume == pytest.approx(8.0 * unit.volume, rel=1e-6)


def test_integrated_intensity_is_mean_times_volume() -> None:
    volume, mask = _solid_cube(n=6)
    volume[mask] = np.random.default_rng(0).uniform(10.0, 50.0, size=int(mask.sum()))
    morph = morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))

    assert morph.integrated_intensity == pytest.approx(morph.volume * volume[mask].mean())


def test_centre_of_mass_shift_is_zero_for_symmetric_intensity() -> None:
    """Intensity symmetric about the centroid keeps the weighted centroid there."""
    n = 21
    c = (n - 1) / 2.0
    zz, yy, xx = np.ogrid[:n, :n, :n]
    mask = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= 8.0**2
    volume = 100.0 + (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2  # radial, symmetric

    morph = morphology_features(volume.astype(float), mask, spacing=(1.0, 1.0, 1.0))
    assert morph.centre_of_mass_shift == pytest.approx(0.0, abs=1e-9)


def test_morans_i_is_positive_for_a_smooth_gradient() -> None:
    """A monotonic ramp is positively autocorrelated: I > 0 and C < 1."""
    mask = np.ones((10, 10, 10), dtype=bool)
    ramp = np.arange(10, dtype=np.float64)[:, None, None] * np.ones((10, 10, 10))
    smooth = morphology_features(ramp, mask, spacing=(1.0, 1.0, 1.0))

    # ...and a salt-and-pepper field is anti-correlated: lower I, higher C.
    noise = np.indices((10, 10, 10)).sum(axis=0) % 2 * 1.0
    checker = morphology_features(noise + ramp * 1e-6, mask, spacing=(1.0, 1.0, 1.0))

    assert smooth.morans_i > 0.0
    assert smooth.gearys_c < 1.0
    assert smooth.morans_i > checker.morans_i


# --------------------------------------------------------------------------
# Local intensity
# --------------------------------------------------------------------------


def test_global_peak_is_at_least_the_local_peak() -> None:
    phantom = generate_texture_phantom(size=(20, 20, 20), seed=0)
    features = local_intensity_features(phantom.volume, phantom.mask, phantom.spacing)
    assert features.global_intensity_peak >= features.local_intensity_peak - 1e-9


def test_local_peak_of_a_uniform_region_is_that_value() -> None:
    volume = np.full((9, 9, 9), 42.0)
    mask = np.ones((9, 9, 9), dtype=bool)
    features = local_intensity_features(volume, mask, spacing=(1.0, 1.0, 1.0))
    assert features.local_intensity_peak == pytest.approx(42.0)
    assert features.global_intensity_peak == pytest.approx(42.0)


def test_peak_averages_over_the_sphere_not_just_the_voxel() -> None:
    """A lone spike is smoothed by its cooler surroundings; the peak is below it."""
    volume = np.full((11, 11, 11), 10.0)
    volume[5, 5, 5] = 1000.0
    mask = np.ones((11, 11, 11), dtype=bool)

    features = local_intensity_features(volume, mask, spacing=(1.0, 1.0, 1.0))
    assert features.local_intensity_peak < 1000.0
    assert features.local_intensity_peak > 10.0


# --------------------------------------------------------------------------
# Intensity-volume histogram
# --------------------------------------------------------------------------


def test_ivh_volume_fractions_are_monotone() -> None:
    phantom = generate_texture_phantom(size=(24, 24, 24), seed=1)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)
    ivh = intensity_volume_histogram(disc)

    assert 0.0 <= ivh.volume_fraction_at_90pct <= ivh.volume_fraction_at_10pct <= 1.0
    assert ivh.intensity_at_10pct_volume >= ivh.intensity_at_90pct_volume
    assert ivh.volume_fraction_difference == pytest.approx(
        ivh.volume_fraction_at_10pct - ivh.volume_fraction_at_90pct
    )


def test_ivh_needs_at_least_two_levels() -> None:
    volume = np.full((6, 6, 6), 5.0)
    mask = np.ones((6, 6, 6), dtype=bool)
    disc = discretise(volume, mask, method="fbs", bin_width=1.0)
    with pytest.raises(FeatureError, match="at least 2 grey levels"):
        intensity_volume_histogram(disc)


# --------------------------------------------------------------------------
# Input validation and degenerate ROIs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("compute", [morphology_features, local_intensity_features])
def test_bad_spacing_raises(compute) -> None:
    volume, mask = _solid_cube(n=6)
    with pytest.raises(FeatureError, match="spacing must"):
        compute(volume, mask, spacing=(1.0, 0.0, 1.0))
    with pytest.raises(FeatureError, match="spacing must"):
        compute(volume, mask, spacing=(1.0, 1.0))


@pytest.mark.parametrize("compute", [morphology_features, local_intensity_features])
def test_empty_mask_raises(compute) -> None:
    volume = np.zeros((6, 6, 6))
    with pytest.raises(FeatureError, match="ROI mask is empty"):
        compute(volume, np.zeros((6, 6, 6), dtype=bool))


def test_morphology_of_a_flat_roi_raises() -> None:
    """A single-voxel-thick sheet has no enclosed volume to mesh."""
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 1:7, 1:7] = True
    with pytest.raises(FeatureError, match="planar|degenerate|could not build"):
        morphology_features(np.ones((1, 8, 8)), mask, spacing=(1.0, 1.0, 1.0))


def test_morans_i_raises_on_constant_intensity_when_asked_directly() -> None:
    """A constant ROI leaves Moran's I undefined (0/0); it must raise, not nan."""
    from rphantom.features import _morans_i

    coords = np.argwhere(np.ones((4, 4, 4), dtype=bool)).astype(float)
    with pytest.raises(FeatureError, match="Moran's I is undefined"):
        _morans_i(coords, np.ones(coords.shape[0]))


def test_constant_intensity_roi_raises_not_nan() -> None:
    """Moran's I is 0/0 on constant intensity; the whole call raises, by design."""
    volume = np.full((8, 8, 8), 3.0)
    mask = np.ones((8, 8, 8), dtype=bool)
    with pytest.raises(FeatureError, match="constant ROI|undefined"):
        morphology_features(volume, mask, spacing=(1.0, 1.0, 1.0))


def test_determinism_on_a_synthetic_phantom() -> None:
    a = generate_texture_phantom(size=(20, 20, 20), seed=5)
    b = generate_texture_phantom(size=(20, 20, 20), seed=5)

    ma = morphology_features(a.volume, a.mask, a.spacing).to_dict()
    mb = morphology_features(b.volume, b.mask, b.spacing).to_dict()
    assert ma == mb
