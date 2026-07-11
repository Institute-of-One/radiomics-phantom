"""Contract tests for :mod:`rphantom.features`.

The IBSI reference values in ``test_features_ibsi.py`` pin *correctness* on one
tiny phantom.  These tests pin everything that phantom cannot see: input
validation, degenerate ROIs, the run-length decomposition on volumes whose
answer can be worked out by hand, and the interaction with
:mod:`rphantom.phantom`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom.features import (
    AGGREGATIONS,
    Discretisation,
    FeatureError,
    _run_length_matrix,
    discretise,
    glcm_features,
    glrlm_features,
    intensity_histogram,
    intensity_statistics,
)
from rphantom.phantom import generate_texture_phantom


@pytest.fixture
def uniform_roi() -> tuple[np.ndarray, np.ndarray]:
    """A volume whose ROI holds a single intensity."""
    volume = np.full((4, 4, 4), 7.0)
    return volume, np.ones((4, 4, 4), dtype=bool)


# --------------------------------------------------------------------------
# Discretisation
# --------------------------------------------------------------------------


def test_fixed_bin_size_places_values_by_bin_width() -> None:
    volume = np.array([[[0.0, 0.9, 1.0, 2.5, 4.0]]])
    mask = np.ones_like(volume, dtype=bool)

    disc = discretise(volume, mask, method="fbs", bin_width=1.0)

    assert disc.roi_levels.tolist() == [1, 1, 2, 3, 5]
    assert disc.n_levels == 5
    assert disc.intensity_range == (0.0, 4.0)


def test_fixed_bin_number_puts_the_maximum_in_the_last_bin() -> None:
    volume = np.array([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    mask = np.ones_like(volume, dtype=bool)

    disc = discretise(volume, mask, method="fbn", bin_number=4)

    assert disc.roi_levels.tolist() == [1, 2, 3, 4, 4]
    assert disc.n_levels == 4


def test_explicit_intensity_range_clips_and_fixes_the_level_count() -> None:
    volume = np.array([[[-100.0, 0.0, 50.0, 400.0]]])
    mask = np.ones_like(volume, dtype=bool)

    disc = discretise(volume, mask, method="fbs", bin_width=25.0, intensity_range=(0.0, 100.0))

    assert disc.intensity_range == (0.0, 100.0)
    assert disc.roi_levels.tolist() == [1, 1, 3, 5]  # -100 and 400 clipped to the bounds
    assert disc.n_levels == 5  # floor(100 / 25) + 1: the upper bound opens its own bin


def test_explicit_intensity_range_pins_the_level_count_regardless_of_the_data() -> None:
    """The bin grid comes from the range, so sparse data still spans every level."""
    volume = np.array([[[0.0, 10.0, 20.0]]])
    mask = np.ones_like(volume, dtype=bool)

    disc = discretise(volume, mask, method="fbs", bin_width=25.0, intensity_range=(0.0, 100.0))

    assert disc.roi_levels.tolist() == [1, 1, 1]
    assert disc.n_levels == 5  # unchanged: comparable matrices across images


def test_discretisation_is_deterministic() -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=0)
    a = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=10.0)
    b = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=10.0)

    assert np.array_equal(a.levels, b.levels)
    assert a.n_levels == b.n_levels


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(method="fbs"), "requires a positive bin_width"),
        (dict(method="fbs", bin_width=0.0), "requires a positive bin_width"),
        (dict(method="fbs", bin_width=1.0, bin_number=4), "bin_number must not be given"),
        (dict(method="fbn"), "requires bin_number"),
        (dict(method="fbn", bin_number=0), "requires bin_number"),
        (dict(method="fbn", bin_number=4, bin_width=1.0), "bin_width must not be given"),
        (dict(method="quantile", bin_width=1.0), "must be 'fbs' or 'fbn'"),
    ],
)
def test_discretise_rejects_mis_specified_methods(kwargs: dict, match: str) -> None:
    volume = np.array([[[1.0, 2.0, 3.0]]])
    mask = np.ones_like(volume, dtype=bool)

    with pytest.raises(FeatureError, match=match):
        discretise(volume, mask, **kwargs)


def test_discretise_rejects_a_bad_intensity_range() -> None:
    volume = np.array([[[1.0, 2.0, 3.0]]])
    mask = np.ones_like(volume, dtype=bool)

    with pytest.raises(FeatureError, match="finite and increasing"):
        discretise(volume, mask, method="fbs", bin_width=1.0, intensity_range=(5.0, 1.0))


def test_fixed_bin_number_refuses_a_constant_roi(uniform_roi) -> None:
    volume, mask = uniform_roi
    with pytest.raises(FeatureError, match="undefined for a constant ROI"):
        discretise(volume, mask, method="fbn", bin_number=4)


def test_fixed_bin_size_accepts_a_constant_roi(uniform_roi) -> None:
    volume, mask = uniform_roi
    disc = discretise(volume, mask, method="fbs", bin_width=1.0)

    assert disc.n_levels == 1
    assert np.all(disc.roi_levels == 1)


# --------------------------------------------------------------------------
# Input validation, shared by every family
# --------------------------------------------------------------------------


def test_empty_mask_raises() -> None:
    volume = np.zeros((4, 4, 4))
    with pytest.raises(FeatureError, match="ROI mask is empty"):
        intensity_statistics(volume, np.zeros((4, 4, 4), dtype=bool))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(FeatureError, match="does not match volume shape"):
        intensity_statistics(np.zeros((4, 4, 4)), np.ones((4, 4, 5), dtype=bool))


def test_non_boolean_mask_raises() -> None:
    with pytest.raises(FeatureError, match="mask must be boolean"):
        intensity_statistics(np.zeros((4, 4, 4)), np.ones((4, 4, 4), dtype=np.uint8))


def test_two_dimensional_volume_raises() -> None:
    with pytest.raises(FeatureError, match="volume must be 3D"):
        intensity_statistics(np.zeros((4, 4)), np.ones((4, 4), dtype=bool))


def test_non_finite_intensities_raise_rather_than_propagate() -> None:
    volume = np.ones((4, 4, 4))
    volume[0, 0, 0] = np.nan
    mask = np.ones((4, 4, 4), dtype=bool)

    with pytest.raises(FeatureError, match="non-finite"):
        intensity_statistics(volume, mask)


# --------------------------------------------------------------------------
# Degenerate ROIs raise instead of returning nan
# --------------------------------------------------------------------------


def test_constant_roi_raises_for_intensity_statistics(uniform_roi) -> None:
    volume, mask = uniform_roi
    with pytest.raises(FeatureError, match="zero intensity variance"):
        intensity_statistics(volume, mask)


def test_single_grey_level_raises_for_glcm(uniform_roi) -> None:
    volume, mask = uniform_roi
    disc = discretise(volume, mask, method="fbs", bin_width=1.0)

    with pytest.raises(FeatureError, match="single grey level"):
        glcm_features(disc, "3D_comb")


def test_single_grey_level_is_fine_for_glrlm(uniform_roi) -> None:
    """A one-level ROI has perfectly well-defined runs, so it must not raise."""
    volume, mask = uniform_roi
    disc = discretise(volume, mask, method="fbs", bin_width=1.0)

    features = glrlm_features(disc, "3D_comb")
    assert features.grey_level_variance == pytest.approx(0.0)
    assert features.run_entropy >= 0.0


def test_roi_too_small_for_any_voxel_pair_raises() -> None:
    volume = np.arange(64, dtype=float).reshape(4, 4, 4)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[3, 3, 3] = True  # two voxels, never neighbours
    disc = discretise(volume, mask, method="fbs", bin_width=1.0)

    with pytest.raises(FeatureError, match="no non-empty texture matrix"):
        glcm_features(disc, "3D_comb")


def test_unknown_aggregation_raises() -> None:
    volume = np.arange(64, dtype=float).reshape(4, 4, 4)
    disc = discretise(volume, np.ones((4, 4, 4), dtype=bool), method="fbs", bin_width=8.0)

    with pytest.raises(FeatureError, match="aggregation must be one of"):
        glcm_features(disc, "4D_avg")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The run-length decomposition, on volumes solvable by hand
# --------------------------------------------------------------------------


def test_run_length_matrix_of_a_single_line() -> None:
    """One line ``1 1 2 2 2`` along x: one run of 1s (length 2), one of 2s (length 3)."""
    levels = np.array([[[1, 1, 2, 2, 2]]], dtype=np.int32)
    mask = np.ones_like(levels, dtype=bool)

    matrix = _run_length_matrix(levels, mask, (0, 0, 1), n_levels=2, max_run=5)

    assert matrix[0, 1] == 1  # level 1, run length 2
    assert matrix[1, 2] == 1  # level 2, run length 3
    assert matrix.sum() == 2


def test_leaving_the_roi_breaks_a_run() -> None:
    """``1 1 [gap] 1 1`` must give two runs of length 2, not one of length 4."""
    levels = np.array([[[1, 1, 1, 1, 1]]], dtype=np.int32)
    mask = np.array([[[True, True, False, True, True]]])

    matrix = _run_length_matrix(levels, mask, (0, 0, 1), n_levels=1, max_run=5)

    assert matrix[0, 1] == 2
    assert matrix.sum() == 2


def test_runs_do_not_wrap_between_lines() -> None:
    """Two separate rows of the same level are two runs, not one."""
    levels = np.array([[[3, 3], [3, 3]]], dtype=np.int32)
    mask = np.ones_like(levels, dtype=bool)

    matrix = _run_length_matrix(levels, mask, (0, 0, 1), n_levels=3, max_run=2)

    assert matrix[2, 1] == 2  # level 3, run length 2, twice
    assert matrix.sum() == 2


def test_diagonal_direction_finds_the_diagonal_run() -> None:
    levels = np.array([[[1, 9], [9, 1]]], dtype=np.int32)
    mask = np.ones_like(levels, dtype=bool)

    diagonal = _run_length_matrix(levels, mask, (0, 1, 1), n_levels=9, max_run=2)
    anti_diagonal = _run_length_matrix(levels, mask, (0, 1, -1), n_levels=9, max_run=2)

    # Along (dy, dx) = (1, 1) the main diagonal reads 1, 1: one run of length 2.
    assert diagonal[0, 1] == 1
    # Along (1, -1) the anti-diagonal reads 9, 9: one run of length 2.
    assert anti_diagonal[8, 1] == 1


def test_run_percentage_is_one_when_no_two_neighbours_share_a_level() -> None:
    """Every voxel is its own run, so runs == voxels."""
    levels = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
    mask = np.ones((2, 2, 2), dtype=bool)
    disc = Discretisation(levels, mask, 8, "fbs", 1.0, None, (1.0, 8.0))

    assert glrlm_features(disc, "3D_avg").run_percentage == pytest.approx(1.0)


# --------------------------------------------------------------------------
# GLCM structural properties
# --------------------------------------------------------------------------


def test_glcm_probabilities_make_joint_maximum_a_probability() -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=1)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    for aggregation in AGGREGATIONS:
        glcm = glcm_features(disc, aggregation)
        assert 0.0 < glcm.joint_maximum <= 1.0
        assert 0.0 < glcm.angular_second_moment <= 1.0
        assert -1.0 <= glcm.correlation <= 1.0 + 1e-12
        assert glcm.joint_entropy >= 0.0


def test_aggregation_methods_disagree() -> None:
    """If two aggregations returned the same value, one of them is not implemented."""
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=2)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    contrasts = {a: glcm_features(disc, a).contrast for a in AGGREGATIONS}
    assert len(set(contrasts.values())) == len(AGGREGATIONS)


# --------------------------------------------------------------------------
# Integration with the synthetic phantom
# --------------------------------------------------------------------------


def test_features_of_a_synthetic_phantom_are_finite_and_deterministic() -> None:
    phantom = generate_texture_phantom(size=(24, 24, 24), seed=3)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    first = {
        **intensity_statistics(phantom.volume, phantom.mask).to_dict(),
        **intensity_histogram(disc).to_dict(),
        **glcm_features(disc, "3D_comb").to_dict("3D_comb"),
        **glrlm_features(disc, "3D_comb").to_dict("3D_comb"),
    }
    assert all(np.isfinite(v) for v in first.values())

    again = generate_texture_phantom(size=(24, 24, 24), seed=3)
    disc_again = discretise(again.volume, again.mask, method="fbs", bin_width=25.0)
    second = {
        **intensity_statistics(again.volume, again.mask).to_dict(),
        **intensity_histogram(disc_again).to_dict(),
        **glcm_features(disc_again, "3D_comb").to_dict("3D_comb"),
        **glrlm_features(disc_again, "3D_comb").to_dict("3D_comb"),
    }
    assert first == second


def test_lesion_mask_selects_the_lesion_texture() -> None:
    """The lesion is brighter than the background, so its mean must be higher."""
    phantom = generate_texture_phantom(size=(32, 32, 32), lesion_hu_offset=60.0, seed=4)

    lesion = intensity_statistics(phantom.volume, phantom.mask)
    background = intensity_statistics(phantom.volume, ~phantom.mask)

    assert lesion.mean > background.mean + 40.0


def test_coarser_texture_lowers_glcm_contrast() -> None:
    """A longer correlation length means neighbours are more alike."""
    contrasts = []
    for corr_length in (2.0, 8.0):
        phantom = generate_texture_phantom(
            size=(32, 32, 32), corr_length=corr_length, lesion=False, seed=5
        )
        disc = discretise(phantom.volume, phantom.mask | True, method="fbs", bin_width=25.0)
        contrasts.append(glcm_features(disc, "3D_comb").contrast)

    assert contrasts[0] > contrasts[1]


def test_to_dict_tags_are_unique_and_suffixed() -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=6)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    tags: set[str] = set()
    for aggregation in AGGREGATIONS:
        for tag in glcm_features(disc, aggregation).to_dict(aggregation):
            assert tag.endswith(aggregation)
            tags.add(tag)
    assert len(tags) == 25 * len(AGGREGATIONS)
