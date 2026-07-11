"""Contract tests for the zone, neighbourhood and dependence families.

The IBSI digital phantom pins the *values* of GLSZM, GLDZM, NGTDM and NGLDM,
but it is far too small and too convex to pin some of the definitions those
values rest on.  These tests do that, on volumes whose answer can be worked out
by hand:

* the border distance uses face-connectivity, while zones use full connectivity
  -- a distinction the phantom collapses, because every one of its zones touches
  the ROI border;
* NGLDM dependence counts the centre voxel;
* every family fails loudly on a degenerate ROI.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom.features import (
    ZONE_AGGREGATIONS,
    Discretisation,
    FeatureError,
    _border_distance,
    _distance_zones,
    _label_zones,
    _ngldm_matrix,
    _size_zones,
    discretise,
    gldzm_features,
    glszm_features,
    ngldm_features,
    ngtdm_features,
)
from rphantom.phantom import generate_texture_phantom


def _disc(levels: np.ndarray, mask: np.ndarray | None = None, n_levels: int | None = None):
    """Build a Discretisation straight from integer levels, bypassing binning."""
    levels = np.asarray(levels, dtype=np.int32)
    if mask is None:
        mask = levels > 0
    n_levels = n_levels or int(levels[mask].max())
    return Discretisation(levels, mask, n_levels, "fbs", 1.0, None, (1.0, float(n_levels)))


# --------------------------------------------------------------------------
# Border distance: face-connected, and 1 at the border
# --------------------------------------------------------------------------


def test_border_voxels_are_at_distance_one() -> None:
    mask = np.ones((3, 3, 3), dtype=bool)
    distance = _border_distance(mask)

    assert distance[0, 0, 0] == 1
    assert distance[1, 1, 1] == 2  # the only voxel not touching a face of the cube


def test_a_single_slice_volume_has_every_voxel_on_the_border() -> None:
    """In 3D the slice above and below a lone slice is outside the ROI."""
    assert np.all(_border_distance(np.ones((1, 5, 5), dtype=bool)) == 1)


def test_border_distance_is_face_connected_not_chebyshev() -> None:
    """IBSI measures the distance to the ROI edge with 4-/6-connectivity.

    Here the only nearby exit from the centre is *diagonal*.  Under the
    Chebyshev norm the centre would be one step from outside; under IBSI's
    face-connected definition it is two.  The digital phantom cannot see this
    difference, because all of its zones touch the border already.
    """
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1, 1] = False  # punch out one corner of the 3x3 block

    distance = _border_distance(mask)

    assert distance[2, 2] == 2  # would be 1 under a Chebyshev/chessboard metric
    assert distance[1, 2] == 1


# --------------------------------------------------------------------------
# Zones: full connectivity, and the minimum distance rule
# --------------------------------------------------------------------------


def test_zones_are_diagonally_connected() -> None:
    """Two voxels touching only at a corner belong to the same zone."""
    levels = np.array([[[3, 0], [0, 3]]], dtype=np.int32)
    mask = levels > 0

    labelled = _label_zones(levels, mask, n_levels=3)[2]  # level 3
    assert labelled.max() == 1  # one zone, not two


def test_zone_sizes_are_counted_per_grey_level() -> None:
    levels = np.array([[[1, 1, 2], [1, 2, 2]]], dtype=np.int32)
    zones = _size_zones(levels, np.ones_like(levels, dtype=bool), n_levels=2)

    assert sorted(zip(zones.levels.tolist(), zones.values.tolist(), strict=True)) == [
        (1, 3),
        (2, 3),
    ]
    assert zones.n_voxels == 6


def test_zone_distance_is_the_minimum_over_its_voxels() -> None:
    """A zone reaching the border has distance 1 even if most of it is deep inside."""
    levels = np.ones((1, 5, 5), dtype=np.int32)
    mask = np.ones((1, 5, 5), dtype=bool)
    disc = _disc(levels, mask, n_levels=1)

    features = gldzm_features(disc, "3D")
    assert features.small_distance_emphasis == pytest.approx(1.0)
    assert features.large_distance_emphasis == pytest.approx(1.0)
    assert features.zone_distance_variance == pytest.approx(0.0)


def test_an_interior_zone_has_distance_greater_than_one() -> None:
    """A lone voxel walled in by another grey level is two steps from the edge."""
    levels = np.ones((3, 5, 5), dtype=np.int32)
    levels[1, 2, 2] = 2  # dead centre, so its nearest escape is two steps up in z
    disc = _disc(levels, np.ones((3, 5, 5), dtype=bool), n_levels=2)

    zones = _distance_zones(disc.levels, disc.mask, disc.n_levels)
    assert sorted(zip(zones.levels.tolist(), zones.values.tolist(), strict=True)) == [
        (1, 1),
        (2, 2),
    ]


# --------------------------------------------------------------------------
# GLSZM
# --------------------------------------------------------------------------


def test_glszm_of_a_single_uniform_zone() -> None:
    """One zone of 8 voxels at level 1: every feature follows by hand."""
    disc = _disc(np.ones((2, 2, 2), dtype=np.int32), n_levels=1)
    features = glszm_features(disc, "3D")

    assert features.small_zone_emphasis == pytest.approx(1 / 64)  # 1 / 8^2
    assert features.large_zone_emphasis == pytest.approx(64.0)
    assert features.zone_percentage == pytest.approx(1 / 8)
    assert features.zone_size_entropy == pytest.approx(0.0)  # a single (i, j) cell
    assert features.grey_level_variance == pytest.approx(0.0)


def test_glszm_zone_percentage_is_one_when_every_voxel_is_its_own_zone() -> None:
    levels = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
    disc = _disc(levels, n_levels=8)

    assert glszm_features(disc, "3D").zone_percentage == pytest.approx(1.0)


# --------------------------------------------------------------------------
# NGLDM: the dependence includes the centre voxel
# --------------------------------------------------------------------------


#: A level-1 voxel completely surrounded by level-2 voxels: it resembles none of
#: its eight neighbours, so its dependence is 1 only because the centre counts.
_ISOLATED_CENTRE = np.array([[[2, 2, 2], [2, 1, 2], [2, 2, 2]]], dtype=np.int32)


def test_ngldm_dependence_counts_the_centre_voxel() -> None:
    matrix = _ngldm_matrix(
        _ISOLATED_CENTRE[0], np.ones((3, 3), dtype=bool), n_levels=2, alpha=0, max_dependence=9
    )

    assert matrix.counts.shape == (2, 9)  # columns are dependence 1 .. 9
    assert matrix.counts[0, 0] == 1  # the isolated centre: level 1, dependence 1
    assert matrix.counts.sum() == 9  # one neighbourhood per voxel
    # Of the level-2 ring, the 4 corner voxels see 2 like neighbours (dependence 3)
    # and the 4 edge voxels see 4 (dependence 5).
    assert matrix.counts[1, 2] == 4
    assert matrix.counts[1, 4] == 4


def test_ngldm_low_dependence_emphasis_stays_finite_for_an_isolated_level() -> None:
    """This is why the centre voxel is counted: otherwise 1 / j**2 divides by zero."""
    disc = _disc(_ISOLATED_CENTRE, n_levels=2)

    features = ngldm_features(disc, "2D")
    assert np.isfinite(features.low_dependence_emphasis)
    assert features.dependence_count_percentage == pytest.approx(1.0)


def test_ngldm_alpha_widens_what_counts_as_dependent() -> None:
    disc = _disc(_ISOLATED_CENTRE, n_levels=2)

    strict = ngldm_features(disc, "2D", alpha=0)
    tolerant = ngldm_features(disc, "2D", alpha=1)

    assert tolerant.high_dependence_emphasis > strict.high_dependence_emphasis


def test_ngldm_rejects_a_negative_alpha() -> None:
    disc = _disc(np.array([[[1, 2]]], dtype=np.int32), n_levels=2)
    with pytest.raises(FeatureError, match="alpha must be non-negative"):
        ngldm_features(disc, "2D", alpha=-1)


# --------------------------------------------------------------------------
# Degenerate ROIs raise instead of returning nan
# --------------------------------------------------------------------------


def test_ngtdm_raises_on_a_single_grey_level() -> None:
    disc = _disc(np.ones((3, 3, 3), dtype=np.int32), n_levels=1)
    with pytest.raises(FeatureError, match="single grey level"):
        ngtdm_features(disc, "3D")


def test_glszm_and_ngldm_survive_a_single_grey_level() -> None:
    """Zones and dependences remain well defined when there is nothing to contrast."""
    disc = _disc(np.ones((3, 3, 3), dtype=np.int32), n_levels=1)

    assert glszm_features(disc, "3D").grey_level_variance == pytest.approx(0.0)
    assert ngldm_features(disc, "3D").grey_level_variance == pytest.approx(0.0)


@pytest.mark.parametrize(
    "features", [glszm_features, gldzm_features, ngtdm_features, ngldm_features]
)
def test_unknown_aggregation_raises(features) -> None:
    phantom = generate_texture_phantom(size=(16, 16, 16), seed=0)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    with pytest.raises(FeatureError, match="aggregation must be one of"):
        features(disc, "3D_avg")  # a GLCM aggregation name, not a zone one


# --------------------------------------------------------------------------
# Integration with the synthetic phantom
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", ZONE_AGGREGATIONS)
def test_all_zone_families_are_finite_on_a_synthetic_phantom(aggregation: str) -> None:
    phantom = generate_texture_phantom(size=(24, 24, 24), seed=1)
    disc = discretise(phantom.volume, phantom.mask, method="fbs", bin_width=25.0)

    for compute in (glszm_features, gldzm_features, ngtdm_features, ngldm_features):
        values = compute(disc, aggregation).to_dict(aggregation)
        assert values, compute.__name__
        assert all(np.isfinite(v) for v in values.values()), compute.__name__


def test_coarser_texture_makes_larger_zones() -> None:
    """A longer correlation length means neighbours share a bin more often."""
    emphases = []
    for corr_length in (2.0, 8.0):
        phantom = generate_texture_phantom(
            size=(32, 32, 32), corr_length=corr_length, lesion=False, seed=2
        )
        disc = discretise(
            phantom.volume, np.ones(phantom.shape, bool), method="fbs", bin_width=25.0
        )
        emphases.append(glszm_features(disc, "3D").large_zone_emphasis)

    assert emphases[1] > emphases[0]


def test_coarser_texture_lowers_ngtdm_contrast() -> None:
    contrasts = []
    for corr_length in (2.0, 8.0):
        phantom = generate_texture_phantom(
            size=(32, 32, 32), corr_length=corr_length, lesion=False, seed=3
        )
        disc = discretise(
            phantom.volume, np.ones(phantom.shape, bool), method="fbs", bin_width=25.0
        )
        contrasts.append(ngtdm_features(disc, "3D").contrast)

    assert contrasts[0] > contrasts[1]
