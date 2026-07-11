"""Validate :mod:`rphantom.features` against the IBSI digital phantom.

IBSI publishes reference values for its digital phantom with a tolerance of
*zero*: a compliant implementation must reproduce every value exactly, at the
three significant digits to which they are published.  This module asserts
exactly that, once per reference value, so a regression names the feature that
broke rather than merely reporting that "features changed".

Coverage: all 482 published digital-phantom reference values.  (IBSI leaves four
further morphology features -- the OMBB and MVEE densities -- unstandardised,
with no reference value, so they are outside this count and not implemented.)
:func:`test_reference_coverage_is_declared` pins that boundary so it cannot
silently drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rphantom.features import (
    AGGREGATIONS,
    ZONE_AGGREGATIONS,
    discretise,
    glcm_features,
    gldzm_features,
    glrlm_features,
    glszm_features,
    intensity_histogram,
    intensity_statistics,
    intensity_volume_histogram,
    local_intensity_features,
    morphology_features,
    ngldm_features,
    ngtdm_features,
)
from tests.ibsi_reference import MASK, PHANTOM, REFERENCE, SPACING

#: Every family, by IBSI tag prefix, with its count of published reference values.
IMPLEMENTED_FAMILIES = {
    "stat_": 18,
    "ih_": 23,
    "cm_": 150,
    "rlm_": 96,
    "szm_": 48,
    "dzm_": 48,
    "ngt_": 15,
    "ngl_": 51,
    "morph_": 25,
    "loc_": 2,
    "ivh_": 6,
}


def round_to_3_significant_digits(value: float) -> float:
    """Round as IBSI publishes: three significant digits."""
    if value == 0.0:
        return 0.0
    if not math.isfinite(value):
        raise AssertionError(f"feature value is not finite: {value}")
    return round(value, -int(math.floor(math.log10(abs(value)))) + 2)


@pytest.fixture(scope="module")
def computed() -> dict[str, float]:
    """Every implemented feature of the digital phantom, keyed by IBSI tag.

    IBSI prescribes no interpolation, no re-segmentation and no discretisation
    for this phantom: its intensities are already integers.  A fixed bin size of
    1 is the identity discretisation and reproduces that.
    """
    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)

    values: dict[str, float] = {}
    values.update(intensity_statistics(PHANTOM, MASK).to_dict())
    values.update(intensity_histogram(disc).to_dict())
    for aggregation in AGGREGATIONS:
        values.update(glcm_features(disc, aggregation).to_dict(aggregation))
        values.update(glrlm_features(disc, aggregation).to_dict(aggregation))
    for aggregation in ZONE_AGGREGATIONS:
        values.update(glszm_features(disc, aggregation).to_dict(aggregation))
        values.update(gldzm_features(disc, aggregation).to_dict(aggregation))
        values.update(ngtdm_features(disc, aggregation).to_dict(aggregation))
        values.update(ngldm_features(disc, aggregation).to_dict(aggregation))
    values.update(morphology_features(PHANTOM, MASK, SPACING).to_dict())
    values.update(local_intensity_features(PHANTOM, MASK, SPACING).to_dict())
    values.update(intensity_volume_histogram(disc).to_dict())
    return values


# --------------------------------------------------------------------------
# The phantom itself
# --------------------------------------------------------------------------


def test_phantom_matches_its_published_description() -> None:
    assert PHANTOM.shape == (4, 4, 5)  # (z, y, x) of the 5 x 4 x 4 (x, y, z) phantom
    assert int(MASK.sum()) == 74
    assert sorted(np.unique(PHANTOM[MASK]).tolist()) == [1.0, 3.0, 4.0, 6.0]
    # "grey levels 2 and 5 are absent; 1 is the lowest and 6 the highest"
    assert PHANTOM[MASK].min() == 1.0
    assert PHANTOM[MASK].max() == 6.0


def test_identity_discretisation_spans_six_levels_two_of_them_empty() -> None:
    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)

    assert disc.n_levels == 6
    assert np.array_equal(disc.roi_levels, PHANTOM[MASK].astype(np.int32))
    occupied = np.unique(disc.roi_levels)
    assert 2 not in occupied and 5 not in occupied


def test_fixed_bin_size_1_and_fixed_bin_number_6_agree_on_this_phantom() -> None:
    """IBSI offers both as valid options for the phantom; they must coincide."""
    by_size = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)
    by_number = discretise(PHANTOM, MASK, method="fbn", bin_number=6)

    assert by_size.n_levels == by_number.n_levels == 6
    assert np.array_equal(by_size.levels, by_number.levels)


# --------------------------------------------------------------------------
# The 482 reference values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tag", sorted(REFERENCE))
def test_feature_matches_ibsi_reference_value(tag: str, computed: dict[str, float]) -> None:
    assert tag in computed, f"{tag} has a published reference value but is not computed"
    assert round_to_3_significant_digits(computed[tag]) == REFERENCE[tag]


def test_reference_coverage_is_declared() -> None:
    """Every reference value we ship belongs to a family we claim to implement."""
    counts = {prefix: 0 for prefix in IMPLEMENTED_FAMILIES}
    for tag in REFERENCE:
        matching = [p for p in IMPLEMENTED_FAMILIES if tag.startswith(p)]
        assert len(matching) == 1, f"{tag} matches {matching} implemented families"
        counts[matching[0]] += 1

    assert counts == IMPLEMENTED_FAMILIES
    assert sum(counts.values()) == 482


# --------------------------------------------------------------------------
# Identities that the reference values imply
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_sum_variance_equals_cluster_tendency(aggregation: str) -> None:
    """Both are the variance of ``i + j``, so IBSI publishes them as equal."""
    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)
    glcm = glcm_features(disc, aggregation)

    assert glcm.sum_variance == pytest.approx(glcm.cluster_tendency, rel=1e-12)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_difference_average_equals_dissimilarity(aggregation: str) -> None:
    """Both are the mean of ``|i - j|`` under the same distribution."""
    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)
    glcm = glcm_features(disc, aggregation)

    assert glcm.difference_average == pytest.approx(glcm.dissimilarity, rel=1e-12)


def test_intensity_histogram_reproduces_intensity_statistics_here() -> None:
    """With an identity discretisation the two families must coincide."""
    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)
    stats = intensity_statistics(PHANTOM, MASK)
    hist = intensity_histogram(disc)

    shared = set(type(stats).TAGS) & set(type(hist).TAGS)
    assert len(shared) == 16  # mode is histogram-only; energy and RMS are statistics-only
    for field in sorted(shared):
        assert getattr(stats, field) == pytest.approx(getattr(hist, field), rel=1e-12)


# --------------------------------------------------------------------------
# What the digital phantom cannot tell us
# --------------------------------------------------------------------------


def test_phantom_does_not_discriminate_the_percentile_definition() -> None:
    """An honest negative result, recorded so nobody mistakes it for coverage.

    ``P10``/``P90`` are used by ``rmad``, ``iqr`` and ``qcod``.  On this phantom
    the nearest-rank and linear-interpolation definitions agree, so these tests
    do *not* pin our choice.  Pinning it needs the IBSI CT benchmark
    (configurations A-E), which is future work.
    """
    values = np.sort(PHANTOM[MASK])
    for k in (10.0, 25.0, 75.0, 90.0):
        rank = min(max(int(np.ceil(k / 100.0 * values.size)), 1), values.size)
        nearest_rank = float(values[rank - 1])
        assert nearest_rank == pytest.approx(float(np.percentile(values, k)))


def test_phantom_does_not_discriminate_the_gldzm_distance_metric() -> None:
    """Another honest negative result.

    IBSI measures the distance to the ROI edge with face-connectivity.  Every
    zone of this phantom touches the border, so the *zone* distances -- minima
    over each zone -- come out identical under a Chebyshev metric even though the
    per-voxel distance maps differ.  ``test_features_zones.py`` pins the metric on
    a volume that can tell the two apart.
    """
    from scipy.ndimage import distance_transform_cdt

    from rphantom.features import _label_zones

    disc = discretise(PHANTOM, MASK, method="fbs", bin_width=1.0)

    def zone_distances(metric: str) -> list[tuple[int, int]]:
        padded = distance_transform_cdt(np.pad(MASK, 1, constant_values=False), metric=metric)
        distance = padded[1:-1, 1:-1, 1:-1]
        found = []
        for level, labelled in enumerate(_label_zones(disc.levels, disc.mask, disc.n_levels), 1):
            for zone in range(1, int(labelled.max()) + 1):
                found.append((level, int(distance[labelled == zone].min())))
        return sorted(found)

    assert not np.array_equal(
        distance_transform_cdt(MASK, metric="taxicab"),
        distance_transform_cdt(MASK, metric="chessboard"),
    )
    assert zone_distances("taxicab") == zone_distances("chessboard")
