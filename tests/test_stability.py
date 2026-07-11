"""Tests for :mod:`rphantom.stability`.

ICC(2,1) is validated against ``pingouin`` (labelled ``ICC(A,1)`` there) as an
independent oracle, the way the feature core is validated against IBSI.  The
concordance coefficient and the atlas builder are checked on cases whose answer
is analytic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rphantom import generate_texture_phantom
from rphantom.stability import (
    StabilityError,
    build_stability_atlas,
    concordance_correlation,
    intraclass_correlation,
)

pingouin = pytest.importorskip("pingouin")
import pandas as pd  # noqa: E402  (only needed alongside pingouin)


def _pingouin_icc_a1(table: np.ndarray) -> float:
    n, k = table.shape
    rows = [{"target": t, "rater": r, "score": table[t, r]} for t in range(n) for r in range(k)]
    result = pingouin.intraclass_corr(
        data=pd.DataFrame(rows), targets="target", raters="rater", ratings="score"
    )
    return float(result.set_index("Type").loc["ICC(A,1)", "ICC"])


# --------------------------------------------------------------------------
# ICC(2,1) against pingouin
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_icc_matches_pingouin(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(4, 10)), int(rng.integers(3, 7))
    table = rng.normal(size=(n, k)) + np.arange(n)[:, None] * rng.uniform(0.0, 3.0)

    assert intraclass_correlation(table).icc == pytest.approx(_pingouin_icc_a1(table), abs=1e-9)


def test_icc_is_one_for_perfectly_reproducible_measurements() -> None:
    """Identical columns: every condition agrees, ICC is 1."""
    column = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    table = np.column_stack([column, column, column])
    assert intraclass_correlation(table).icc == pytest.approx(1.0)


def test_icc_is_near_zero_when_targets_do_not_differ() -> None:
    """Pure measurement noise, no target signal: ICC near 0."""
    rng = np.random.default_rng(0)
    table = rng.normal(size=(20, 5))  # no per-row structure
    assert abs(intraclass_correlation(table).icc) < 0.2


def test_icc_reports_the_anova_dimensions() -> None:
    result = intraclass_correlation(np.arange(12.0).reshape(4, 3))
    assert result.n_targets == 4
    assert result.n_conditions == 3


@pytest.mark.parametrize(
    ("table", "match"),
    [
        (np.ones((1, 3)), "at least 2 targets"),
        (np.ones((3, 1)), "at least 2 targets"),
        (np.ones((3, 3)), "total variance"),
        (np.full((3, 3), np.nan), "must be finite"),
    ],
)
def test_icc_rejects_degenerate_tables(table, match) -> None:
    with pytest.raises(StabilityError, match=match):
        intraclass_correlation(table)


# --------------------------------------------------------------------------
# Concordance correlation
# --------------------------------------------------------------------------


def test_ccc_is_one_for_identical_vectors() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert concordance_correlation(x, x).ccc == pytest.approx(1.0)


def test_ccc_penalises_a_shift_more_than_pearson_does() -> None:
    """A constant offset leaves Pearson r at 1 but drops the CCC below it."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x + 2.0
    result = concordance_correlation(x, y)

    assert result.pearson_r == pytest.approx(1.0)
    assert result.ccc < 1.0
    assert result.ccc == pytest.approx(result.pearson_r * result.bias_correction)


def test_ccc_matches_lins_formula() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = 0.7 * x + 0.5 * rng.normal(size=40) + 1.0

    mx, my = x.mean(), y.mean()
    expected = 2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2)
    assert concordance_correlation(x, y).ccc == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        (np.ones(3), np.ones(4), "equal-length"),
        (np.ones(1), np.ones(1), "at least 2"),
        (np.full(3, np.nan), np.ones(3), "must be finite"),
        (np.ones(3), np.ones(3), "both constant"),
    ],
)
def test_ccc_rejects_bad_input(x, y, match) -> None:
    with pytest.raises(StabilityError, match=match):
        concordance_correlation(x, y)


# --------------------------------------------------------------------------
# The atlas builder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_atlas():
    phantoms = [
        generate_texture_phantom(size=(16, 16, 16), corr_length=cl, lesion=False, seed=s)
        for cl, s in [(3.0, 0), (5.0, 1), (7.0, 2)]
    ]
    conditions = [
        {},
        {"noise_sigma": 15.0, "seed": 1},
        {"psf_fwhm_mm": 3.0, "seed": 1},
    ]
    # Lesion-free phantoms use the whole volume as ROI.
    full_extractor = _whole_volume_extractor()
    return build_stability_atlas(
        phantoms,
        conditions,
        condition_labels=["reference", "noise", "blur"],
        feature_extractor=full_extractor,
        include_morphology=False,
    )


def _whole_volume_extractor():
    from rphantom.features import extract_features

    def extract(volume, mask, spacing, *, bin_width, include_morphology):
        roi = np.ones(volume.shape, dtype=bool)
        return extract_features(
            volume, roi, spacing, bin_width=bin_width, include_morphology=include_morphology
        )

    return extract


def test_atlas_has_a_reliability_for_every_common_feature(small_atlas) -> None:
    assert small_atlas.n_targets == 3
    assert small_atlas.n_conditions == 3
    assert len(small_atlas.reliabilities) > 100
    assert set(small_atlas.reliabilities) == set(small_atlas.matrices)


def test_atlas_matrices_have_the_sweep_shape(small_atlas) -> None:
    for matrix in small_atlas.matrices.values():
        assert matrix.shape == (3, 3)


def test_atlas_iccs_match_a_direct_computation(small_atlas) -> None:
    for tag, reliability in small_atlas.reliabilities.items():
        matrix = small_atlas.matrices[tag]
        if float(np.var(matrix)) == 0.0:
            continue
        assert reliability.icc == pytest.approx(intraclass_correlation(matrix).icc, abs=1e-9)


def test_atlas_reference_column_drives_concordance(small_atlas) -> None:
    """ccc_min is the worst concordance of a degraded column against column 0."""
    for tag, reliability in small_atlas.reliabilities.items():
        matrix = small_atlas.matrices[tag]
        if float(np.var(matrix)) == 0.0:
            continue
        cccs = []
        for j in range(1, 3):
            try:
                cccs.append(concordance_correlation(matrix[:, 0], matrix[:, j]).ccc)
            except StabilityError:
                cccs.append(float("nan"))  # a constant column, as the atlas records it
        assert reliability.ccc_min == pytest.approx(float(np.nanmin(cccs)), abs=1e-9)


def test_ranked_orders_by_icc_ascending(small_atlas) -> None:
    ranked = small_atlas.ranked(by="icc")
    iccs = [r.icc for r in ranked]
    assert iccs == sorted(iccs)


def test_morphology_shape_features_are_perfectly_stable_under_intensity_only_change() -> None:
    """Noise and blur leave the ROI geometry untouched, so shape ICC is 1."""
    phantoms = [
        generate_texture_phantom(size=(20, 20, 20), lesion_radii_mm=(r, r, r), seed=0)
        for r in (6.0, 8.0)
    ]
    conditions = [{}, {"noise_sigma": 20.0, "seed": 1}]
    atlas = build_stability_atlas(phantoms, conditions)

    # Elongation of a sphere is 1 for both phantoms and every condition; such a
    # zero-variance feature is reported as perfectly reliable.
    assert atlas.reliabilities["morph_pca_elongation"].icc == pytest.approx(1.0)


def test_atlas_rejects_too_few_phantoms_or_conditions() -> None:
    p = generate_texture_phantom(size=(16, 16, 16), seed=0)
    with pytest.raises(StabilityError, match="at least 2 phantoms"):
        build_stability_atlas([p], [{}, {}])
    with pytest.raises(StabilityError, match="at least 2 conditions"):
        build_stability_atlas([p, p], [{}])


def test_atlas_rejects_mismatched_labels() -> None:
    p = generate_texture_phantom(size=(16, 16, 16), seed=0)
    with pytest.raises(StabilityError, match="condition_labels must match"):
        build_stability_atlas([p, p], [{}, {}], condition_labels=["only-one"])
