"""Tests for the Phantom Studio compute core.

The GUI itself needs a display, but its logic does not.  These tests pin the
headless core: it builds a reference/degraded pair, tabulates feature changes,
stays deterministic, and reports rather than raises when a degraded ROI is
degenerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
from studio_core import (  # noqa: E402
    STUDIO_FEATURES,
    AcquisitionParams,
    PhantomParams,
    compute_studio_result,
    orthogonal_slices,
)


def test_identity_acquisition_leaves_features_unchanged() -> None:
    result = compute_studio_result(PhantomParams(size=24, seed=0), AcquisitionParams())

    assert result.error is None
    assert result.feature_rows
    for _tag, ref, deg, change in result.feature_rows:
        assert deg == pytest.approx(ref)
        assert change == pytest.approx(0.0, abs=1e-9)


def test_noise_moves_features_and_is_reported_as_change() -> None:
    result = compute_studio_result(
        PhantomParams(size=32, seed=0), AcquisitionParams(noise_sigma=30.0, seed=1)
    )
    changes = {tag: change for tag, _, _, change in result.feature_rows}

    # Noise inflates intensity variance and GLCM contrast substantially.
    assert changes["stat_var"] > 20.0
    assert changes["cm_contrast_3D_comb"] > 20.0


def test_result_carries_both_volumes_and_masks() -> None:
    result = compute_studio_result(
        PhantomParams(size=24, seed=0), AcquisitionParams(resample_mm=2.0, seed=1)
    )
    assert result.reference_volume.shape == (24, 24, 24)
    assert result.degraded_volume.shape == (12, 12, 12)  # 2 mm resample halves the grid
    assert result.degraded_mask.shape == result.degraded_volume.shape
    assert result.reference_spacing == (1.0, 1.0, 1.0)
    assert result.degraded_spacing == (2.0, 2.0, 2.0)


def test_studio_result_is_deterministic() -> None:
    params = (PhantomParams(size=24, seed=3), AcquisitionParams(noise_sigma=15.0, seed=2))
    a = compute_studio_result(*params)
    b = compute_studio_result(*params)

    assert np.array_equal(a.degraded_volume, b.degraded_volume)
    assert a.feature_rows == b.feature_rows


def test_every_studio_feature_is_present_for_a_normal_phantom() -> None:
    result = compute_studio_result(PhantomParams(size=32, seed=0), AcquisitionParams())
    tags = {row[0] for row in result.feature_rows}
    assert tags == set(STUDIO_FEATURES)


def test_lesion_free_phantom_uses_the_whole_volume() -> None:
    """A lesion-free phantom has an empty mask; the core falls back to the full ROI."""
    result = compute_studio_result(
        PhantomParams(size=24, lesion=False, seed=0), AcquisitionParams(noise_sigma=10.0)
    )
    assert result.error is None
    assert result.feature_rows
    assert not result.reference_mask.any()


def test_slice_indices_are_central() -> None:
    result = compute_studio_result(PhantomParams(size=32, seed=0), AcquisitionParams())
    assert result.reference_slice_index == 16
    assert result.degraded_slice_index == 16


# --------------------------------------------------------------------------
# 3D orthogonal slicing and the specification summary
# --------------------------------------------------------------------------


def test_orthogonal_slices_are_the_three_planes() -> None:
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    planes = orthogonal_slices(volume, fraction=0.5)

    assert planes["z"] == 2 and planes["y"] == 2 and planes["x"] == 3
    assert planes["axial"].shape == (5, 6)  # (y, x)
    assert planes["coronal"].shape == (4, 6)  # (z, x)
    assert planes["sagittal"].shape == (4, 5)  # (z, y)
    assert np.array_equal(planes["axial"], volume[2, :, :])
    assert np.array_equal(planes["coronal"], volume[:, 2, :])
    assert np.array_equal(planes["sagittal"], volume[:, :, 3])


@pytest.mark.parametrize(("fraction", "expected_z"), [(0.0, 0), (0.5, 5), (1.0, 10)])
def test_slice_fraction_maps_across_the_stack(fraction, expected_z) -> None:
    volume = np.zeros((11, 4, 4))
    assert orthogonal_slices(volume, fraction)["z"] == expected_z


def test_orthogonal_slices_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        orthogonal_slices(np.zeros((4, 4)))


def test_spec_lines_report_the_phantom_geometry() -> None:
    result = compute_studio_result(
        PhantomParams(size=32, corr_length=6.0, anisotropy_z=2.0, seed=0),
        AcquisitionParams(),
    )
    spec = dict(result.spec_lines())

    assert spec["Volume"] == "32x32x32 voxels (z, y, x)"
    assert spec["Correlation length"] == "z=12  y=6  x=6 mm"  # z is 2x with anisotropy 2
    assert "Lesion volume" in spec
    assert spec["Seed"] == "0"


def test_spec_lines_flag_a_resampled_degraded_grid() -> None:
    result = compute_studio_result(
        PhantomParams(size=32, seed=0), AcquisitionParams(resample_mm=2.0)
    )
    spec = dict(result.spec_lines())
    assert "Degraded grid" in spec
    assert "resampled" in spec["Degraded grid"]


def test_spec_lines_report_no_lesion_when_absent() -> None:
    result = compute_studio_result(
        PhantomParams(size=24, lesion=False, seed=0), AcquisitionParams()
    )
    spec = dict(result.spec_lines())
    assert spec["Lesion"] == "none"
