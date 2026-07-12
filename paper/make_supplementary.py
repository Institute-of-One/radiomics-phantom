"""Generate the machine-readable supplementary tables, deterministically.

Writes:
  paper/supplementary/feature_inventory.csv  -- every IBSI-1 benchmark output,
      whether it is used in the stability atlas, and why not.
  paper/supplementary/stability_atlas.csv    -- per-feature ICC/CCC for the atlas,
      with a status flag distinguishing measured agreement from constant features.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rphantom import (  # noqa: E402
    build_stability_atlas,
    extract_features,
    generate_texture_phantom,
)
from rphantom.stability import concordance_correlation  # noqa: E402
from tests.ibsi_reference import REFERENCE  # noqa: E402

OUT = Path(__file__).resolve().parent / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_NAMES = {
    "stat": "intensity statistics",
    "ih": "intensity histogram",
    "ivh": "intensity-volume histogram",
    "morph": "morphology",
    "loc": "local intensity",
    "cm": "grey-level co-occurrence",
    "rlm": "grey-level run length",
    "szm": "grey-level size zone",
    "dzm": "grey-level distance zone",
    "ngt": "neighbourhood grey-tone difference",
    "ngl": "neighbouring grey-level dependence",
}
DIRECTIONAL = {"cm", "rlm"}  # six aggregations; atlas uses 3D_comb
ZONELIKE = {"szm", "dzm", "ngt", "ngl"}  # three aggregations; atlas uses 3D
EXCLUDED_FAMILIES = {"morph", "loc", "ivh"}


def split_tag(tag: str) -> tuple[str, str, str]:
    """Return (family_prefix, feature_name, aggregation) for an IBSI tag."""
    fam = tag.split("_")[0]
    for suffix in (
        "_3D_comb",
        "_3D_avg",
        "_2_5D_comb",
        "_2_5D_avg",
        "_2D_comb",
        "_2D_avg",
        "_3D",
        "_2_5D",
        "_2D",
    ):
        if tag.endswith(suffix):
            return fam, tag[: -len(suffix)], suffix[1:]
    return fam, tag, ""


def atlas_feature_tags() -> set[str]:
    """The 136 feature tags the stability atlas actually computes."""
    p = generate_texture_phantom(size=(16, 16, 16), lesion=False, seed=0)
    roi = np.ones(p.shape, dtype=bool)
    return set(extract_features(p.volume, roi, p.spacing, include_morphology=False))


def write_feature_inventory(atlas_tags: set[str]) -> None:
    rows = []
    for tag in sorted(REFERENCE):
        fam, name, agg = split_tag(tag)
        in_atlas = tag in atlas_tags
        reason = ""
        if not in_atlas:
            if fam in EXCLUDED_FAMILIES:
                reason = "family excluded from atlas (needs lesion mask; not on whole-volume ROI)"
            elif fam in DIRECTIONAL and agg != "3D_comb":
                reason = "non-primary aggregation (atlas uses 3D_comb for directional families)"
            elif fam in ZONELIKE and agg != "3D":
                reason = "non-primary aggregation (atlas uses 3D for zone/neighbourhood families)"
        rows.append(
            {
                "feature_family": FAMILY_NAMES.get(fam, fam),
                "feature_name": name,
                "aggregation": agg or "none",
                "included_in_ibsi_benchmark": 1,
                "included_in_stability_atlas": int(in_atlas),
                "exclusion_reason": reason,
            }
        )
    path = OUT / "feature_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_atlas = sum(r["included_in_stability_atlas"] for r in rows)
    print(f"wrote {path} ({len(rows)} benchmark features, {n_atlas} in atlas)")


def write_stability_atlas() -> None:
    phantoms = [
        generate_texture_phantom(size=(24, 24, 24), corr_length=cl, lesion=False, seed=s)
        for cl, s in [(3.0, 0), (4.5, 1), (6.0, 2), (7.5, 3), (9.0, 4)]
    ]
    conditions = [
        {},
        {"noise_sigma": 15.0, "seed": 1},
        {"noise_sigma": 30.0, "seed": 1},
        {"psf_fwhm_mm": 3.0, "seed": 1},
        {"noise_sigma": 20.0, "psf_fwhm_mm": 2.0, "seed": 1},
    ]

    def whole(volume, mask, spacing, *, bin_width, include_morphology):
        roi = np.ones(volume.shape, dtype=bool)
        return extract_features(
            volume, roi, spacing, bin_width=bin_width, include_morphology=include_morphology
        )

    atlas = build_stability_atlas(
        phantoms, conditions, feature_extractor=whole, include_morphology=False
    )

    rows = []
    for tag, rel in atlas.reliabilities.items():
        matrix = atlas.matrices[tag]
        constant = float(np.var(matrix)) == 0.0
        ref = matrix[:, 0]
        cccs = []
        if not constant:
            for j in range(1, matrix.shape[1]):
                try:
                    cccs.append(concordance_correlation(ref, matrix[:, j]).ccc)
                except Exception:  # noqa: BLE001
                    pass
        rows.append(
            {
                "feature_name": tag,
                "feature_family": FAMILY_NAMES.get(split_tag(tag)[0], split_tag(tag)[0]),
                "ICC_2_1": round(float(rel.icc), 4),
                "median_CCC": round(float(np.median(cccs)), 4) if cccs else "",
                "minimum_value": round(float(matrix.min()), 6),
                "maximum_value": round(float(matrix.max()), 6),
                "status": "constant (ICC forced to 1.0; not measured agreement)"
                if constant
                else "measured",
            }
        )
    rows.sort(key=lambda r: (r["status"].startswith("constant"), -r["ICC_2_1"]))
    path = OUT / "stability_atlas.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_const = sum(r["status"].startswith("constant") for r in rows)
    print(f"wrote {path} ({len(rows)} atlas features, {n_const} constant)")


if __name__ == "__main__":
    tags = atlas_feature_tags()
    write_feature_inventory(tags)
    write_stability_atlas()
