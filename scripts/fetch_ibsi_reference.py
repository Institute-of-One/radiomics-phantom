"""Regenerate ``tests/ibsi_reference.py`` from the authoritative IBSI sources.

This script exists so that the test fixtures are *provenanced* rather than
transcribed.  It downloads the IBSI-1 digital phantom (NIfTI) and the published
reference-value table (xlsx), and writes a plain Python module containing the
phantom, its ROI mask, and the reference values for the feature families that
:mod:`rphantom.features` implements.

Run it only when the implemented feature families change::

    python scripts/fetch_ibsi_reference.py

Requires network access.  Uses the standard library plus ``numpy``: no
``nibabel`` (a 348-byte NIfTI-1 header is easy enough to read) and no
``openpyxl`` (an xlsx file is a zip of XML).

Sources
-------
Phantom
    https://github.com/theibsi/data_sets, ``ibsi_1_digital_phantom``, CC BY 4.0.
Reference values
    https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx
"""

from __future__ import annotations

import gzip
import re
import struct
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

PHANTOM_BASE = (
    "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_1_digital_phantom/nifti/"
)
REFERENCE_URL = "https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx"
OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "ibsi_reference.py"

#: Only the families implemented by rphantom.features.  IBSI publishes a few
#: morphology values (OMBB and MVEE densities) as "not standardised"; those
#: cells are empty and so are skipped naturally.
TAG_PREFIXES = (
    "stat_",
    "ih_",
    "cm_",
    "rlm_",
    "szm_",
    "dzm_",
    "ngt_",
    "ngl_",
    "morph_",
    "loc_",
    "ivh_",
)

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

_NIFTI_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "rphantom/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def read_nifti(raw: bytes) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read a NIfTI-1 volume and its voxel spacing, without ``nibabel``."""
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw

    endian = "<" if struct.unpack("<i", data[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", data[:4])[0] != 348:
        raise ValueError("not a NIfTI-1 file: header size is not 348 bytes.")

    dim = struct.unpack(endian + "8h", data[40:56])
    datatype = struct.unpack(endian + "h", data[70:72])[0]
    pixdim = struct.unpack(endian + "8f", data[76:108])
    vox_offset = int(struct.unpack(endian + "f", data[108:112])[0])

    if datatype not in _NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}.")

    n_dims = dim[0]
    shape = dim[1 : 1 + n_dims]
    values = np.frombuffer(
        data, dtype=endian + _NIFTI_DTYPES[datatype], count=int(np.prod(shape)), offset=vox_offset
    )
    # NIfTI stores voxels in Fortran order, fastest-varying axis first (x, y, z).
    return values.reshape(shape, order="F"), pixdim[1 : 1 + n_dims]


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return index - 1


def read_xlsx_sheet(raw: bytes, sheet_name: str) -> list[list[str]]:
    """Read one worksheet of an xlsx file into rows of strings, without ``openpyxl``."""
    archive = zipfile.ZipFile(BytesIO(raw))

    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = {
        rel.get("Id"): rel.get("Target")
        for rel in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    }
    target = None
    for sheet in workbook.find(_XLSX_NS + "sheets"):
        if sheet.get("name") == sheet_name:
            target = relationships[sheet.get(_REL_NS + "id")].lstrip("/")
    if target is None:
        raise ValueError(f"sheet {sheet_name!r} not found in workbook.")
    if not target.startswith("xl/"):
        target = "xl/" + target

    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        table = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.iter(_XLSX_NS + "t")) for item in table]

    rows = []
    for row in ET.fromstring(archive.read(target)).iter(_XLSX_NS + "row"):
        cells: dict[int, str] = {}
        for cell in row.iter(_XLSX_NS + "c"):
            value = cell.find(_XLSX_NS + "v")
            if cell.get("t") == "s":
                text = shared[int(value.text)] if value is not None else ""
            else:
                text = value.text if value is not None and value.text else ""
            cells[_column_index(cell.get("r"))] = text
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def _format_volume(array: np.ndarray) -> str:
    slices = []
    for z in range(array.shape[0]):
        rows = "".join(
            "        [" + ", ".join(str(int(v)) for v in array[z, y]) + "],\n"
            for y in range(array.shape[1])
        )
        slices.append("    [\n" + rows + "    ],")
    return "[\n" + "\n".join(slices) + "\n]"


def main() -> None:
    """Download the sources and rewrite the fixture module."""
    phantom, spacing = read_nifti(_download(PHANTOM_BASE + "image/phantom.nii.gz"))
    mask, _ = read_nifti(_download(PHANTOM_BASE + "mask/mask.nii.gz"))

    # NIfTI gives (x, y, z); rphantom works in (z, y, x).
    phantom = np.transpose(phantom, (2, 1, 0))
    mask = np.transpose(mask, (2, 1, 0)).astype(bool)
    spacing_zyx = tuple(float(s) for s in reversed(spacing))

    rows = read_xlsx_sheet(_download(REFERENCE_URL), "digital phantom")
    header = rows[0]
    tag_col, value_col = header.index("tag"), header.index("reference value")

    reference: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) <= tag_col or not row[value_col]:
            continue
        tag = row[tag_col].strip()
        if tag.startswith(TAG_PREFIXES):
            reference[tag] = float(row[value_col])

    body = f'''"""IBSI digital phantom and its reference feature values.

Test fixtures, generated by ``scripts/fetch_ibsi_reference.py``; do not edit by hand.

The digital phantom is a 5 x 4 x 4 (x, y, z) volume with 2 mm isotropic voxels,
published by the Image Biomarker Standardisation Initiative for verifying
radiomics implementations.  Six of its 80 voxels lie outside the ROI.  Grey
levels 2 and 5 are absent from the ROI, so texture matrices span 6 grey levels
of which two are empty.

Reference values carry a tolerance of zero: an implementation must reproduce
them exactly, to the three significant digits at which IBSI publishes them.

Sources
-------
Phantom
    https://github.com/theibsi/data_sets (ibsi_1_digital_phantom), CC BY 4.0.
Reference values
    https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx

Zwanenburg et al., Radiology 295(2):328-338, 2020.
https://doi.org/10.1148/radiol.2020191145
"""

from __future__ import annotations

import numpy as np

#: Phantom intensities, indexed (z, y, x).
PHANTOM = np.array({_format_volume(phantom)}, dtype=np.float64)

#: ROI intensity mask, indexed (z, y, x).
MASK = np.array({_format_volume(mask.astype(int))}, dtype=bool)

#: Voxel spacing in mm, (dz, dy, dx).
SPACING = {spacing_zyx}

#: IBSI feature tag -> published reference value for the digital phantom.
REFERENCE: dict[str, float] = {{
'''
    for tag in sorted(reference, key=lambda t: (t.split("_")[0], t)):
        body += f"    {tag!r}: {reference[tag]!r},\n"
    body += "}\n"

    OUTPUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(reference)} reference values, ROI of {int(mask.sum())} voxels)")


if __name__ == "__main__":
    main()
