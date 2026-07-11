"""Render the central axial slice of a phantom to a PNG.

Usage
-----
    python examples/render_phantom_slice.py [--seed 0] [--out outputs/phantom_slice.png]

Produces a two-panel figure: the intensity slice with the lesion boundary
overlaid, and the lesion mask on the same slice.  Uses the Agg backend, so it
runs headless (CI, servers) without a display.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: must be set before pyplot is imported.

import matplotlib.pyplot as plt  # noqa: E402

from rphantom import generate_texture_phantom  # noqa: E402


def main() -> None:
    """Generate one phantom and save its central slice as a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/phantom_slice.png"),
        help="Output PNG path (default: outputs/phantom_slice.png).",
    )
    args = parser.parse_args()

    phantom = generate_texture_phantom(
        size=(64, 64, 64),
        spacing=(1.0, 1.0, 1.0),
        corr_length=6.0,
        anisotropy=(1.0, 1.0, 1.0),
        hu_mean=40.0,
        hu_sd=25.0,
        lesion=True,
        lesion_radii_mm=(12.0, 12.0, 12.0),
        lesion_hu_offset=60.0,
        lesion_corr_length=3.0,
        seed=args.seed,
    )

    z = phantom.shape[0] // 2
    dy, dx = phantom.spacing[1], phantom.spacing[2]
    extent = (0.0, phantom.shape[2] * dx, phantom.shape[1] * dy, 0.0)

    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    im = ax_img.imshow(phantom.volume[z], cmap="gray", extent=extent, origin="upper")
    ax_img.contour(
        phantom.mask[z],
        levels=[0.5],
        colors="tab:red",
        linewidths=1.2,
        extent=extent,
        origin="upper",
    )
    ax_img.set_title(f"Intensity, z = {z} (seed {phantom.seed})")
    fig.colorbar(im, ax=ax_img, label="HU-like", shrink=0.85)

    ax_mask.imshow(phantom.mask[z], cmap="gray", extent=extent, origin="upper")
    ax_mask.set_title("Ground-truth lesion mask")

    for ax in (ax_img, ax_mask):
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    gt = phantom.ground_truth
    print(f"Wrote {args.out}")
    print(f"  shape          : {phantom.shape} @ {phantom.spacing} mm")
    print(f"  corr_lengths   : {gt['corr_lengths_mm']} mm")
    print(f"  lesion volume  : {gt['lesion_params']['volume_mm3']:.1f} mm^3")
    print(f"  intensity range: [{phantom.volume.min():.1f}, {phantom.volume.max():.1f}] HU-like")


if __name__ == "__main__":
    main()
