"""Phantom Studio -- an interactive prototype for radiomics feature stability.

Generate a synthetic 3D texture phantom, degrade it as a simulated scanner would,
and watch every IBSI feature move in real time.  The phantom's three-dimensionality
is shown through orthogonal axial / coronal / sagittal cuts, with a slider to
scrub through the stack; a side panel reports the phantom's exact specification.

A desktop GUI built only on the standard library's tkinter plus matplotlib -- no
new runtime dependency.  Run it with::

    python apps/phantom_studio.py

The compute logic lives in :mod:`apps.studio_core` and is tested headlessly; this
file is only the view.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_core import (  # noqa: E402
    AcquisitionParams,
    PhantomParams,
    compute_studio_result,
    orthogonal_slices,
)

# A control specification: (attribute, label, from, to, resolution).
_PHANTOM_CONTROLS = [
    ("size", "Volume size (voxels)", 24, 64, 8),
    ("corr_length", "Correlation length (mm)", 1.0, 15.0, 0.5),
    ("anisotropy_z", "Z anisotropy", 1.0, 4.0, 0.5),
    ("hu_sd", "Texture SD (HU)", 5.0, 60.0, 5.0),
    ("seed", "Phantom seed", 0, 20, 1),
]
_ACQUISITION_CONTROLS = [
    ("psf_fwhm_mm", "In-plane blur FWHM (mm)", 0.0, 8.0, 0.5),
    ("slice_fwhm_mm", "Slice profile FWHM (mm)", 0.0, 8.0, 0.5),
    ("noise_sigma", "Noise sigma (HU)", 0.0, 60.0, 2.0),
    ("noise_correlation_mm", "Noise correlation (mm)", 0.0, 5.0, 0.5),
    ("dose", "Relative dose", 0.25, 8.0, 0.25),
    ("quantise_step", "Quantise step (HU)", 0.0, 25.0, 1.0),
    ("resample_mm", "Resample to (mm, 0=off)", 0.0, 4.0, 0.5),
]

_PLANES = ("axial", "coronal", "sagittal")
_PLANE_TITLES = {"axial": "Axial (y,x)", "coronal": "Coronal (z,x)", "sagittal": "Sagittal (z,y)"}
_LESION_COLOUR = "#e74c3c"
_MARKER_COLOUR = "#3498db"


class PhantomStudio(ttk.Frame):
    """The Phantom Studio main window."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=8)
        self.grid(sticky="nsew")
        master.title("Phantom Studio -- 3D radiomics feature stability")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._vars: dict[str, tk.DoubleVar] = {}
        self._pending: str | None = None
        self._result = None
        self._slice_fraction = tk.DoubleVar(value=0.5)

        self._build_controls()
        self._build_canvas()
        self._build_info_panel()
        self._build_feature_table()
        self.update_idletasks()
        self._recompute()

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> None:
        panel = ttk.Frame(self)
        panel.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 8))

        ttk.Label(panel, text="Phantom", font=("", 11, "bold")).pack(anchor="w", pady=(0, 2))
        for attr, label, lo, hi, step in _PHANTOM_CONTROLS:
            self._add_slider(panel, attr, label, lo, hi, step, getattr(PhantomParams(), attr))

        self._lesion = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel, text="Embed lesion", variable=self._lesion, command=self._recompute
        ).pack(anchor="w", pady=(2, 6))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(panel, text="Acquisition", font=("", 11, "bold")).pack(anchor="w", pady=(0, 2))
        for attr, label, lo, hi, step in _ACQUISITION_CONTROLS:
            self._add_slider(panel, attr, label, lo, hi, step, getattr(AcquisitionParams(), attr))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(panel, text="Reset", command=self._reset).pack(fill="x", pady=(2, 0))
        self._status = ttk.Label(panel, text="", foreground="#a00", wraplength=190)
        self._status.pack(anchor="w", pady=(6, 0))

    def _add_slider(self, parent, attr, label, lo, hi, step, initial) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        var = tk.DoubleVar(value=float(initial))
        self._vars[attr] = var
        header = ttk.Frame(row)
        header.pack(fill="x")
        ttk.Label(header, text=label, font=("", 8)).pack(side="left")
        value_label = ttk.Label(header, text=f"{float(initial):g}", font=("", 8, "bold"))
        value_label.pack(side="right")

        def on_change(_event=None, v=var, lab=value_label, s=step) -> None:
            snapped = round(v.get() / s) * s
            lab.config(text=f"{snapped:g}")
            self._schedule()

        ttk.Scale(row, from_=lo, to=hi, variable=var, command=on_change).pack(fill="x")

    def _build_canvas(self) -> None:
        wrapper = ttk.Frame(self)
        wrapper.grid(row=0, column=1, sticky="nsew")
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)

        self._figure = Figure(figsize=(8.6, 5.4), dpi=100)
        self._axes = {}
        for r, band in enumerate(("Reference", "Degraded")):
            for c, plane in enumerate(_PLANES):
                ax = self._figure.add_subplot(2, 3, r * 3 + c + 1)
                self._axes[(band, plane)] = ax
        self._figure.subplots_adjust(
            left=0.03, right=0.99, top=0.92, bottom=0.04, wspace=0.08, hspace=0.18
        )
        self._canvas = FigureCanvasTkAgg(self._figure, master=wrapper)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        slider_row = ttk.Frame(wrapper)
        slider_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        slider_row.columnconfigure(1, weight=1)
        ttk.Label(slider_row, text="Axial slice (z)", font=("", 8)).grid(row=0, column=0)
        ttk.Scale(
            slider_row, from_=0.0, to=1.0, variable=self._slice_fraction, command=self._on_scrub
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self._slice_label = ttk.Label(slider_row, text="", font=("", 8, "bold"), width=10)
        self._slice_label.grid(row=0, column=2)

    def _build_info_panel(self) -> None:
        panel = ttk.Frame(self, padding=(8, 0, 0, 0))
        panel.grid(row=0, column=2, rowspan=2, sticky="ns", padx=(8, 0))
        ttk.Label(panel, text="Phantom specification", font=("", 11, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self._spec = ttk.Frame(panel)
        self._spec.pack(anchor="w", fill="x")
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)
        legend = ttk.Frame(panel)
        legend.pack(anchor="w")
        ttk.Label(legend, text="— lesion mask", foreground=_LESION_COLOUR, font=("", 8)).pack(
            anchor="w"
        )
        ttk.Label(legend, text="— current axial z", foreground=_MARKER_COLOUR, font=("", 8)).pack(
            anchor="w"
        )
        ttk.Label(
            panel,
            text=(
                "Coronal and sagittal cuts reveal the 3rd dimension: raise Z anisotropy\n"
                "to see the texture stretch vertically. Drag the slice slider to scrub\n"
                "through the stack; the blue line marks that z on the side views."
            ),
            font=("", 8),
            foreground="#555",
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _build_feature_table(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=0)

        columns = ("feature", "reference", "degraded", "change")
        self._table = ttk.Treeview(frame, columns=columns, show="headings", height=7)
        headings = {
            "feature": "IBSI feature (3D)",
            "reference": "Reference",
            "degraded": "Degraded",
            "change": "Change %",
        }
        widths = {"feature": 260, "reference": 130, "degraded": 130, "change": 110}
        for col in columns:
            self._table.heading(col, text=headings[col])
            self._table.column(col, width=widths[col], anchor="e" if col != "feature" else "w")
        self._table.grid(row=0, column=0, sticky="nsew")
        self._table.tag_configure("big", foreground="#c0392b")
        self._table.tag_configure("small", foreground="#7f8c8d")

    # -- interaction --------------------------------------------------------

    def _schedule(self) -> None:
        """Debounce slider drags into one recompute."""
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(140, self._recompute)

    def _on_scrub(self, _event=None) -> None:
        """Slice slider moved: only redraw, the volumes are already computed."""
        if self._result is not None:
            self._draw(self._result)

    def _phantom_params(self) -> PhantomParams:
        return PhantomParams(
            size=int(round(self._vars["size"].get())),
            corr_length=self._vars["corr_length"].get(),
            anisotropy_z=self._vars["anisotropy_z"].get(),
            hu_sd=self._vars["hu_sd"].get(),
            lesion=self._lesion.get(),
            seed=int(round(self._vars["seed"].get())),
        )

    def _acquisition_params(self) -> AcquisitionParams:
        return AcquisitionParams(
            psf_fwhm_mm=self._vars["psf_fwhm_mm"].get(),
            slice_fwhm_mm=self._vars["slice_fwhm_mm"].get(),
            noise_sigma=self._vars["noise_sigma"].get(),
            noise_correlation_mm=self._vars["noise_correlation_mm"].get(),
            dose=max(self._vars["dose"].get(), 0.25),
            quantise_step=self._vars["quantise_step"].get(),
            resample_mm=self._vars["resample_mm"].get(),
            seed=1,
        )

    def _recompute(self) -> None:
        self._pending = None
        try:
            result = compute_studio_result(self._phantom_params(), self._acquisition_params())
        except Exception as exc:  # noqa: BLE001 -- show, never crash the UI
            self._status.config(text=f"{type(exc).__name__}: {exc}")
            return
        self._result = result
        self._status.config(text=result.error or "")
        self._draw(result)
        self._fill_table(result)
        self._fill_spec(result)

    def _draw(self, result) -> None:
        fraction = self._slice_fraction.get()
        vmin = float(result.reference_volume.min())
        vmax = float(result.reference_volume.max())

        bands = (
            ("Reference", result.reference_volume, result.reference_mask),
            ("Degraded", result.degraded_volume, result.degraded_mask),
        )
        z_ref = None
        for band, volume, mask in bands:
            planes = orthogonal_slices(volume, fraction)
            mask_planes = orthogonal_slices(mask, fraction) if mask.any() else None
            if band == "Reference":
                z_ref = planes["z"]
            for plane in _PLANES:
                ax = self._axes[(band, plane)]
                ax.clear()
                ax.imshow(planes[plane], cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
                if mask_planes is not None:
                    layer = mask_planes[plane]
                    if layer.any() and not layer.all():
                        ax.contour(layer, levels=[0.5], colors=_LESION_COLOUR, linewidths=1.0)
                # On the side views, z runs vertically: mark the current axial plane.
                if plane in ("coronal", "sagittal"):
                    ax.axhline(planes["z"], color=_MARKER_COLOUR, lw=1.0, ls="--")
                ax.set_xticks([])
                ax.set_yticks([])
                if band == "Reference":
                    ax.set_title(_PLANE_TITLES[plane], fontsize=9)
            self._axes[(band, "axial")].set_ylabel(band, fontsize=10)

        self._canvas.draw_idle()
        nz = result.reference_volume.shape[0]
        self._slice_label.config(text=f"z = {z_ref} / {nz - 1}")

    def _fill_table(self, result) -> None:
        self._table.delete(*self._table.get_children())
        for tag, ref_value, deg_value, change in result.feature_rows:
            if change != change:  # nan
                change_text, tag_style = "n/a", ()
            else:
                change_text = f"{change:+.1f}"
                tag_style = ("big",) if abs(change) >= 25.0 else ("small",)
            self._table.insert(
                "",
                "end",
                values=(tag, f"{ref_value:.4g}", f"{deg_value:.4g}", change_text),
                tags=tag_style,
            )

    def _fill_spec(self, result) -> None:
        for child in self._spec.winfo_children():
            child.destroy()
        for r, (label, value) in enumerate(result.spec_lines()):
            ttk.Label(self._spec, text=f"{label}:", font=("", 8)).grid(
                row=r, column=0, sticky="w", padx=(0, 6)
            )
            ttk.Label(self._spec, text=value, font=("", 8, "bold")).grid(
                row=r, column=1, sticky="w"
            )

    def _reset(self) -> None:
        for attr, _, _, _, _ in _PHANTOM_CONTROLS:
            self._vars[attr].set(float(getattr(PhantomParams(), attr)))
        for attr, _, _, _, _ in _ACQUISITION_CONTROLS:
            self._vars[attr].set(float(getattr(AcquisitionParams(), attr)))
        self._lesion.set(True)
        self._slice_fraction.set(0.5)
        self._recompute()


def main() -> None:
    """Launch the Phantom Studio window."""
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    PhantomStudio(root)
    root.minsize(1180, 720)
    root.mainloop()


if __name__ == "__main__":
    main()
