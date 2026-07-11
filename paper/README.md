# Manuscript (paper/)

The preprint manuscript and its reproducible figures, version-controlled with the
code as part of the research package. Structure follows the IORN house convention
(see IORN-001 / ctp-core).

| File | Role |
|---|---|
| `radiomics-phantom_medRxiv.md` | **Canonical manuscript** (pandoc-style Markdown, single source of truth) |
| `radiomics-phantom_medRxiv.pdf` | Compiled manuscript, figures embedded — the file uploaded to medRxiv |
| `medRxiv_submission_kit.md` | Copy-paste-ready fields and checklist for the medRxiv form |
| `make_figures.py` | Regenerates every figure and number from the library (deterministic) |
| `build_pdf.py` | Renders the Markdown manuscript to PDF without pandoc/LaTeX |
| `figures/` | Figure PNGs and `results.json` (every number quoted in the text) |

## Reproduce

```bash
python -m pip install -e ".[dev]" reportlab   # reportlab only needed for build_pdf.py
python paper/make_figures.py      # -> figures/*.png and figures/results.json
python paper/build_pdf.py         # -> radiomics-phantom_medRxiv.pdf
# or, when pandoc + a LaTeX engine are available:
# pandoc paper/radiomics-phantom_medRxiv.md -o paper/radiomics-phantom_medRxiv.pdf
```

Every quoted value is produced by `make_figures.py` and written to
`figures/results.json`, so the text and figures cannot diverge. The manuscript is
released under CC BY 4.0 (the code under MIT).

## Workflow

This package follows the shared IORN publication workflow: **GitHub → Zenodo →
medRxiv** (peer-reviewed venue thereafter). See
`IoO/docs/IoO_OpenCore_Publication_Runbook.md` for the full runbook.
