# Contributing to radiomics-phantom

Thank you for your interest. This is a research kernel meant to be **citable and
reproducible**, so the contribution bar is less about volume of code and more
about keeping the guarantees below intact. Please read this before opening a pull
request.

## Development setup

Python 3.10 or newer.

```bash
git clone https://github.com/Institute-of-One/radiomics-phantom
cd radiomics-phantom
python -m pip install -e ".[dev]"
python -m pytest          # expect the whole suite to pass
```

The IBSI reference fixtures used by the tests are committed, so the suite runs
offline. To regenerate them from the authoritative sources (only needed if the
implemented feature families change), run:

```bash
python scripts/fetch_ibsi_reference.py   # needs network access
```

## The five guarantees

Every change must preserve these. They are what make the project trustworthy.

1. **Pure, minimal dependencies.** Runtime code imports only `numpy`, `scipy`,
   `scikit-image`, `matplotlib` (and `pingouin` where declared). No PyRadiomics,
   no deep-learning frameworks, no pandas in the runtime path. Reference oracles
   (MIRP, pingouin) are permitted in tests only.
2. **IBSI correctness, at zero tolerance.** Feature code is validated against the
   IBSI digital-phantom reference values, which must be reproduced *exactly* to
   three significant digits. New feature families need a matching entry in
   `tests/test_features_ibsi.py`; `test_reference_coverage_is_declared` pins the
   boundary of what is implemented so it cannot silently drift.
3. **Determinism.** Every stochastic function takes a `seed` and, given the same
   inputs, produces a bit-identical result across runs and processes. Do not use
   `Date.now`-style or unseeded randomness.
4. **No silent failure.** A degenerate or undefined result raises a specific,
   messageful exception (`FeatureError`, `AcquisitionError`, `StabilityError`,
   `NormalizationError`) — never a silent `nan`. If a statistic is `0/0`, say so.
5. **No patient data, ever.** No DICOM, no scans, nothing derived from a human
   subject. Everything is generated from a seed.

## Code style

The house style follows the existing modules:

- Return results as frozen `@dataclass` objects with typed fields and a `to_dict()`
  keyed by official IBSI tag where applicable.
- Pure functions, type hints, and a docstring on every public function giving
  parameters, returns, and the exceptions raised.
- Arrays are indexed `(z, y, x)`; spacing is `(dz, dy, dx)` in millimetres.
- Keep comments about *why*, not *what*; match the surrounding density.

Run the linters before submitting:

```bash
ruff check .
ruff format --check .
mypy rphantom
```

## Adding a feature family (worked checklist)

1. Implement it in `rphantom/features.py` from the IBSI definition, over the
   appropriate aggregations. Reuse the shared matrix machinery where you can.
2. Add its tags to `scripts/fetch_ibsi_reference.py` (`TAG_PREFIXES`) and
   regenerate `tests/ibsi_reference.py`.
3. Extend `IMPLEMENTED_FAMILIES` and the fixture in `tests/test_features_ibsi.py`.
4. Add contract tests (degenerate ROIs, hand-solvable cases) in the relevant
   `tests/test_features_*.py`.
5. **Mutation-check your oracle.** Before trusting a first-pass "all green",
   deliberately break the implementation and confirm the reference tests catch it.
   This has repeatedly surfaced real bugs; see the module histories.
6. Export from `rphantom/__init__.py`, bump the version, update `CHANGELOG.md`.

## Pull requests

- One focused change per PR; keep the diff reviewable.
- All tests green, linters clean.
- Update `CHANGELOG.md` under `Unreleased`.
- Describe *what physical or mathematical fact* your change relies on, and how you
  verified it (reference value, analytic case, mutation test).

## Licence of contributions

Code contributions are licensed under the [MIT License](LICENSE); documentation
and figures under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). By
contributing you agree your work is released under these terms.
