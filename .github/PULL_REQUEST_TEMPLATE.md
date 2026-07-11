## Summary

What this PR changes, in one or two sentences.

## What it relies on, and how it was verified

State the physical or mathematical fact the change depends on, and how you checked
it — an IBSI reference value, an analytic case, a mutation test.

## Checklist

- [ ] `python -m pytest` passes
- [ ] `ruff check .` and `ruff format --check .` clean
- [ ] `mypy rphantom` clean
- [ ] New/changed behaviour has tests (contract + reference where applicable)
- [ ] Degenerate inputs raise a specific exception, never return `nan`
- [ ] Stochastic paths take a `seed` and are bit-reproducible
- [ ] `CHANGELOG.md` updated under `Unreleased`
- [ ] If feature families changed: `scripts/fetch_ibsi_reference.py` re-run and
      `tests/ibsi_reference.py` regenerated
