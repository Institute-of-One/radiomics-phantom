# radiomics-phantom documentation

An open, fully synthetic research kernel for studying the stability of radiomics
features under acquisition variation. No patient data, no DICOM, no deep
learning — every volume is generated from a seed and every feature is computed
from first principles.

## Contents

- **[Architecture](architecture.md)** — the module map, the data flow from
  phantom to normalised feature, and the load-bearing design decisions.
- **[Methodology](methodology.md)** — the science: how the phantom is built, how
  the feature core is validated against IBSI, and what the stability and
  normalisation statistics mean.
- **[API reference](api.md)** — the public functions and dataclasses of each
  module, grouped by pipeline stage.

## Where to start

If you want to *use* the library, the [README](../README.md) quick-start is the
fastest path. If you want to *understand or extend* it, read
[Architecture](architecture.md) then [Methodology](methodology.md). To contribute,
see [CONTRIBUTING](../CONTRIBUTING.md).

## The pipeline in one line

```
generate_texture_phantom → simulate_acquisition → extract_features → build_stability_atlas → normalise_feature
```

Each arrow is a module boundary; each stage is deterministic and validated. The
[Phantom Studio](../apps/phantom_studio.py) GUI drives the first three stages
interactively.
