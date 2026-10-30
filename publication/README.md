# Paper 7 — Temporal Response Identifiability

## Research question

Can temporal delay be distinguished from response
attenuation in sparsely sampled transcriptomic
time-course experiments?

## Scientific status

Exploratory research. Not a validated clinical
diagnostic tool.

The real-data comparisons in v25 did not show
an improvement from adding a delay parameter.
Reference ensembling in v28-v30 did not provide
a clear advantage over the simpler fixed-reference
benchmark.

Simulation results describe the specified
synthetic experiments, not clinical sensitivity
or specificity.

## Contents

- code/: four selected analysis scripts
- tables/: five main aggregate result tables
- figures/: one benchmark comparison figure

## Key data sources

- GSE243039
- GSE246591
- GSE127918

Original public datasets are available at:
https://www.ncbi.nlm.nih.gov/geo/

## Reproducibility status

These scripts document selected analysis stages,
not a fully standalone end-to-end pipeline.

Some scripts depend on outputs of preceding
stages and on locally obtained GEO data.
Those dependencies must be reconstructed and
documented before claiming full reproducibility.

No raw transcriptomic matrices, personal logs,
large intermediate files or patient-level
clinical inferences are included.

## Planned extension

DelayScope: identifiability-aware design of
informative temporal sampling schedules.

The application and its experimental evaluation
are not yet part of this release.
