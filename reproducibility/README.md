# Paper 7 Reproducibility Protocol

Run from the repository root:

```bash
python reproducibility/replay.py
```

The complete development replay executes:

1. V112 — GSE194200 source download
2. V113 — raw counts and baselines
3. V114 — drug-transfer and DMSO stress tests
4. V115 — paired-vehicle models
5. V116 — original freeze function only
6. V116c — eLife publisher-source download
7. V117 — second-study evaluation
8. V118 — evidence audit
9. V119 — noncircular DMSO analysis
10. V120 — post-hoc two-stage audit
11. Verification of recorded numerators

The original broken GSE149428 GEO downloader in V116
is bypassed, but its original freeze function is run
and the recorded SHA256 is checked.

Expected development numerators:

- V113 drug-aware baseline: 12/24
- V115 ridge same-drug: 21/24
- V115 ridge cross-drug: 18/24
- V117 fixed same-drug: 78/90
- V117 ridge cross-drug: 96/180 dependent predictions
- V119 matched-reference timing: 84/90
- V120 joint success: 79/90
- V120 DMSO rejected: 29/30

These values reproduce prior computations.
They are NOT prospective independent validation
of V120 or of biological onset detection.

Data are downloaded from public sources and stored
outside Git. Source availability is an external dependency.

No original research scripts are modified by the replay.
