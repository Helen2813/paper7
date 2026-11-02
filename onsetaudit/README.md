# OnsetAudit 0.3 — local transcriptomic timing workspace

An offline-by-default **local HTTP application** for inspecting temporal RNA-seq data. It accepts CSV/TSV, gzip-compressed CSV/TSV, and ZIPs containing one table; never uploads input to a remote service. Start from the Paper 7 repository root:

```bash
bash launch_onsetaudit.sh
```

The launcher creates `.venv-onsetaudit` on first use and installs the pinned `numpy==2.2.6`, `pandas==2.3.3` requirements only if missing. It starts a server bound to `127.0.0.1` and opens the browser. Stop with **Ctrl+C**. All uploaded originals are copied to `onsetaudit/_generated/inputs/` with SHA256-hashed filenames. The exported report ZIP is copied to `onsetaudit/_generated/reports/`. Both folders are excluded from Git.

## Input format

Metadata should contain at least sample ID, drug/treatment, and duration in hours; grouped evaluation also requires a batch/library/experiment and replicate field. Examples: `sample_column_name`, `drug`, `timepoint`, `library`, `replicate`.

An expression matrix has genes in rows and sample IDs in columns. The first column contains gene IDs (or, for GSE214965-style raw counts, the initial header omits the gene-ID label). **Every sample name must match one metadata record.** Select either raw integer counts (converted to log2(1+CPM)) or processed logCPM (used as supplied). Batch-adjusted logCPM from a published study may have preprocessing performed before a heldout split; this is noted as a limitation.

## What is implemented

- Metadata/schema and exact expression-to-sample matching
- Coverage matrix, replicate/batch inventory and low-depth warnings
- Gene time-trajectory display (observed means ± SD, no fitted onset)
- Onset schedule audit: 0 vs 6 h, 6 vs 12 h or custom onsets under explicit model restrictions
- Experimental **known-drug exposure-duration** inference with grouped holdouts and training-only controls
- Fixed `200` genes, `variance_floor=0.25`, `ridge_amplitude=0.25`, as in Paper 7 V115/V120 family
- Pooled DMSO activity gate + time-matched training-only DMSO time inference
- Vehicle-only baseline receives identical training control inputs
- Report ZIP with predictions, per-fold settings, warnings and SHA256 provenance
- Built-in **synthetic** demonstration (clearly identified as non-evidence)

## Scientific boundaries

The estimator does **not** identify biological transcription onset from a single expression profile. It estimates experimentally recorded duration of exposure where matched vehicle and training time courses exist, assuming the treatment identity is known. It is not a generalized clinically validated predictor.

The V120 hybrid was designed after inspecting GSE149428 results. The app is an **experimental implementation** and not a substitute for a locked third-study validation. On designs with just two batches, group holdout trains on only one batch; this is a documented adaptation of the V115 training protocol. Independent 95% coverage and methodological novelty are **not demonstrated**.

## Tests

```bash
python -m unittest discover -s onsetaudit/tests -v
```

## Repository hygiene

Only app code and documentation belong in Git. Original GEO matrices, uploaded experimental samples, archived ZIP reports and virtual environments are excluded. Previous frozen `paper7_v*.py` scripts remain unchanged.
