# RECOMB 2027 Paper 7: verified time-course geometry

**Manuscript:** *Evaluating the Identifiability of Temporal Delay and Response Attenuation in Endometrial Decidualization*.

**Main estimand:** descriptive model-conditional geometric separation of a hypothetical shifted trajectory and its affine baseline/amplitude re-expression on the recorded grid. It is **not** a biological onset or known-lag estimator.

## Public data

- GEO accession: [GSE246591](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246591)
- File: `GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz`
- Required input SHA-256: `1d340b52b0eac564f73fe892ab28560748a122a7d3d51e21aa3c13e8a667cb08`
- 40 samples: 4 cultures (`S169`, `S170`, `S506`, `S508`) × times (`0`, `3`, `6`, `9`, `12`, `18`, `24`, `36`, `48`, `96` hours)
- 60,676 rows are *source features* (source header `Ensemvl.ID`), not proven unique mapped genes.

## Runtime

Author's verified runtime: Python 3.12.15; NumPy 2.2.6; pandas 2.3.3. Matplotlib supports figure rebuilding. `requirements.txt` gives package versions. Source files, data downloads, analytical outputs and generation code remain distinct.

## Run, no flags

From this directory (the scripts internally resolve their repository root), use:

```bash
python3 scripts/compute_gse246591_delta.py
python3 scripts/p7_selection_identifiability_audit.py
python3 scripts/p7_dual_component_audit.py
python3 scripts/plot_manuscript_figures.py
```

Alternatively `bash run_reproduction.sh` uses the Paper 7 Python environment if present on the research Mac.

The first script obtains the original GEO source when not locally cached, verifies its SHA and 40 samples, and creates `evidence/gse246591_delta_summary.csv` plus `analysis_output/gse246591_delta_gene_level.csv.gz`. Subsequent scripts audit selection and the two geometric components, without label tuning or new inferential claims. The plotting script recreates Figures 3 and 4 using verified aggregate summaries. Source-identity checks will stop on mismatches.

## Outputs and claims

- Figure 3: the 16 culture-by-candidate-shift median conditional distances (3, 6, 12, 24 hours).
- Figure 4: culture-level medians and descriptive rates among near-zero residual records. Points are **not** independent gene-level scatter.
- Supplementary Tables S1-S3: primary geometry, signal--shape and candidate-dependent eligibility.
- Thresholds 0.5 (dynamic range), 0.05 (near-zero residual) and 0.20 (display-only R/S) are exploratory cutoffs, not p-values.
- The range decomposition is deterministic; empirical shifted values use piecewise-linear interpolation and a flat pre-zero rule.
- No observed ground truth of molecular lag, no within-culture same-hour technical/biological replication, no empirical noise-calibrated confidence interval.

## Related clinical context

GSE94036, GSE51981 and GSE6364 provide distinct perturbation and cross-sectional score analyses. Their statistical results are **not** part of the GSE246591 temporal benchmark. Independent reproduction of the original PGR score additionally requires the historical frozen mapped weight file (SHA256 `2de4c185923ef5d13fdfcfec061d66736e03d24cbbb9157e5b9b1de3b51690e0`), which has not been included here.

## Code availability and limitations

This README documents the exact manuscript workflow; prior prototype OnsetAudit code elsewhere in the repository does not validate these analyses. Please cite the corresponding preprint/manuscript once available and retain the GEO accession and source SHA in downstream use.
