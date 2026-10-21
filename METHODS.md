# Paper 7 — Methods draft / analysis contract

## Primary scientific question

In stimulated endometrial stromal cells, can delayed-onset transcriptional responses be distinguished from reduced response amplitudes at the observed sampling times, without assuming each repeated measurement is an independent donor?

## Data and scope

Primary discovery: GSE243039, EPC treatment, four control and four endometriosis donor series measured at days 0, 2, 4, 6 and 8. Manual sample/donor review is mandatory: original published control analyses may not retain all four controls. GSE246591 is a *separate* early-time reference (different stimulus); GSE35287 is an independent but sparse stimulation series; GSE145701 is a 14-day endpoint, NOT a trajectory. No pooling of assays or conditions during phase 1.

## Pre-case-analysis choices

Select externally justified gene programs (>=3 measured genes; gene panels and rationale frozen), exclusion and QC rules, time units, expression transformations, parameter constraints, and evaluation plan before reading case curves. Freeze source files and code version in a timestamped manifest, with hashes. This is a time-stamped secondary analysis plan, not a prospective experimental preregistration.

## Expression and module scoring

For normalized nonnegative GSE243039 expression values, transform log2(x+1). For each gene g, compute mu_g and sd_g on **verified control samples only**; z_gs=(log2(x_gs+1)-mu_g)/sd_g. Module score is the arithmetic mean of its measured gene z-scores; exclude fewer than 3 measured genes. Each donor/time is one observation. Controls receive equal weight in the reference mean. Training controls are refit for held-out-control sensitivity, including normalization.

## Mechanism models

For program m, let f_m(t) be a control-only mean trajectory, represented by shape-preserving cubic interpolation (PCHIP) on observed times. We fit each donor i with:

M_amp: y_im(t)= b_im + a_im [f_m(t)-f_m(0)] + eps_im(t)

M_lag: y_im(t)= b_im + a_im [f_m(max(0,t-d_im))-f_m(0)] + eps_im(t)

with a in [0,2.5] and onset delay d in [0,2] days. Delay is nonnegative. This is *not* a global time translation: before response onset the lagged program is clamped to its day-zero reference. The fitted intercept absorbs donor baseline shifts. The lag profile is evaluated on 81 predefined grid points; at equal RSS the smaller lag is preferred. NO claim that this particular form is identified by the data without diagnostics.

Compare full-fit parameters and RMSE, plus leave-one-*timepoint*-out error for each **independent donor**. The main predictive comparison is the per-donor RMSE difference; do not treat folds as independent test subjects or report p-values based on 40 samples. Residuals beyond M_lag are described as unexplained shape variation, not automatically regulatory rewiring.

## Identifiability checks

For first-order perturbations around d=0, the local amplitude direction is u(t)=f(t)-f(0), and onset-lag direction is v(t)=-f'(t) for t>0 (0 at baseline). Remove nuisance intercept from both, scale to unit length, and report column correlation, singular values and scaled condition number; repeat with PCHIP and smoothed spline derivative estimates. Degenerate curves give infinite condition number. This is **local** and is not a structural/global identifiability proof, parameter CI, or formal hypothesis test.

## Follow-up experiment-design suggestion

Given the control-only reference and allowed candidate sampling times, fix first/last samples, search three middle timepoints, and rank by smallest singular value of the two normalized Jacobian columns. This ranks separability under a simplified equal-noise model, *not* a proven optimal laboratory schedule. Biological timing feasibility, response noise, costs, batch effects and interpolation sensitivity must be checked before recommending an actual experiment.

## Simulations

Known-normal-curve oracle simulations include no defect, amplitude reduction, lag, both, and non-time-warpable shape perturbation. Compare estimated amplitude, estimated lag and within-donor held-out-timepoint errors across repeated simulations. This validates execution but does not establish actual biological performance: simulations with noisy estimated reference curves and subject-level confounding must follow.

## Reporting and go/no-go

- Identity and independent donor counts verified; at least 3 control donors with 5 timepoints; otherwise no case fit.
- Explicitly curated modules, measured gene coverage, PCHIP/spline sensitivity and reference-LODO results disclosed.
- A delay hypothesis needs stable estimation and improvement against M_amp across donors, not just in-sample RSS (which necessarily cannot increase in the nested model).
- Any claim of new methodological contribution requires comparison against relevant functional-data-registration baselines (SITAR-style), trajectory alignment (Genes2Genes where data representation permits), and simple spline/GAM approaches; not implemented yet.
- External studies have different time grids and stimulation: verify reproducibility of **response programs**, not personal lag parameters, unless anchored cross-protocol identifiability can be justified.
- No causal claims about endometriosis, no fertility prediction, no non-invasive diagnostic or medical decision claim from these data alone.

## Version

2026-10-20: v0.1 research implementation, methods subject to revision **before examining patient profiles**.
