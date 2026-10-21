"""Stage-1 case/control workflow gated by frozen, manually verified annotations.

No case profiles are read into the fitting workflow before the lock is validated.
On this stage, all comparisons are descriptive/predictive, not causal or clinical.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper7_pilot import (read_gmt, get_session, locate_expression, read_matrix,
                          gene_symbols_and_expression, match_columns, sha256_file)
from .data import validated_review, verify_lock, score_all_samples
from .model import make_reference, compare_models, fit_delayed, design_identifiability, suggest_sampling_times


def expression_and_samples(pilot_dir: Path, gmt: Path, review: Path, lock: Path,
                           offline: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str]:
    base_meta = pilot_dir / 'metadata' / 'GSE243039_samples.csv'
    meta = pd.read_csv(base_meta, dtype={'gsm': str})
    locked = verify_lock(lock, review, gmt, base_meta)
    reviewed = validated_review(review, expected_gsm=set(meta.gsm))
    study = 'GSE243039'
    filename = locate_expression(pilot_dir, get_session(), study, offline)
    raw_hash = sha256_file(filename)
    raw = read_matrix(filename)
    expr, _ = gene_symbols_and_expression(raw, pilot_dir, study)
    mapping = match_columns(list(map(str, expr.columns)), meta, study, pilot_dir)
    gsm_cols = {gsm: col for col, gsm in mapping.items()}
    if not set(reviewed.gsm).issubset(gsm_cols):
        raise ValueError('Cannot align all reviewed expression samples to GEO GSM identifiers')
    expr = expr.loc[:, [gsm_cols[g] for g in reviewed.gsm]].copy()
    expr.columns = list(reviewed.gsm)
    scores, coverage, norms = score_all_samples(expr, reviewed, read_gmt(gmt))
    return scores, coverage, norms, {"raw_sha256": raw_hash, "lock": locked,
                                      "expression_path": str(filename)}, expr


def make_plot(df: pd.DataFrame, ref, program: str, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for donor, group in df.groupby('donor_id'):
        label = 'control' if group.group.iloc[0] == 'control' else 'endometriosis'
        ax.plot(group.days, group.score, marker='o', linewidth=1.1, alpha=0.55,
                color=('#247f74' if label == 'control' else '#bc784f'),
                label=label if label not in ax.get_legend_handles_labels()[1] else None)
    dense = np.linspace(ref.days.min(), ref.days.max(), 150)
    ax.plot(dense, ref.interpolator()(dense), color='black', linewidth=2, label='control mean / fitted reference')
    ax.set(xlabel='Days after EPC stimulation', ylabel='Program score (z against controls)',
           title=f'{program}: donor-level temporal profiles')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run_locked_fit(results_dir: Path, gmt: Path, review_path: Path, lock_path: Path,
                   out: Path, offline: bool = True, external_ref: bool = False) -> pd.DataFrame:
    if external_ref:
        raise NotImplementedError('External reference use must be developed as protocol-aware validation, not pooled fitting')
    scores, coverage, norms, manifest, expression = expression_and_samples(results_dir, gmt, review_path, lock_path, offline)
    reviewed = validated_review(review_path)
    programs = read_gmt(gmt)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / 'tables' / 'gene_coverage.csv', index=False)
    norms.to_csv(out / 'tables' / 'control_fitted_gene_normalization.csv', index=False)
    scores.to_csv(out / 'tables' / 'donor_program_scores.csv', index=False)
    rows, diag, recommended, sensitivity, profiles = [], [], [], [], []
    n_control_donors = scores.loc[scores.group == 'control', 'donor_id'].nunique()
    can_loo_control_ref = n_control_donors >= 4
    for program, p in scores.groupby('program'):
        controls = p.query('group == "control"').copy()
        try:
            ref = make_reference(controls)
        except ValueError as err:
            diag.append({'program': program, 'status': f'FAIL: {err}'})
            continue
        make_plot(p, ref, program, out / 'figures' / f'{program}_trajectories.png')
        for derivative in ('pchip', 'spline'):
            diagnostic = design_identifiability(ref, ref.days, derivative=derivative)
            diag.append({'program': program, 'derivative': derivative, 'status': 'COMPUTED_LOCAL_ONLY', **diagnostic})
        # Pre-exposure sampling-grid suggestions rely entirely on control reference.
        grid = sorted(set([*np.arange(0, float(ref.days[-1]) + 0.001, 0.5), float(ref.days[-1])]))
        opt = suggest_sampling_times(ref, grid, n_samples=5, top_k=3)
        for rank, rec in opt.iterrows():
            recommended.append({'program': program, 'rank': rank+1, **rec.to_dict()})
        for donor_id, donor_df in p.groupby('donor_id'):
            group = donor_df.group.iloc[0]
            if group == 'control' and not can_loo_control_ref:
                rows.append({'program': program, 'group': group, 'donor_id': donor_id,
                             'analysis_status': 'SKIPPED: >=4 control donors required for control-LODO'})
                continue
            if group == 'control':
                # Leave control donor out of gene centering/scaling and reference fitting.
                train_ctrls = list(reviewed.loc[(reviewed.group == 'control') & (reviewed.donor_id != donor_id), 'gsm'])
                fold_scores, _, _ = score_all_samples(expression, reviewed, programs,
                                                      train_control_gsms=train_ctrls)
                fm = fold_scores.loc[(fold_scores.program == program) &
                                     (fold_scores.group == 'control') & (fold_scores.donor_id != donor_id)]
                try:
                    ref_donor = make_reference(fm, min_donors=3)
                except ValueError as e:
                    rows.append({'program': program, 'group': group, 'donor_id': donor_id,
                                 'analysis_status': f'EXCLUDED: {e}'})
                    continue
                fit_df = fold_scores.loc[(fold_scores.program == program) & (fold_scores.donor_id == donor_id)]
            else:
                ref_donor = ref
                fit_df = donor_df
            fit_df = fit_df.sort_values('days')
            try:
                fitted = compare_models(ref_donor, fit_df.days.to_numpy(), fit_df.score.to_numpy())
                rows.append({'program': program, 'group': group, 'donor_id': donor_id,
                             'analysis_status': 'OK', **fitted})
                # Raw SSE profiles for patient donors; these are NOT confidence intervals.
                if group == 'endometriosis':
                    lag_profile = fit_delayed(ref_donor, fit_df.days.to_numpy(), fit_df.score.to_numpy())
                    for lag, rss in zip(lag_profile['profile_delay'], lag_profile['profile_sse']):
                        profiles.append({'program': program, 'donor_id': donor_id,
                                         'lag_days': float(lag), 'sse': float(rss)})
            except ValueError as err:
                rows.append({'program': program, 'group': group, 'donor_id': donor_id,
                             'analysis_status': f'EXCLUDED: {err}'})
        # With only three control donors, a 3-donor reference is impossible after LODO.
        # Keep case fits, but explicitly skip this sensitivity analysis.
        if not can_loo_control_ref:
            continue
        # Reference-sensitivity analysis: refit normalization and reference without each control donor.
        for leftout in sorted(controls.donor_id.unique()):
            train_ctrls = list(reviewed.loc[(reviewed.group == 'control') & (reviewed.donor_id != leftout), 'gsm'])
            fold_scores, _, _ = score_all_samples(expression, reviewed, programs,
                                                  train_control_gsms=train_ctrls)
            ref_lodo = make_reference(fold_scores.loc[(fold_scores.program == program) &
                                                       (fold_scores.group == 'control') &
                                                       (fold_scores.donor_id != leftout)])
            cases = fold_scores.loc[(fold_scores.program == program) & (fold_scores.group == 'endometriosis')]
            for donor, cd in cases.groupby('donor_id'):
                fit = compare_models(ref_lodo, cd.days.to_numpy(), cd.score.to_numpy())
                sensitivity.append({'program': program, 'case_donor': donor,
                                    'excluded_control_donor': leftout,
                                    'fit_delay_days': fit['fit_delay_days'],
                                    'fit_amplitude': fit['fit_both_amp'],
                                    'loto_improvement': fit['loto_delta_rmse']})
    fitted_df = pd.DataFrame(rows)
    fitted_df.to_csv(out / 'tables' / 'donor_model_comparisons.csv', index=False)
    pd.DataFrame(diag).to_csv(out / 'tables' / 'control_local_identifiability.csv', index=False)
    pd.DataFrame(recommended).to_csv(out / 'tables' / 'CONTROL_ONLY_exploratory_sampling_designs.csv', index=False)
    pd.DataFrame(sensitivity, columns=['program', 'case_donor', 'excluded_control_donor',
          'fit_delay_days', 'fit_amplitude', 'loto_improvement']).to_csv(
          out / 'tables' / 'case_sensitivity_to_control_reference.csv', index=False)
    pd.DataFrame(profiles).to_csv(out / 'tables' / 'case_lag_RSS_profiles_NOT_CI.csv', index=False)
    if not fitted_df.empty and 'analysis_status' in fitted_df:
        ok = fitted_df.loc[fitted_df.analysis_status == 'OK']
        if not ok.empty:
            summary = ok.groupby(['program','group']).agg(n_donors=('donor_id','nunique'),
                median_delay_days=('fit_delay_days','median'),
                median_amplitude=('fit_both_amp','median'),
                median_loto_improvement=('loto_delta_rmse','median'),
                fraction_donors_loto_improves=('loto_delta_rmse',lambda s: (s > 0).mean())).reset_index()
            summary.to_csv(out / 'tables' / 'descriptive_group_summary_NO_PVALUES.csv', index=False)
    manifest['analysis_output'] = str(out.resolve())
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'INTERPRETATION.txt').write_text(
        'STAGE-1 EXPLORATORY OBSERVATIONAL ANALYSIS - NOT CLINICAL OR CAUSAL.\n'
        'Lag = onset delay constrained to 0..2 days, not true menstrual phase.\n'
        'All normalization and program selection anchored on controls only.\n'
        'LOTO = held-out timepoint within donor, NOT independent held-out patient.\n'
        'Group medians are descriptive, not donor-level significance tests.\n'
        + ('Control LODO sensitivity SKIPPED: <4 control donors.\n' if not can_loo_control_ref else '')
        + 'Local design criterion is not global identifiability; do not claim mechanism without robust validation.\n'
        'Lag RSS profiles are not confidence intervals.\n'
        'A high unexplained residual is not proof of cellular rewiring.\n'
        'GSE246591/GSE127918 are NOT merged: protocols and measurement modalities differ.\n', encoding='utf-8')
    return fitted_df
