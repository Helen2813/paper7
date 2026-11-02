#!/usr/bin/env python3
"""Paper 7 / OnsetAudit v0.7: frozen cross-program challenge and gene attribution.

OFFLINE | NO FLAGS | MAIN ONLY | NO GENE/THRESHOLD/MODEL RETUNING.

Reconstructs the original v0.4 grouped-holdout programs using verified
GSE194200 inputs. Scores both active treatments against BOTH existing drug
programs (including 24 previously unscored cross-program challenges) and
accounts for each gene's algebraic contribution to the existing gate margin.

Both active drugs represent true perturbations. An off-drug score is NOT a
false-positive negative control. The gate uses a different selected 200-gene
feature space for each program, so scores are not calibrated treatment odds.

This is post-hoc descriptive analysis of a previously inspected benchmark,
not new validation, onset estimation, or an improved classifier.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import onsetaudit_v06_gate_baseline_audit as prior

ROOT = Path(__file__).resolve().parent
REPORT_BASE = ROOT / 'onsetaudit/_generated/reports'
DRUGS = prior.DRUGS
GROUPS = prior.GROUPS
TIMES = prior.TIMES
FPS = prior.FPS
AUDIT_VERSION = 'v0.7.0'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def verify_v06(log: prior.Log) -> Path:
    candidates = sorted(REPORT_BASE.glob('v06_gate_baseline_*/audit_summary.json'),
                        key=lambda p: p.parent.name, reverse=True)
    require(bool(candidates), 'Missing v0.6 verified report; run v0.6 first')
    path = candidates[0]
    prior.verify_manifest(path.parent)
    report = json.loads(path.read_text(encoding='utf-8'))
    require(report.get('audit_version') == 'v0.6.0' and
            report.get('frozen_correct_and_accepted') == 22 and
            report.get('frozen_vehicle_rejected') == 10 and
            report.get('frozen_false_positive_distinct') == 2 and
            report.get('oracle_known_time_false_positive_distinct') == 2 and
            report.get('frozen_selected_gene_hashes_reproduced') is True and
            report.get('all_48_original_margins_reproduced') is True,
            'v0.6 status or frozen metrics differ; refusing to continue')
    require(report.get('metadata_sha256') == prior.EXPECTED_METADATA_SHA and
            report.get('counts_sha256') == prior.EXPECTED_COUNTS_SHA,
            'v0.6 source hashes differ')
    original_code = ROOT / 'onsetaudit_v06_gate_baseline_audit.py'
    report_code = path.parent / original_code.name
    require(original_code.is_file() and report_code.is_file() and
            prior.digest(original_code) == prior.digest(report_code),
            'v0.6 source code differs from verified report copy')
    log.write('[V06 VERIFIED]', path, '| SHA256 manifest: OK')
    return path


def reconstruct_program(ds, lookup, frozen_folds, held: str, drug: str):
    from core import TOP_GENES, VAR_FLOOR, RIDGE
    X, counts, meta = ds.matrix, ds.counts, ds.meta
    require((TOP_GENES, VAR_FLOOR, RIDGE) == (200, 0.25, 0.25),
            'Frozen model constants changed')
    require(X is not None and counts is not None and meta is not None,
            'Raw counts + normalized matrix required')
    training = [g for g in GROUPS if g != held]
    units = [(g, 'r1') for g in training]
    for group, replicate in units:
        for t in TIMES:
            for treatment in ('DMSO', drug):
                require((group, replicate, treatment, t) in lookup,
                        'Missing training replicate/drug/time cell')
    require(len(units) == 2, 'Expected exactly two original training groups')
    contrasts = np.stack([
        np.column_stack([
            X[:, lookup[(group, rep, drug, t)]] -
            X[:, lookup[(group, rep, 'DMSO', t)]] for t in TIMES
        ]) for group, rep in units
    ], axis=1)
    response = contrasts.mean(axis=1)
    between = np.var(contrasts, axis=1, ddof=1).mean(axis=1)
    temporal_signal = np.var(response, axis=1)
    rank = temporal_signal / (between + VAR_FLOOR)
    eligible = np.isfinite(rank) & (temporal_signal > 1e-12)
    train_samples = [lookup[(g, r, d, t)] for g, r in units
                     for d in ('DMSO', drug) for t in TIMES]
    eligible &= ((counts[:, train_samples] >= 10).sum(axis=1) >= 3)
    eligible_ids = np.flatnonzero(eligible)
    require(len(eligible_ids) >= TOP_GENES, f'Not enough selectable genes: {held}/{drug}')
    genes = eligible_ids[np.argsort(-rank[eligible_ids], kind='stable')[:TOP_GENES]]
    gene_hash = hashlib.sha256(np.asarray(genes, dtype=np.int64).tobytes()).hexdigest()
    require(gene_hash == frozen_folds[(held, drug)]['selection_score_sha256'],
            f'Frozen gene selection SHA differs: {held}/{drug}')
    scales = np.sqrt(between[genes] + VAR_FLOOR)
    effects = response[genes, :] / scales[:, None]
    time_vehicle = np.column_stack([
        np.mean([X[genes, lookup[(g, r, 'DMSO', t)]] for g, r in units], axis=0)
        for t in TIMES
    ])
    pooled_vehicle = time_vehicle.mean(axis=1)
    return dict(genes=genes, scales=scales, effects=effects,
                by_time_vehicle=time_vehicle, pooled_vehicle=pooled_vehicle,
                lam=RIDGE*TOP_GENES, gene_sha256=gene_hash)


def score_observation(obs: np.ndarray, program: dict):
    """Exact original gate and duration rules plus additive gene accounting."""
    scales = program['scales']
    effects = program['effects']
    lam = program['lam']
    null_diff = (obs - program['pooled_vehicle']) / scales
    null_loss = float(np.mean(null_diff * null_diff))
    gate_losses = np.asarray([
        prior.ridge_loss(null_diff, effects[:, j], lam)
        for j in range(len(TIMES))
    ])
    best_gate_i = int(np.argmin(gate_losses))
    margin = float(null_loss - gate_losses[best_gate_i])
    time_losses = np.asarray([
        prior.ridge_loss((obs - program['by_time_vehicle'][:, j]) / scales,
                         effects[:, j], lam) for j in range(len(TIMES))
    ])
    predicted = float(TIMES[int(np.argmin(time_losses))])
    effect = effects[:, best_gate_i]
    amplitude = max(0., float(null_diff @ effect + lam) / float(effect @ effect + lam))
    residual = null_diff - amplitude*effect
    per_gene = (null_diff**2 - residual**2)/len(null_diff)
    ridge_penalty = -lam*(amplitude-1.)**2 / len(null_diff)
    reconstructed = float(per_gene.sum() + ridge_penalty)
    require(math.isclose(reconstructed, margin, rel_tol=1e-10, abs_tol=1e-10),
            'Additive gate margin identity failed')
    positive = per_gene[per_gene > 0]
    negative = per_gene[per_gene < 0]
    positive_sum = float(positive.sum())
    positive_sorted = np.sort(positive)[::-1]
    top_10_share = float(positive_sorted[:10].sum() / positive_sum) if positive_sum > 0 else None
    top_20_share = float(positive_sorted[:20].sum() / positive_sum) if positive_sum > 0 else None
    effective_positive_genes = (float(positive_sum**2 / np.sum(positive**2))
                                if len(positive) else None)
    return {
        'gate_margin': margin,
        'gate_positive': bool(margin > 0.),
        'predicted_duration_h': predicted,
        'best_gate_template_h': float(TIMES[best_gate_i]),
        'null_loss': null_loss,
        'active_best_loss': float(gate_losses[best_gate_i]),
        'amplitude_best_template': float(amplitude),
        'gene_positive_sum': positive_sum,
        'gene_negative_sum': float(negative.sum()),
        'ridge_penalty_component': float(ridge_penalty),
        'n_positive_gene_contributors': int(len(positive)),
        'n_negative_gene_contributors': int(len(negative)),
        'top_10_positive_share': top_10_share,
        'top_20_positive_share': top_20_share,
        'effective_number_positive_genes': effective_positive_genes,
    }, per_gene


def main_analysis(ds, model, log: prior.Log):
    meta, X = ds.meta, ds.matrix
    require(meta is not None and X is not None and ds.counts is not None,
            'Dataset is incomplete')
    require(len(meta) == 36 and X.shape == (24421, 36),
            'Unexpected verified GSE194200 dimensions')
    require(set(meta['group']) == set(GROUPS) and
            set(meta['drug']) == {'DMSO', *DRUGS} and
            set(meta['time']) == set(TIMES) and
            set(meta['replicate']) == {'r1'}, 'Unexpected frozen study design')
    lookup = {}
    for idx, row in meta.iterrows():
        k = (row['group'], row['replicate'], row['drug'], float(row['time']))
        require(k not in lookup, 'Duplicate design unit')
        lookup[k] = idx
    require(len(lookup) == 36, 'Unexpected number of design cells')
    frozen_folds = {(x['holdout_group'], x['drug']): x for x in model['folds']}
    frozen_active = {x['sample']: x for x in model['predictions']}
    frozen_controls = {(x['sample'], x['drug']): x
                       for x in model['vehicle_drug_predictions']}
    require(len(frozen_folds) == 6 and len(frozen_active) == 24 and
            len(frozen_controls) == 24, 'Frozen model inventory differs')
    scores_rows, top_fp_genes = [], []
    for held in GROUPS:
        for program_drug in DRUGS:
            program = reconstruct_program(ds, lookup, frozen_folds, held, program_drug)
            for actual_drug in ('DMSO', *DRUGS):
                for t in TIMES:
                    obs_idx = lookup[(held, 'r1', actual_drug, t)]
                    sample = str(meta.iloc[obs_idx]['sample'])
                    score, attribution = score_observation(X[program['genes'], obs_idx], program)
                    if actual_drug == program_drug:
                        frozen = frozen_active[sample]
                        require(math.isclose(float(frozen['gate_margin']), score['gate_margin'],
                                             rel_tol=1e-10, abs_tol=2e-9) and
                                float(frozen['predicted_hours']) == score['predicted_duration_h'] and
                                bool(frozen['accepted_active']) == score['gate_positive'],
                                f'Original active score/time not reproduced: {sample}/{program_drug}')
                    if actual_drug == 'DMSO':
                        frozen = frozen_controls[(sample, program_drug)]
                        require(math.isclose(float(frozen['gate_margin']), score['gate_margin'],
                                             rel_tol=1e-10, abs_tol=2e-9) and
                                bool(frozen['accepted_as_active']) == score['gate_positive'],
                                f'Original DMSO score not reproduced: {sample}/{program_drug}')
                    row = {
                        'sample': sample,
                        'heldout_group': held,
                        'true_treatment': actual_drug,
                        'scored_against_program': program_drug,
                        'true_time_h': float(t),
                        'challenge_type': ('vehicle' if actual_drug == 'DMSO' else
                                           'own_program' if actual_drug == program_drug else
                                           'cross_active_not_negative_control'),
                        **score,
                        'predicted_duration_correct': bool(score['predicted_duration_h'] == float(t)),
                        'selected_genes_sha256': program['gene_sha256'],
                    }
                    scores_rows.append(row)
                    if actual_drug == 'DMSO' and score['gate_positive']:
                        indices = np.argsort(-attribution, kind='stable')
                        n_written = 0
                        for index in indices:
                            if attribution[index] <= 0 or n_written >= 25:
                                break
                            n_written += 1
                            top_fp_genes.append({
                                'sample': sample, 'heldout_group': held, 'time_h': float(t),
                                'scored_against_program': program_drug,
                                'contribution_rank': n_written,
                                'gene_id': str(ds.gene_ids[int(program['genes'][index])]),
                                'positive_margin_component': float(attribution[index]),
                                'positive_fraction_of_all_positive':
                                    float(attribution[index] / score['gene_positive_sum']),
                            })
    df = pd.DataFrame(scores_rows)
    fp_genes = pd.DataFrame(top_fp_genes)
    require(len(df) == 72 and df['sample'].nunique() == 36, 'Expected 72 sample × program scores')
    require((df.challenge_type == 'vehicle').sum() == 24 and
            (df.challenge_type == 'own_program').sum() == 24 and
            (df.challenge_type == 'cross_active_not_negative_control').sum() == 24,
            'Unexpected challenge category counts')
    require(int((df.loc[df.challenge_type == 'vehicle', 'gate_positive']).sum()) == 4,
            'Frozen 4 drug-program vehicle false calls differ')
    fp_set = set(df.loc[(df.challenge_type == 'vehicle') & df.gate_positive, 'sample'])
    require(fp_set == FPS, f'Frozen false-positive IDs changed: {fp_set}')
    own = df[df.challenge_type == 'own_program'].copy()
    require(int((own.gate_positive & own.predicted_duration_correct).sum()) == 22 and
            int(own.gate_positive.sum()) == 24, 'Frozen 22/24 active performance differs')
    require(set(fp_genes['sample']) == FPS and 0 < len(fp_genes) <= 100 and
            fp_genes.groupby(['sample', 'scored_against_program']).ngroups == 4,
            'Expected positive gene components for all four frozen FP program calls')
    # Compare original active sample against a *different active program*.
    cross = []
    for sample, grp in df.loc[df.true_treatment != 'DMSO'].groupby('sample', sort=True):
        own_row = grp.loc[grp.challenge_type == 'own_program']
        other_row = grp.loc[grp.challenge_type == 'cross_active_not_negative_control']
        require(len(own_row) == len(other_row) == 1,
                f'Paired cross-program score missing for {sample}')
        own_r, alt_r = own_row.iloc[0], other_row.iloc[0]
        cross.append({
            'sample': sample,
            'heldout_group': own_r.heldout_group,
            'true_treatment': own_r.true_treatment,
            'true_time_h': float(own_r.true_time_h),
            'own_program': own_r.scored_against_program,
            'own_gate_margin': float(own_r.gate_margin),
            'own_gate_accepted': bool(own_r.gate_positive),
            'own_predicted_duration_h': float(own_r.predicted_duration_h),
            'own_duration_correct': bool(own_r.predicted_duration_correct),
            'other_active_program': alt_r.scored_against_program,
            'other_program_margin': float(alt_r.gate_margin),
            'other_program_accepted': bool(alt_r.gate_positive),
            'other_program_predicted_duration_h': float(alt_r.predicted_duration_h),
            'other_program_duration_correct': bool(alt_r.predicted_duration_correct),
            'both_programs_accepted': bool(own_r.gate_positive and alt_r.gate_positive),
            'IMPORTANT': 'OTHER_ACTIVE_TREATMENT_IS_NOT_A_NEGATIVE_CONTROL',
        })
    cross_df = pd.DataFrame(cross)
    require(len(cross_df) == 24 and set(cross_df['sample']) == set(frozen_active),
            'Cross-program challenge is incomplete')
    control_df = df[df.challenge_type == 'vehicle'].copy()
    cross_accept_n = int(cross_df.other_program_accepted.sum())
    cross_time_n = int(cross_df.other_program_duration_correct.sum())
    cross_joint_n = int((cross_df.other_program_accepted &
                         cross_df.other_program_duration_correct).sum())
    log.write('[REPLAY] 6 frozen feature hashes; original 24 active and 24 DMSO margins: EXACT')
    log.write('[FROZEN] correct+accepted 22/24; DMSO rejected 10/12; no retuning')
    log.write(f'[CROSS PROGRAM] {cross_accept_n}/24 other-active responses gate accepted '
              f'(NOT false positives; not independent negative controls)')
    log.write(f'[CROSS DURATION] {cross_time_n}/24 other-program duration labels correct; '
              f'{cross_joint_n}/24 both accepted and correct (exploratory)')
    for drug in DRUGS:
        sub = cross_df.loc[cross_df.true_treatment == drug]
        log.write(f'[CROSS] {drug} -> {sub.other_active_program.iloc[0]}: '
                  f'{int(sub.other_program_accepted.sum())}/{len(sub)} accepted; '
                  f'{int(sub.other_program_duration_correct.sum())}/{len(sub)} time correct')
    for sample in sorted(FPS):
        for drug in DRUGS:
            r = control_df.loc[(control_df['sample'] == sample) &
                               (control_df['scored_against_program'] == drug)].iloc[0]
            log.write(f'[FP ATTR] {sample} / {drug}: margin={r.gate_margin:+.4f}, '
                      f'positive genes={r.n_positive_gene_contributors}/200, '
                      f'top10 positive share={r.top_10_positive_share:.3f}, '
                      f'effective positive genes={r.effective_number_positive_genes:.1f}')
    return df, cross_df, control_df, fp_genes


def safe_table(df: pd.DataFrame) -> str:
    temp = df.copy()
    for c in temp.columns:
        if pd.api.types.is_bool_dtype(temp[c]):
            temp[c] = temp[c].map({True: 'Yes', False: 'No'})
        elif pd.api.types.is_float_dtype(temp[c]):
            temp[c] = temp[c].map(lambda x: f'{x:,.4f}' if pd.notna(x) else '—')
    return temp.to_html(index=False, classes='data', border=0, escape=True)


def dashboard(cross: pd.DataFrame, controls: pd.DataFrame, genes: pd.DataFrame,
              report06: Path, meta_hash: str, counts_hash: str) -> str:
    n_acc = int(cross.other_program_accepted.sum())
    n_time = int(cross.other_program_duration_correct.sum())
    n_joint = int((cross.other_program_accepted & cross.other_program_duration_correct).sum())
    false = controls.loc[controls.gate_positive]
    overview = cross.groupby(['heldout_group', 'true_treatment'], sort=True).agg(
        samples=('sample','size'),
        cross_accepted=('other_program_accepted','sum'),
        cross_time_correct=('other_program_duration_correct','sum'),
    ).reset_index()
    intro = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>OnsetAudit v0.7 — cross-program challenge</title>
<style>
body{{margin:0;background:#f4f6fa;color:#13253b;font-family:system-ui,-apple-system,Arial,sans-serif}}
main{{max-width:1180px;margin:36px auto;padding:0 20px}} h1{{font-size:27px;letter-spacing:-.6px}}
p{{line-height:1.55;color:#53647a}} h2{{font-size:20px;margin-top:30px}}
.meta{{color:#6b7e96;font-size:14px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.card{{background:white;border:1px solid #dae3ef;border-radius:13px;padding:17px;margin:12px 0;overflow:auto}}
.cards .card{{margin:0}}.card b{{display:block;font-size:26px}}small{{color:#738199}}
.warning{{border-left:4px solid #c48d28;background:#fff8e9;border-radius:8px;padding:15px 18px;margin-top:18px;line-height:1.55}}
.data{{width:100%;border-collapse:collapse;font-size:12px;min-width:700px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e3e8f2;white-space:nowrap}}
th{{background:#f0f4fa}} code{{overflow-wrap:anywhere}}
@media(max-width:780px){{.cards{{grid-template-columns:repeat(2,1fr)}} main{{padding:0 10px}}}}
</style></head><body><main>
<h1>OnsetAudit v0.7 · Cross-program challenge &amp; gene attribution</h1>
<p class="meta">GSE194200 · frozen v0.4 grouped holdout · retrospective descriptive</p>
<div class="cards"><div class="card"><b>22/24</b><small>Frozen active duration + gate correct</small></div>
<div class="card"><b>10/12</b><small>Frozen DMSO rejected</small></div>
<div class="card"><b>{n_acc}/24</b><small>Cross-program active calls accepted</small></div>
<div class="card"><b>{n_joint}/24</b><small>Cross-program accepted + correct duration</small></div></div>
<div class="warning"><b>What this does NOT show.</b> SW016789 and Thapsigargin are both true active treatments. A cross-program accepted sample is <b>not</b> a false-positive control. This is not a validated drug-identity classifier; the two 200-gene spaces differ and gate margins are not drug posterior probabilities. No threshold, genes, or model settings were changed. All cross-program results are post-hoc. There is no direct measurement of molecular onset.</div>
<p>The cross-program time-only count is <b>{n_time}/24</b>. A correct duration under a different drug model suggests transferable temporal information within this dataset, not drug specificity or generalization to endometrium.</p>
<h2>Cross-program challenge by heldout experiment</h2><div class="card">'''
    intro += safe_table(overview)
    intro += '</div><h2>All 24 active cross-program challenges</h2><div class="card">'
    intro += safe_table(cross[['sample','heldout_group','true_treatment','true_time_h',
                               'own_gate_margin','own_predicted_duration_h',
                               'other_active_program','other_program_margin',
                               'other_program_accepted','other_program_predicted_duration_h',
                               'other_program_duration_correct']])
    intro += '</div><h2>Frozen DMSO control gene-contribution audit (24 program calls)</h2><div class="card">'
    intro += safe_table(controls[['sample','heldout_group','true_time_h','scored_against_program',
                                   'gate_margin','gate_positive','best_gate_template_h',
                                   'n_positive_gene_contributors','top_10_positive_share',
                                   'top_20_positive_share','effective_number_positive_genes',
                                   'gene_positive_sum','gene_negative_sum','ridge_penalty_component']])
    intro += '</div><h2>Up to 100 highest positive gene contributions across four dependent false-positive program calls</h2><div class="card">'
    intro += safe_table(genes)
    intro += '''</div><div class="warning"><b>Attribution caution.</b> Per-gene components are an exact algebraic decomposition of the frozen ridge-loss difference, not causal feature importance. The best gate template is chosen from four times, while duration prediction uses time-matched vehicle residuals. Top gene lists are generated after examining errors and must not be used for feature retuning.</div>'''
    intro += f'<p class="meta">Source v0.6 audit: <code>{html.escape(str(report06))}</code><br>Metadata SHA-256: <code>{meta_hash}</code><br>Counts SHA-256: <code>{counts_hash}</code></p>'
    intro += '</main></body></html>'
    return intro


def run(out: Path, log: prior.Log):
    log.write('='*78)
    log.write('PAPER 7 | ONSETAUDIT v0.7 | CROSS-PROGRAM + GENE ATTRIBUTION')
    log.write('OFFLINE | NO FLAGS | MAIN ONLY | FROZEN v0.4 MODEL AND GATE')
    log.write('='*78)
    v06_path = verify_v06(log)
    head, ev, v04_path, v05_path, metadata, counts = prior.inputs_and_reports(log)
    ds, model = prior.dataset_and_replay(ev, metadata, counts, log)
    scores, cross, controls, top_genes = main_analysis(ds, model, log)
    scores.to_csv(out/'all_72_sample_program_scores.csv', index=False)
    cross.to_csv(out/'cross_program_24_active_samples.csv', index=False)
    controls.to_csv(out/'dmso_24_program_gene_attributions.csv', index=False)
    top_genes.to_csv(out/'false_positive_top_gene_components.csv', index=False)
    html_report = out/'onsetaudit_v07_dashboard.html'
    html_report.write_text(dashboard(cross, controls, top_genes, v06_path,
                                     prior.EXPECTED_METADATA_SHA,
                                     prior.EXPECTED_COUNTS_SHA), encoding='utf-8')
    summary = {
        'audit_version': AUDIT_VERSION,
        'status': 'RETROSPECTIVE_CROSS_ACTIVE_CHALLENGE_AND_ALGEBRAIC_GENE_DECOMPOSITION',
        'dataset': 'GSE194200', 'source_git_head_at_run': head,
        'source_v04_report': str(v04_path), 'source_v04_sha256': prior.digest(v04_path),
        'source_v05_report': str(v05_path), 'source_v05_sha256': prior.digest(v05_path),
        'source_v06_report': str(v06_path), 'source_v06_sha256': prior.digest(v06_path),
        'metadata_sha256': prior.EXPECTED_METADATA_SHA,
        'counts_sha256': prior.EXPECTED_COUNTS_SHA,
        'selected_gene_hashes_reproduced': True,
        'frozen_24_active_and_24_control_scores_reproduced': True,
        'frozen_active_correct_and_accepted': 22,
        'frozen_dmso_rejected_distinct': 10,
        'frozen_dmso_false_positive_distinct_samples': sorted(FPS),
        'frozen_dmso_false_drug_program_calls': 4,
        'active_cross_program_challenge_n': 24,
        'active_cross_program_accepted_n': int(cross.other_program_accepted.sum()),
        'active_cross_program_duration_correct_n': int(cross.other_program_duration_correct.sum()),
        'active_cross_program_accepted_and_correct_duration_n':
            int((cross.other_program_accepted & cross.other_program_duration_correct).sum()),
        'gene_attribution_unit': 'Algebraic per-gene component of selected-best-template ridge gate margin',
        'interpretation_limits': [
            'Cross-program treatment samples are positive perturbations, NOT vehicle negative controls',
            'Cross-program gate margins come from different gene spaces and are not drug identity probabilities',
            'Best gate time and reported duration time use different references and can differ',
            'Gene attributions are additive algebraic accounting, not causal effects or pre-specified biomarkers',
            'Post-hoc errors in one cell line/one cohort; do not tune genes, times or thresholds',
            'Each held-out group is a single experiment, not a fully independent new laboratory/cell population',
            'No molecular onset truth or independent endometrial validation is available',
        ],
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'pandas': pd.__version__,
    }
    prior.save_json(out/'audit_summary.json', summary)
    (out/'README.txt').write_text(
        'OnsetAudit v0.7 cross-program active challenge plus algebraic gene attributions.\n'
        'Frozen GSE194200 v0.4 model, original scores and gate threshold 0.0 reproduced.\n'
        'The other active drug is NOT a vehicle negative control; cross-drug acceptance is NOT a false-positive rate.\n'
        'Per-gene contribution identity includes a negative ridge penalty component.\n'
        'No model/gene/threshold changes or prospective validation. This is not biological onset.\n'
        'Source raw counts remain on the local Mac and are SHA-256 identified.\n',
        encoding='utf-8')
    shutil.copyfile(Path(__file__), out/Path(__file__).name)
    log.write('[OUTPUT]', out)
    log.write('[DASHBOARD]', html_report)
    log.write('[POLICY] v0.4 core and all prior reports unchanged; no retuning')


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    REPORT_BASE.mkdir(parents=True, exist_ok=True)
    out = REPORT_BASE / f'v07_cross_program_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    log = prior.Log(out/'console.log')
    try:
        run(out, log)
        log.write('[READY] v0.7 cross-program diagnostic completed')
    except BaseException:
        log.write('[STOP] v0.7 failed; previous app/data/reports not modified')
        log.write(traceback.format_exc())
        raise
    finally:
        log.close()
    paths = sorted(x for x in out.iterdir()
                   if x.is_file() and x.name not in
                   ('sha256_manifest.json', 'onsetaudit_v07_audit.zip'))
    prior.save_json(out/'sha256_manifest.json', {x.name: prior.digest(x) for x in paths})
    zpath = out/'onsetaudit_v07_audit.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for x in sorted(out.iterdir()):
            if x.is_file() and x != zpath:
                z.write(x, arcname=x.name)
    print('[ZIP]', zpath, flush=True)
    if sys.platform == 'darwin':
        subprocess.run(['open', str(out/'onsetaudit_v07_dashboard.html')], check=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('[STOP]', type(exc).__name__ + ':', exc, file=sys.stderr, flush=True)
        sys.exit(1)
