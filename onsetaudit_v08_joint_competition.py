#!/usr/bin/env python3
"""Paper 7 / OnsetAudit v0.8: shared-feature joint DMSO/drug/time challenge.

OFFLINE | NO FLAGS | MAIN ONLY | UNCHANGED v0.4 CORE | POST-HOC PROTOTYPE.

Why: the original SW016789 and Thapsigargin gate margins are computed in
DIFFERENT 200-gene spaces and cannot be compared as drug probabilities. This
exploratory, separately labelled competitor instead constructs one shared
training-only union of frozen 200-gene feature sets per held-out experiment,
with one common normalization and one common loss for 12 hypotheses:
DMSO at 1/2/6/24h, SW016789 at 1/2/6/24h, and Thapsigargin at 1/2/6/24h.
All hypothesis times are searched; no recorded true time is used for inference.

This is a NEW, RETROSPECTIVE, EXPLORATORY JOINT OBJECTIVE, *NOT* the frozen
v0.4 classifier, a calibrated identity model, or independent validation.
The frozen model and all preceding outputs must reproduce exactly and are not
modified. Model performance on this inspected dataset is for diagnosis only.
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

import onsetaudit_v07_cross_program_attribution as v07

ROOT = Path(__file__).resolve().parent
REPORT_BASE = ROOT / 'onsetaudit/_generated/reports'
AUDIT_VERSION = 'v0.8.0'
DRUGS = tuple(v07.DRUGS)
GROUPS = tuple(v07.GROUPS)
TIMES = tuple(v07.TIMES)
LABELS = ('DMSO', *DRUGS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_v07(log: v07.prior.Log) -> Path:
    candidates = sorted(
        REPORT_BASE.glob('v07_cross_program_*/audit_summary.json'),
        key=lambda p: p.parent.name,
        reverse=True
    )

    require(
        bool(candidates),
        'Missing v0.7 cross-program report: run v0.7 first'
    )

    path = candidates[0]

    v07.prior.verify_manifest(path.parent)

    report = json.loads(path.read_text(encoding='utf-8'))

    require(
        report.get('audit_version') == 'v0.7.0'
        and report.get('frozen_active_correct_and_accepted') == 22
        and report.get('frozen_dmso_rejected_distinct') == 10
        and report.get('active_cross_program_accepted_n') == 24
        and report.get('active_cross_program_duration_correct_n') == 17
        and report.get('selected_gene_hashes_reproduced') is True
        and report.get(
            'frozen_24_active_and_24_control_scores_reproduced'
        ) is True,
        'v0.7 frozen/cross-program results do not match expected report'
    )

    require(
        report.get('metadata_sha256')
        == v07.prior.EXPECTED_METADATA_SHA
        and report.get('counts_sha256')
        == v07.prior.EXPECTED_COUNTS_SHA,
        'v0.7 source data hashes differ'
    )

    local_code = ROOT / 'onsetaudit_v07_cross_program_attribution.py'
    snap_code = path.parent / local_code.name

    require(
        local_code.is_file()
        and snap_code.is_file()
        and v07.prior.digest(local_code)
        == v07.prior.digest(snap_code),
        'v0.7 script does not match its verified report snapshot'
    )

    cross_csv = path.parent / 'cross_program_24_active_samples.csv'
    cross = pd.read_csv(cross_csv)

    require(
        len(cross) == 24
        and int(cross['other_program_accepted'].sum()) == 24
        and int(cross['other_program_duration_correct'].sum()) == 17,
        'v0.7 exported CSV differs from verified totals'
    )

    log.write(
        '[V07 VERIFIED]',
        path,
        '| SHA-256: OK; 24/24 accepted; 17/24 time correct'
    )

    return path


def study_lookup(ds):
    meta = ds.meta

    require(
        meta is not None
        and ds.matrix is not None
        and ds.counts is not None,
        'Verified count matrix and metadata required'
    )

    require(
        len(meta) == 36
        and ds.matrix.shape == (24421, 36),
        'Unexpected verified GSE194200 dimensions'
    )

    require(
        set(meta['group']) == set(GROUPS)
        and set(meta['drug']) == set(LABELS)
        and set(meta['time']) == set(TIMES)
        and set(meta['replicate']) == {'r1'},
        'Unexpected GSE194200 study design'
    )

    lookup = {}

    for idx, row in meta.iterrows():
        k = (
            str(row['group']),
            str(row['replicate']),
            str(row['drug']),
            float(row['time'])
        )

        require(k not in lookup, 'Duplicate sample design cell')
        lookup[k] = int(idx)

    require(
        len(lookup) == 36,
        'Expected 36 unique experiment/treatment/time combinations'
    )

    return lookup


def common_fold(ds, model, lookup: dict, held: str):
    """Training-only common feature union, scale and 12 model hypotheses."""

    from core import TOP_GENES, VAR_FLOOR, RIDGE

    require(
        (TOP_GENES, VAR_FLOOR, RIDGE) == (200, 0.25, 0.25),
        'Frozen feature/ridge constants changed'
    )

    frozen_folds = {
        (x['holdout_group'], x['drug']): x
        for x in model['folds']
    }

    require(
        len(frozen_folds) == 6,
        'Original 6 training folds missing'
    )

    programs = {
        drug: v07.reconstruct_program(
            ds, lookup, frozen_folds, held, drug
        )
        for drug in DRUGS
    }

    a, b = (programs[d]['genes'] for d in DRUGS)

    common = np.union1d(a, b).astype(np.int64)
    overlap = int(np.intersect1d(a, b).size)

    require(
        common.size == 400 - overlap
        and 200 <= common.size <= 400,
        'Invalid selected training-only feature union'
    )

    X = ds.matrix

    train_groups = tuple(g for g in GROUPS if g != held)

    require(
        len(train_groups) == 2,
        'Original 2-group training design changed'
    )

    for group in train_groups:
        for t in TIMES:
            for drug in LABELS:
                require(
                    (group, 'r1', drug, t) in lookup,
                    f'Missing training design cell {group}/{drug}/{t}'
                )

    vehicle = np.column_stack([
        np.mean(
            [
                X[common, lookup[(g, 'r1', 'DMSO', t)]]
                for g in train_groups
            ],
            axis=0
        )
        for t in TIMES
    ])

    effect_unscaled = {}
    between_by_drug = {}

    for drug in DRUGS:

        contrasts = np.stack(
            [
                np.column_stack(
                    [
                        X[common, lookup[(g, 'r1', drug, t)]]
                        - X[common, lookup[(g, 'r1', 'DMSO', t)]]
                        for t in TIMES
                    ]
                )
                for g in train_groups
            ],
            axis=1
        )

        effect_unscaled[drug] = np.mean(
            contrasts, axis=1
        )

        between_by_drug[drug] = np.var(
            contrasts, axis=1, ddof=1
        ).mean(axis=1)

    shared_between = np.mean(
        np.stack(
            [between_by_drug[d] for d in DRUGS],
            axis=1
        ),
        axis=1
    )

    scales = np.sqrt(shared_between + VAR_FLOOR)

    require(
        np.all(np.isfinite(scales))
        and np.all(scales > 0),
        'Invalid training-only shared scales'
    )

    effects = {
        drug: effect_unscaled[drug] / scales[:, None]
        for drug in DRUGS
    }

    require(
        all(np.isfinite(x).all() for x in effects.values()),
        'Invalid training-only response templates'
    )

    feature_hash = hashlib.sha256(
        common.tobytes()
    ).hexdigest()

    return {
        'genes': common,
        'vehicle': vehicle,
        'effects': effects,
        'scales': scales,
        'ridge_lambda': RIDGE * len(common),
        'n_features': int(len(common)),
        'overlap': overlap,
        'feature_union_sha256': feature_hash,
        'sw_selected_gene_sha256':
            programs[DRUGS[0]]['gene_sha256'],
        'th_selected_gene_sha256':
            programs[DRUGS[1]]['gene_sha256'],
    }


def joint_scores(obs: np.ndarray, fold: dict):
    """Compare all 12 drug/time hypotheses with identical features and scale.

    Null: ||(x - training_DMSO_at_candidate_t) / common_scale||^2 / p

    Active: ridge_loss(
        (x - training_DMSO_at_candidate_t) / scale,
        training_drug_minus_DMSO_at_t / scale,
        lambda
    )

    All candidate times, INCLUDING DMSO times, are searched.
    No oracle time.
    """

    from core import RIDGE

    p = int(fold['n_features'])

    require(
        obs.shape == (p,),
        'Observed expression vector dimensions differ'
    )

    require(
        math.isclose(
            fold['ridge_lambda'],
            RIDGE * p,
            abs_tol=1e-12
        ),
        'Unexpected joint ridge penalty'
    )

    scores = {label: [] for label in LABELS}

    for ti, t in enumerate(TIMES):

        residual = (
            obs - fold['vehicle'][:, ti]
        ) / fold['scales']

        null_loss = float(
            np.mean(residual * residual)
        )

        scores['DMSO'].append(null_loss)

        for drug in DRUGS:

            active_loss = v07.prior.ridge_loss(
                residual,
                fold['effects'][drug][:, ti],
                fold['ridge_lambda']
            )

            scores[drug].append(
                float(active_loss)
            )

    best = {}

    for label in LABELS:

        arr = np.asarray(
            scores[label], dtype=float
        )

        require(
            arr.shape == (4,)
            and np.all(np.isfinite(arr))
            and np.all(arr >= -1e-10),
            'Invalid objective score'
        )

        ti = int(np.argmin(arr))

        best[label] = {
            'loss': max(0., float(arr[ti])),
            'time_h': float(TIMES[ti])
        }

    # Fixed tie preference:
    # DMSO, SW016789, Thapsigargin.
    predicted = min(
        LABELS,
        key=lambda label: best[label]['loss']
    )

    chosen = best[predicted]

    sorted_losses = sorted([
        (best[label]['loss'], label)
        for label in LABELS
    ])

    if predicted != 'DMSO':

        nearest_competing = min(
            best[label]['loss']
            for label in LABELS
            if label != predicted
        )

        label_gap = float(
            nearest_competing - chosen['loss']
        )

    else:

        label_gap = float(
            min(
                best[label]['loss']
                for label in DRUGS
            ) - chosen['loss']
        )

    return (
        best,
        predicted,
        chosen['time_h'],
        label_gap,
        sorted_losses
    )


def run_joint(ds, model, log: v07.prior.Log):

    lookup = study_lookup(ds)

    rows = []
    folds = []

    for held in GROUPS:

        program = common_fold(
            ds, model, lookup, held
        )

        folds.append({
            'heldout_group': held,
            'n_shared_union_genes':
                program['n_features'],
            'selected_gene_overlap':
                program['overlap'],
            'shared_union_sha256':
                program['feature_union_sha256'],
            'SW016789_200_gene_sha256':
                program['sw_selected_gene_sha256'],
            'Thapsigargin_200_gene_sha256':
                program['th_selected_gene_sha256'],
            'training_groups': ';'.join(
                g for g in GROUPS if g != held
            ),
        })

        log.write(
            '[FOLD]',
            held,
            f'common genes={program["n_features"]}',
            f'overlap={program["overlap"]}',
            'training_only=YES'
        )

        for label in LABELS:

            for t in TIMES:

                idx = lookup[
                    (held, 'r1', label, t)
                ]

                meta = ds.meta.iloc[idx]
                sample = str(meta['sample'])

                obs = ds.matrix[
                    program['genes'], idx
                ]

                (
                    best,
                    predicted,
                    duration,
                    gap,
                    sorted_losses
                ) = joint_scores(obs, program)

                rows.append({
                    'sample': sample,
                    'heldout_group': held,
                    'true_treatment': label,
                    'true_duration_h': float(t),
                    'predicted_treatment': predicted,
                    'predicted_duration_h': duration,
                    'treatment_correct':
                        bool(predicted == label),
                    'time_correct':
                        bool(duration == t),
                    'joint_treatment_time_correct':
                        bool(
                            predicted == label
                            and duration == t
                        ),
                    'active_detected':
                        bool(predicted != 'DMSO'),
                    'best_loss_DMSO':
                        best['DMSO']['loss'],
                    'best_time_DMSO_h':
                        best['DMSO']['time_h'],
                    f'best_loss_{DRUGS[0]}':
                        best[DRUGS[0]]['loss'],
                    f'best_time_{DRUGS[0]}_h':
                        best[DRUGS[0]]['time_h'],
                    f'best_loss_{DRUGS[1]}':
                        best[DRUGS[1]]['loss'],
                    f'best_time_{DRUGS[1]}_h':
                        best[DRUGS[1]]['time_h'],
                    'loss_gap_best_vs_second_class':
                        gap,
                    'n_common_features':
                        program['n_features'],
                    'common_features_sha256':
                        program['feature_union_sha256'],
                })

    df = pd.DataFrame(rows)
    features = pd.DataFrame(folds)

    require(
        len(df) == 36
        and df['sample'].nunique() == 36
        and len(features) == 3,
        'Expected 36 held-out sample predictions in 3 folds'
    )

    require(
        df.groupby(
            ['heldout_group', 'true_treatment']
        ).size().eq(4).all(),
        'Unexpected prediction counts per fold/treatment'
    )

    require(
        set(df['predicted_treatment']).issubset(
            set(LABELS)
        )
        and set(df['predicted_duration_h']).issubset(
            set(TIMES)
        ),
        'Invalid predicted class or time'
    )

    confusion = pd.crosstab(
        df['true_treatment'],
        df['predicted_treatment']
    )

    confusion = confusion.reindex(
        index=list(LABELS),
        columns=list(LABELS),
        fill_value=0
    )

    totals = {
        'n_all': 36,
        'n_active': 24,
        'n_dmso': 12,

        'dmso_correctly_rejected': int(
            (
                (df.true_treatment == 'DMSO')
                & (df.predicted_treatment == 'DMSO')
            ).sum()
        ),

        'active_detected': int(
            (
                (df.true_treatment != 'DMSO')
                & (df.predicted_treatment != 'DMSO')
            ).sum()
        ),

        'active_drug_identity_correct': int(
            (
                (df.true_treatment != 'DMSO')
                & df.treatment_correct
            ).sum()
        ),

        'active_time_correct_regardless_drug': int(
            (
                (df.true_treatment != 'DMSO')
                & df.time_correct
            ).sum()
        ),

        'active_drug_and_time_correct': int(
            (
                (df.true_treatment != 'DMSO')
                & df.joint_treatment_time_correct
            ).sum()
        ),

        'all_3_class_correct': int(
            df.treatment_correct.sum()
        ),

        'all_joint_class_time_correct': int(
            df.joint_treatment_time_correct.sum()
        ),
    }

    log.write(
        '[JOINT OBJECTIVE]',
        'training-only union features, common scales,',
        '12 equally searchable drug/time hypotheses'
    )

    log.write(
        '[JOINT]',
        f'active detected {totals["active_detected"]}/24;',
        f'DMSO rejected {totals["dmso_correctly_rejected"]}/12'
    )

    log.write(
        '[IDENTITY]',
        f'active drug correct '
        f'{totals["active_drug_identity_correct"]}/24',
        '(NOT calibrated probabilities)'
    )

    log.write(
        '[JOINT TIME]',
        f'active drug AND time correct '
        f'{totals["active_drug_and_time_correct"]}/24;',
        f'active time regardless of identity '
        f'{totals["active_time_correct_regardless_drug"]}/24'
    )

    for label in LABELS:

        sub = df.loc[
            df.true_treatment == label
        ]

        log.write(
            '[CLASS]',
            label,
            'n=', len(sub),
            'label_correct=',
            int(sub.treatment_correct.sum()),
            'label_plus_time=',
            int(sub.joint_treatment_time_correct.sum())
        )

    for held in GROUPS:

        sub = df.loc[
            df.heldout_group == held
        ]

        log.write(
            '[GROUP]',
            held,
            f'class_correct='
            f'{int(sub.treatment_correct.sum())}/12;',
            f'joint_correct='
            f'{int(sub.joint_treatment_time_correct.sum())}/12'
        )

    return df, features, confusion, totals


def safe_table(df: pd.DataFrame) -> str:

    temp = df.copy()

    for col in temp.columns:

        if pd.api.types.is_bool_dtype(
            temp[col]
        ):
            temp[col] = temp[col].map({
                True: 'Yes',
                False: 'No'
            })

        elif pd.api.types.is_float_dtype(
            temp[col]
        ):
            temp[col] = temp[col].map(
                lambda v: (
                    f'{v:,.4f}'
                    if pd.notna(v)
                    else '—'
                )
            )

    return temp.to_html(
        index=False,
        classes='data',
        border=0,
        escape=True
    )


def make_dashboard(
    df: pd.DataFrame,
    features: pd.DataFrame,
    confusion: pd.DataFrame,
    totals: dict,
    source_path: Path
) -> str:

    by_group = df.groupby(
        'heldout_group',
        sort=True
    ).agg(
        samples=('sample', 'size'),
        class_correct=('treatment_correct', 'sum'),
        joint_class_time_correct=(
            'joint_treatment_time_correct',
            'sum'
        )
    ).reset_index()

    confusion_display = (
        confusion.copy()
        .reset_index()
        .rename(columns={
            'true_treatment': 'Actual / predicted'
        })
    )

    cols = [
        'sample',
        'heldout_group',
        'true_treatment',
        'true_duration_h',
        'predicted_treatment',
        'predicted_duration_h',
        'treatment_correct',
        'time_correct',
        'joint_treatment_time_correct',
        'best_loss_DMSO',
        'best_loss_SW016789',
        'best_loss_Thapsigargin',
        'loss_gap_best_vs_second_class'
    ]

    html_parts = [f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OnsetAudit v0.8 — shared-feature competition</title>
<style>
body{{
    margin:0;
    background:#f3f5fa;
    color:#14263d;
    font-family:system-ui,-apple-system,Arial,sans-serif
}}
main{{
    max-width:1220px;
    margin:36px auto;
    padding:0 20px
}}
h1{{font-size:28px;letter-spacing:-.5px}}
h2{{font-size:20px;margin-top:30px}}
p{{line-height:1.56;color:#50647b}}
.meta{{font-size:13px;color:#6a7c91}}
.cards{{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px
}}
.card{{
    border:1px solid #dbe3ef;
    background:white;
    border-radius:13px;
    padding:17px;
    margin:12px 0;
    overflow:auto
}}
.cards .card{{margin:0}}
.cards .card b{{font-size:27px;display:block}}
small{{font-size:12px;color:#728397}}
.warning{{
    border-left:4px solid #bd8b31;
    background:#fff8ea;
    border-radius:9px;
    padding:16px 20px;
    line-height:1.55
}}
.data{{
    border-collapse:collapse;
    font-size:12px;
    min-width:680px;
    width:100%
}}
th,td{{
    padding:8px 10px;
    text-align:left;
    white-space:nowrap;
    border-bottom:1px solid #e3e9f2
}}
th{{background:#eef2f9}}
code{{overflow-wrap:anywhere}}
@media(max-width:850px){{
    .cards{{
        grid-template-columns:repeat(2,minmax(0,1fr))
    }}
    main{{padding:0 10px}}
}}
</style>
</head>
<body>
<main>

<h1>OnsetAudit v0.8 · Same-feature joint competition</h1>

<p class="meta">
GSE194200 · frozen v0.4 verified unchanged · retrospective new objective
</p>

<div class="cards">
<div class="card">
<b>{totals['active_detected']}/24</b>
<small>Active treatments detected</small>
</div>
<div class="card">
<b>{totals['dmso_correctly_rejected']}/12</b>
<small>DMSO classified as DMSO</small>
</div>
<div class="card">
<b>{totals['active_drug_identity_correct']}/24</b>
<small>Active drug identities correct</small>
</div>
<div class="card">
<b>{totals['active_drug_and_time_correct']}/24</b>
<small>Active drug AND duration correct</small>
</div>
</div>

<div class="warning">
<b>Study limitations.</b>
This analysis constructs a NEW exploratory 12-way objective
(DMSO, SW016789, Thapsigargin × 1/2/6/24 hours) from the same
previously inspected experiment.

This is NOT the frozen OnsetAudit v0.4 model, not a calibrated
drug-identity posterior, not an independent validation, and not
direct molecular-onset measurement.

All features and scales are computed using the two training
experiments for each held-out experiment.

Time is unknown to inference and searched for controls as well
as both drugs; no test labels are used for training.

No model selection, threshold tuning or gene removal was
performed on the held-out results.
</div>

<p>
Compared with v0.7, which evaluates two different 200-gene
spaces separately, this exploratory objective scores all
12 hypotheses in the <b>same gene union, shared variance
scale and loss function</b>.

It tests whether discriminating drug identity and elapsed
time is feasible within this dataset, not whether v0.4
is a validated classifier.
</p>

<h2>3-class confusion matrix</h2>
<div class="card">''',

    safe_table(confusion_display),

    '</div><h2>Results by held-out experiment</h2><div class="card">',

    safe_table(by_group),

    '</div><h2>Training-only feature spaces</h2><div class="card">',

    safe_table(features),

    '</div><h2>All 36 held-out sample decisions</h2><div class="card">',

    safe_table(df[cols]),

    f'''</div>
<p class="meta">
Frozen v0.7 source:
<code>{html.escape(str(source_path))}</code>
<br>
Metadata SHA-256:
<code>{v07.prior.EXPECTED_METADATA_SHA}</code>
<br>
Counts SHA-256:
<code>{v07.prior.EXPECTED_COUNTS_SHA}</code>
</p>
</main>
</body>
</html>'''
    ]

    return ''.join(html_parts)


def run(out: Path, log: v07.prior.Log):

    log.write('=' * 78)

    log.write(
        'PAPER 7 | ONSETAUDIT v0.8 | COMMON-SPACE 12-WAY COMPETITION'
    )

    log.write(
        'OFFLINE | MAIN ONLY | NO FLAGS | '
        'v0.4 FROZEN; NEW POST-HOC MODEL SEPARATELY LABELLED'
    )

    log.write('=' * 78)

    verified_v07 = verify_v07(log)

    (
        head,
        ev,
        v04_path,
        v05_path,
        metadata,
        counts
    ) = v07.prior.inputs_and_reports(log)

    ds, model = v07.prior.dataset_and_replay(
        ev, metadata, counts, log
    )

    # Verify the v0.7 cross-program frozen behavior.
    old_rows, old_cross, _, _ = v07.main_analysis(
        ds, model, log
    )

    require(
        len(old_rows) == 72
        and len(old_cross) == 24
        and int(old_cross.other_program_accepted.sum()) == 24
        and int(
            old_cross.other_program_duration_correct.sum()
        ) == 17,
        'v0.7 score replay changed'
    )

    require(
        int(model['correct_and_accepted']) == 22
        and int(model['vehicle_rejected']) == 10,
        'Frozen v0.4 result was unexpectedly modified'
    )

    log.write(
        '[REPLAY]',
        'Frozen v0.4 22/24 & 10/12; '
        'v0.7 cross 24/24 & 17/24 EXACT'
    )

    df, fold_df, confusion, totals = run_joint(
        ds, model, log
    )

    df.to_csv(
        out / 'joint_36_heldout_predictions.csv',
        index=False
    )

    fold_df.to_csv(
        out / 'training_only_feature_union.csv',
        index=False
    )

    confusion.to_csv(
        out / 'three_class_confusion.csv'
    )

    dashboard = out / 'onsetaudit_v08_dashboard.html'

    dashboard.write_text(
        make_dashboard(
            df,
            fold_df,
            confusion,
            totals,
            verified_v07
        ),
        encoding='utf-8'
    )

    summary = {
        'audit_version': AUDIT_VERSION,

        'status':
            'POST_HOC_SHARED_FEATURE_JOINT_OBJECTIVE_NOT_FROZEN_MODEL',

        'dataset': 'GSE194200',

        'source_git_head_at_run': head,

        'source_v04_report': str(v04_path),
        'source_v04_sha256':
            v07.prior.digest(v04_path),

        'source_v05_report': str(v05_path),
        'source_v05_sha256':
            v07.prior.digest(v05_path),

        'source_v07_report': str(verified_v07),
        'source_v07_sha256':
            v07.prior.digest(verified_v07),

        'metadata_sha256':
            v07.prior.EXPECTED_METADATA_SHA,

        'counts_sha256':
            v07.prior.EXPECTED_COUNTS_SHA,

        'frozen_active_correct_and_accepted': 22,

        'frozen_dmso_rejected_distinct': 10,

        'cross_program_accepted': 24,

        'cross_program_duration_correct': 17,

        'joint_objective':
            'min over 4 times of DMSO null or drug ridge fit '
            'per class; min of 3 class losses',

        'feature_space':
            'per fold union of frozen training-only '
            'drug-selected 200 gene sets',

        'shared_scale':
            'sqrt(mean of drug-specific train-between-experiment '
            'contrast variances + 0.25)',

        'ridge_penalty':
            'RIDGE(0.25) * number of common genes',

        'true_time_used_in_prediction': False,

        'true_drug_used_in_prediction': False,

        'threshold_fitted_on_heldout': False,

        'results': totals,

        'per_fold_feature_sets':
            fold_df.to_dict(orient='records'),

        'interpretation_limits': [
            'New post-hoc objective on previously inspected '
            'one cell-line cohort',

            'Not a replacement for frozen OnsetAudit v0.4 '
            'or performance claim',

            'Only 3 held-out experiments, each with four '
            'times per treatment',

            'No drug posterior probabilities, calibration, '
            'prospective sample, or biological onset truth',

            'Cross-program active samples are not negative controls',

            'No generalization to endometrium/PGR or '
            'independent platform has been shown',

            'Avoid selecting thresholds, genes or objective '
            'variants based on these results',
        ],

        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }

    v07.prior.save_json(
        out / 'audit_summary.json',
        summary
    )

    (out / 'README.txt').write_text(
        'OnsetAudit v0.8 retrospective shared-feature '
        '12-way candidate audit.\n'
        'Frozen v0.4 22/24 and 10/12 results and '
        'v0.7 24/24 / 17/24 reproduced.\n'
        'This is a NEW exploratory model, not the v0.4 '
        'unchanged gate and NOT external validation.\n'
        'All 12 drug/time options are compared with '
        'common genes/scales/loss, no true time used '
        'for inference.\n'
        'Only training fold data determine gene features, '
        'response means and variance scales.\n'
        'Do not promote to main app or change any '
        'thresholds/features without external validation.\n'
        'Source counts and metadata remain local, identified '
        'by SHA-256 in audit_summary.json.\n',
        encoding='utf-8'
    )

    shutil.copyfile(
        Path(__file__),
        out / Path(__file__).name
    )

    log.write('[OUTPUT]', out)
    log.write('[DASHBOARD]', dashboard)

    log.write(
        '[POLICY]',
        'All v0.4–v0.7 files and original data unchanged; '
        'new model diagnostic only'
    )


def main():

    stamp = datetime.now(timezone.utc).strftime(
        '%Y%m%dT%H%M%SZ'
    )

    REPORT_BASE.mkdir(
        parents=True,
        exist_ok=True
    )

    out = REPORT_BASE / f'v08_shared_competition_{stamp}'

    out.mkdir(
        parents=True,
        exist_ok=False
    )

    log = v07.prior.Log(
        out / 'console.log'
    )

    try:

        run(out, log)

        log.write(
            '[READY]',
            'v0.8 shared-feature 12-way diagnostic complete'
        )

    except BaseException:

        log.write(
            '[STOP]',
            'v0.8 failed; frozen app, old data '
            'and reports unchanged'
        )

        log.write(traceback.format_exc())

        raise

    finally:

        log.close()

    paths = sorted(
        p for p in out.iterdir()
        if p.is_file()
        and p.name not in (
            'sha256_manifest.json',
            'onsetaudit_v08_audit.zip'
        )
    )

    v07.prior.save_json(
        out / 'sha256_manifest.json',
        {
            p.name: v07.prior.digest(p)
            for p in paths
        }
    )

    zip_path = out / 'onsetaudit_v08_audit.zip'

    with zipfile.ZipFile(
        zip_path,
        'w',
        zipfile.ZIP_DEFLATED
    ) as archive:

        for file in sorted(out.iterdir()):

            if file.is_file() and file != zip_path:

                archive.write(
                    file,
                    arcname=file.name
                )

    print('[ZIP]', zip_path, flush=True)

    if sys.platform == 'darwin':

        subprocess.run(
            [
                'open',
                str(
                    out / 'onsetaudit_v08_dashboard.html'
                )
            ],
            check=False
        )


if __name__ == '__main__':

    try:
        main()

    except Exception as exc:

        print(
            '[STOP]',
            type(exc).__name__ + ':',
            exc,
            file=sys.stderr,
            flush=True
        )

        sys.exit(1)
