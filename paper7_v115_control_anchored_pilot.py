#!/usr/bin/env python3
"""Paper 7 V115. Paired-vehicle reference pilot, real experimental durations.
This is a development-set audit, NOT prospective validation or a new-method claim.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, math, platform, shutil, sys, traceback, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper7_gse194200_paired_vehicle_v115' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)
LOG = []
DRUGS = ('SW016789', 'Thapsigargin')
HOURS = (1, 2, 6, 24)
PAIRS = ((1,2), (2,6), (6,24))
METHODS = (
    'vehicle_reference_only',
    'paired_effect_fixed',
    'paired_effect_ridge_amplitude'
)
TOP = 200
RIDGE_PER_GENE = 0.25
VAR_FLOOR = 0.25
XLSX_SHA = '7d220b4ef8e02deda74c1f51295e9636634796a086580323ee315e58d8f5ea7b'
SERIES_SHA = '7aef2271b0d193a059cf95ab0356d32ea3b643495fdf97d7f01267d5f24f6543'

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha(p):
    h = sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''):
            h.update(x)
    return h.hexdigest()

def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(
        OUT/name, index=False,
        encoding='utf-8-sig', float_format='%.15g'
    )
    say('[OUTPUT] {} | {} rows'.format(name, len(frame)))
    return frame

def load():
    base = ROOT/'paper7_gse194200_stress_v114'
    required = (
        '11_sha256_inventory.csv',
        '10_manifest.json',
        'input_v113_09_protocol_manifest.json',
        'input_v113_01_verified_sample_mapping.csv',
        'input_v113_counts_logCPM.npz',
        '02_cross_drug_predictions.csv',
        '05_vehicle_gate_predictions.csv'
    )
    runs = sorted(
        (p for p in base.glob('run_*') if p.is_dir()
         and all((p/n).is_file() for n in required)),
        reverse=True
    )
    need(runs, 'Verified V114 archive not found')
    run = runs[0]

    inv = pd.read_csv(
        run/'11_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    need(
        {'filename','bytes','sha256'}.issubset(inv.columns)
        and inv.filename.is_unique,
        'V114 checksum inventory invalid'
    )
    for r in inv.itertuples(index=False):
        name = str(r.filename)
        p = run/name
        need(
            Path(name).name == name and p.is_file()
            and p.stat().st_size == int(r.bytes)
            and sha(p) == str(r.sha256),
            'V114 artifact changed: '+name
        )

    m114 = json.loads(
        (run/'10_manifest.json').read_text('utf-8')
    )
    m113 = json.loads(
        (run/'input_v113_09_protocol_manifest.json').read_text('utf-8')
    )
    need(
        m114.get('version') == 'v114'
        and m113.get('version') == 'v113'
        and m113.get('raw_xlsx_sha256') == XLSX_SHA
        and m113.get('geo_matrix_sha256') == SERIES_SHA,
        'V113/V114 source provenance mismatch'
    )
    need(
        m114.get('source_v113_sha256')
        == sha(run/'input_v113_09_protocol_manifest.json'),
        'V113 manifest snapshot hash mismatch'
    )

    mapping = pd.read_csv(
        run/'input_v113_01_verified_sample_mapping.csv',
        encoding='utf-8-sig',
        dtype={'gsm': str}
    )
    with np.load(
        run/'input_v113_counts_logCPM.npz',
        allow_pickle=False
    ) as z:
        counts = z['counts'].astype(float)
        archived = z['log2CPM'].astype(float)

    need(
        counts.shape == (24421,39)
        and archived.shape == counts.shape,
        'Expected exactly 24421 x 39 raw-count matrix'
    )
    X = np.log2(
        1+counts*1e6/counts.sum(axis=0)[None,:]
    )
    need(
        np.isfinite(X).all()
        and np.max(np.abs(archived-X)) < 3e-6,
        'V113 stored normalization mismatch'
    )

    expected = {
        (e,'Baseline',0) for e in (1,2,3)
    } | {
        (e,d,h)
        for e in (1,2,3)
        for d in ('DMSO',)+DRUGS
        for h in HOURS
    }
    keys = [
        (int(r.experiment),str(r.treatment),int(r.hours))
        for r in mapping.itertuples(index=False)
    ]
    need(
        len(mapping) == 39
        and len(set(keys)) == 39
        and set(keys) == expected
        and mapping.gsm.is_unique,
        'Sample metadata do not match GEO experimental design'
    )
    lookup = {k:i for i,k in enumerate(keys)}

    cross = pd.read_csv(
        run/'02_cross_drug_predictions.csv',
        encoding='utf-8-sig',
        dtype={'gsm':str}
    )
    gate = pd.read_csv(
        run/'05_vehicle_gate_predictions.csv',
        encoding='utf-8-sig',
        dtype={'gsm':str}
    )
    need(
        len(cross) == 96 and len(gate) == 72,
        'Incomplete V114 prediction archive'
    )
    cross = cross[
        cross.representation == 'raw_logCPM'
    ].copy()
    gate = gate[
        gate.representation == 'raw_logCPM'
    ].copy()
    gate['called_active'] = (
        gate.called_active.astype(str).str.lower() == 'true'
    )
    gate['true_active'] = (
        gate.true_active.astype(str).str.lower() == 'true'
    )
    need(
        len(cross)==48 and len(gate)==36
        and gate.gsm.is_unique
        and set(gate.gsm)==set(
            mapping.gsm[
                ~(mapping.treatment=='Baseline')
            ]
        ),
        'Raw-logCPM V114 predictions incomplete'
    )

    say(
        '[V114 VERIFIED] {} checksum-verified files '
        '| 24421 genes | 39 samples'.format(len(inv))
    )
    say(
        '[TARGET] Experimentally recorded duration '
        'of exposure, NOT transcript onset'
    )
    return run, counts, X, mapping, lookup, cross, gate

def learn(counts, X, lookup, heldout, source):
    train_exp = tuple(
        e for e in (1,2,3) if e != heldout
    )
    v = np.stack([
        np.column_stack([
            X[:,lookup[(e,source,h)]]
            - X[:,lookup[(e,'DMSO',h)]]
            for h in HOURS
        ])
        for e in train_exp
    ], axis=1)

    need(
        v.shape == (24421,2,4),
        'Training contrast shape changed'
    )
    mean_by_duration = v.mean(axis=1)
    between_var = (
        0.5*(v[:,0,:]-v[:,1,:])**2
    ).mean(axis=1)
    signal_var = np.var(
        mean_by_duration,axis=1
    )
    measure = signal_var/(between_var+VAR_FLOOR)

    train_idx = [
        lookup[(e,d,h)]
        for e in train_exp
        for d in (source,'DMSO')
        for h in HOURS
    ]
    eligible = (
        (counts[:,train_idx]>=10).sum(axis=1)>=3
    ) & np.isfinite(measure)
    eligible &= (
        np.isfinite(between_var)
        & (signal_var>1e-12)
    )
    ids = np.flatnonzero(eligible)

    need(
        len(ids)>=TOP,
        'Fewer than 200 training-only eligible genes'
    )
    selection = ids[
        np.argsort(
            -measure[ids],kind='stable'
        )[:TOP]
    ]
    scale = np.sqrt(
        between_var[selection]+VAR_FLOOR
    )
    templates = mean_by_duration[selection,:]

    return (
        selection,scale,templates,measure,len(ids)
    )

def candidate_scores(
    X,lookup,heldout,sample,selection,scale,templates
):
    obs = X[selection,sample]
    scores = {method:{} for method in METHODS}
    amps = {}
    n = len(selection)

    for j,h in enumerate(HOURS):
        vehicle = X[
            selection,
            lookup[(heldout,'DMSO',h)]
        ]
        residual = (obs-vehicle)/scale
        effect = templates[:,j]/scale

        d0 = float(np.mean(residual**2))
        d1 = float(np.mean((residual-effect)**2))

        lam = RIDGE_PER_GENE*n
        denom = float(effect@effect+lam)
        amplitude = max(
            0.,
            float(residual@effect+lam)/denom
        )
        dridge = float((
            np.sum(
                (residual-amplitude*effect)**2
            )
            +lam*(amplitude-1.)**2
        )/n)

        scores['vehicle_reference_only'][h] = d0
        scores['paired_effect_fixed'][h] = d1
        scores['paired_effect_ridge_amplitude'][h] = dridge
        amps[h] = amplitude

    return scores,amps

def calculate(counts,X,meta,ix):
    records = []
    scores_rows = []
    pair_rows = []
    selections = []

    for hold in (1,2,3):
        for source in DRUGS:
            (
                selected,scale,templates,
                ranking,n_eligible
            ) = learn(counts,X,ix,hold,source)

            selections.extend(
                dict(
                    heldout_experiment=hold,
                    source_drug=source,
                    rank=r+1,
                    gene_index=int(g),
                    training_only_effect_rank=float(ranking[g]),
                    n_eligible=n_eligible
                )
                for r,g in enumerate(selected)
            )

            for target in DRUGS:
                for real_h in HOURS:
                    sample = ix[(hold,target,real_h)]
                    gsm = str(meta.iloc[sample].gsm)

                    scores,amps = candidate_scores(
                        X,ix,hold,sample,
                        selected,scale,templates
                    )

                    for method in METHODS:
                        order = sorted(
                            HOURS,
                            key=lambda h:(
                                scores[method][h],h
                            )
                        )
                        chosen = int(order[0])
                        is_correct = chosen==real_h

                        records.append(dict(
                            heldout_experiment=hold,
                            source_drug=source,
                            target_drug=target,
                            transfer=(
                                'same_drug'
                                if source==target
                                else 'cross_drug'
                            ),
                            gsm=gsm,
                            true_hours=real_h,
                            method=method,
                            predicted_hours=chosen,
                            correct=is_correct,
                            absolute_error_hours=abs(
                                chosen-real_h
                            ),
                            top2_contains_true=(
                                real_h in order[:2]
                            ),
                            score_margin=float(
                                scores[method][order[1]]
                                -scores[method][order[0]]
                            ),
                            best_score=float(
                                scores[method][order[0]]
                            ),
                            test_vehicle_grid_available=True
                        ))

                        for h in HOURS:
                            scores_rows.append(dict(
                                heldout_experiment=hold,
                                source_drug=source,
                                target_drug=target,
                                gsm=gsm,
                                true_hours=real_h,
                                method=method,
                                candidate_hours=h,
                                score=scores[method][h],
                                fitted_amplitude=(
                                    amps[h]
                                    if method==
                                    'paired_effect_ridge_amplitude'
                                    else np.nan
                                ),
                                test_vehicle_gsm=str(
                                    meta.iloc[
                                        ix[(hold,'DMSO',h)]
                                    ].gsm
                                )
                            ))

                        for a,b in PAIRS:
                            if real_h not in (a,b):
                                continue
                            yh = min(
                                (a,b),
                                key=lambda h:(
                                    scores[method][h],h
                                )
                            )
                            pair_rows.append(dict(
                                heldout_experiment=hold,
                                source_drug=source,
                                target_drug=target,
                                transfer=(
                                    'same_drug'
                                    if source==target
                                    else 'cross_drug'
                                ),
                                method=method,
                                pair='{}v{}'.format(a,b),
                                gsm=gsm,
                                true_hours=real_h,
                                predicted_hours=int(yh),
                                correct=bool(yh==real_h)
                            ))

            say(
                '[V115 FOLD {} | source={}] '
                'training experiments=2, '
                'eligible genes={}, top={}'
                .format(
                    hold,source,n_eligible,len(selected)
                )
            )

    rec = save(
        '02_paired_vehicle_predictions.csv',
        records
    )
    save(
        '03_candidate_time_scores.csv',
        scores_rows
    )
    pairs = save(
        '04_adjacent_pair_predictions.csv',
        pair_rows
    )
    save(
        '05_train_only_gene_selections.csv',
        selections
    )
    need(
        len(rec)==144
        and len(scores_rows)==576
        and len(pairs)==216
        and len(selections)==1200,
        'Incomplete paired reference audit'
    )
    return rec,pairs

def summarize(records,pairs,cross,gate):
    metrics = []

    for (method,transfer),g in records.groupby(
        ['method','transfer']
    ):
        need(
            len(g)==24,
            'Expected 24 heldout exposures per method/transfer'
        )
        metrics.append(dict(
            method=method,
            transfer=transfer,
            n=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.absolute_error_hours.mean()
            ),
            top2_coverage=float(
                g.top2_contains_true.mean()
            )
        ))
        say(
            '[PAIRED PILOT | {} | {}] correct={}/24 '
            '| MAE={:.3f}h | top2={}/24'
            .format(
                method,transfer,
                int(g.correct.sum()),
                g.absolute_error_hours.mean(),
                int(g.top2_contains_true.sum())
            )
        )

    save('06_paired_method_metrics.csv',metrics)

    old = []
    for transfer in ('same_drug','cross_drug'):
        subset = cross[
            cross.transfer==transfer
        ]
        need(
            len(subset)==24,
            'Incomplete V114 historical comparison'
        )
        old.append(dict(
            method='V114_raw_centroid',
            transfer=transfer,
            n=24,
            correct=int(
                subset.correct.astype(str)
                .str.lower().eq('true').sum()
            ),
            MAE_hours=float(
                subset.absolute_error_hours.mean()
            )
        ))
        say(
            '[V114 REFERENCE | {}] correct={}/24 '
            '| MAE={:.3f}h'
            .format(
                transfer,
                old[-1]['correct'],
                old[-1]['MAE_hours']
            )
        )

    save(
        '07_frozen_V114_baseline_metrics.csv',
        old
    )

    pair_summary = []
    for (method,transfer,pair),sub in pairs.groupby(
        ['method','transfer','pair']
    ):
        need(
            len(sub)==12,
            'Pair sample count unexpected'
        )
        pair_summary.append(dict(
            method=method,
            transfer=transfer,
            pair=pair,
            n=12,
            correct=int(sub.correct.sum()),
            accuracy=float(sub.correct.mean())
        ))
        say(
            '[V115 PAIR | {} | {} | {}] correct={}/12'
            .format(
                method,transfer,pair,
                int(sub.correct.sum())
            )
        )

    save(
        '08_pairwise_resolution_metrics.csv',
        pair_summary
    )

    rawgate = gate.set_index('gsm')
    accepted = rawgate.loc[
        rawgate.true_active
    ]
    negatives = rawgate.loc[
        ~rawgate.true_active
    ]
    need(
        len(accepted)==24
        and len(negatives)==12,
        'DMSO gate counts differ'
    )
    reject = int(
        (~negatives.called_active).sum()
    )
    accept = int(
        accepted.called_active.sum()
    )
    need(
        reject==11 and accept==19,
        'V114 gate replay mismatch'
    )

    end2end = []
    matched = []

    for method,g in records[
        records.transfer=='same_drug'
    ].groupby('method'):
        need(
            len(g)==24 and g.gsm.is_unique,
            'Duplicate same-drug predictions'
        )
        for row in g.itertuples(index=False):
            gateflag = bool(
                rawgate.loc[
                    row.gsm,'called_active'
                ]
            )
            end2end.append(dict(
                method=method,
                gsm=row.gsm,
                heldout_experiment=row.heldout_experiment,
                true_hours=row.true_hours,
                predicted_hours=row.predicted_hours,
                accepted_active=gateflag,
                correct_and_accepted=bool(
                    gateflag and row.correct
                )
            ))

        results = [
            p for p in end2end
            if p['method']==method
        ]
        ncorrect = sum(
            p['correct_and_accepted']
            for p in results
        )
        matched.append(dict(
            method=method,
            DMSO_rejected=reject,
            DMSO_total=12,
            active_accepted=accept,
            active_total=24,
            time_correct_and_accepted=ncorrect,
            all_active_correct_fraction=ncorrect/24
        ))
        say(
            '[V115 + FROZEN DMSO GATE | {}] '
            'DMSO rejected={}/12 '
            '| active accepted={}/24 '
            '| correct AND accepted={}/24'
            .format(
                method,reject,accept,ncorrect
            )
        )

    save(
        '09_end_to_end_active_decisions.csv',
        end2end
    )
    save(
        '10_end_to_end_gate_metrics.csv',
        matched
    )

def main():
    say('='*96)
    say(
        'PAPER 7 | V115 | PAIRED-VEHICLE TEMPORAL '
        'RESPONSE: REAL-DATA DEVELOPMENT PILOT'
    )
    say(
        'OFFLINE | REAL RNA-SEQ | HELDOUT EXPERIMENT '
        '| NO BIOLOGICAL ONSET TRUTH | NO GIT'
    )
    say('='*96)

    need(
        len(sys.argv)==1,
        'Run as python script.py without flags'
    )

    (
        run,counts,X,meta,
        lookup,cross,gate
    ) = load()

    protocol = dict(
        version='v115',
        source_v114_manifest_sha256=sha(
            run/'10_manifest.json'
        ),
        target=(
            'experimentally recorded exposure duration, '
            'not transcription onset'
        ),
        training=(
            'one source drug, two experiments'
        ),
        heldout=(
            'different experiment, same or opposite drug'
        ),
        reference_inputs=(
            'FOUR time-labelled DMSO samples '
            'from the heldout experiment'
        ),
        key_limit=(
            'cannot predict from an isolated sample '
            'without those four controls'
        ),
        features=TOP,
        selection=(
            'training-only across-time contrast '
            'variance / (batch variance + 0.25)'
        ),
        methods=METHODS,
        amplitude_ridge_per_gene=RIDGE_PER_GENE,
        variance_floor=VAR_FLOOR,
        set_rule=(
            'top two candidate durations '
            '(descriptive, NOT coverage guarantee)'
        ),
        gate=(
            'frozen V114 raw-logCPM gate, '
            'no test-time DMSO use in that gate'
        ),
        no_new_test_data=True,
        method_development_on_previously_examined_dataset=True,
        novelty_unproven=True,
        noise_only_sigma_unavailable=True,
        no_git=True,
        no_network=True,
        script_sha256=sha(__file__)
    )

    (
        OUT/'01_protocol.json'
    ).write_text(
        json.dumps(protocol,indent=2)+'\n',
        encoding='utf-8'
    )

    pred,pairs = calculate(
        counts,X,meta,lookup
    )
    summarize(
        pred,pairs,cross,gate
    )

    notes = """# V115: paired-vehicle duration prediction

This is an exploratory development analysis.

All four time-labelled DMSO controls from the
heldout experiment are required.

The target active sample's true exposure duration
is hidden from the prediction algorithm.

For each candidate duration, the method compares
the test active-minus-candidate-DMSO contrast
with a training-only drug-minus-DMSO template.

A vehicle-only comparator receives exactly the
same DMSO reference samples.

Gene filtering, feature selection, response
templates and variability estimates use only
the two training experiments.

A DMSO sample must not be tested against itself
as its own time-matched reference: that would
create circular negative-control evaluation.

Consequently, the V114 sample-only DMSO gate
is reused unchanged for end-to-end analysis.

The top-two prediction set is descriptive,
NOT a calibrated confidence set.

GSE194200 has already been examined in V113
and V114, so this is NOT a genuinely untouched
independent validation dataset.

Exposure duration is NOT a biological onset
or experimentally imposed transcriptional lag.

The three MIN6 experiments do not establish
transferability to primary human endometrium.

Methodological novelty has not been established.
A new independent dataset and published-method
comparisons remain required.
"""

    (OUT/'11_methods_limits.md').write_text(
        notes,encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v115.py'
    )

    for src,dst in (
        (
            '10_manifest.json',
            'input_v114_manifest.json'
        ),
        (
            '02_cross_drug_predictions.csv',
            'input_v114_cross_drug.csv'
        ),
        (
            '05_vehicle_gate_predictions.csv',
            'input_v114_vehicle_gate.csv'
        ),
        (
            'input_v113_01_verified_sample_mapping.csv',
            'input_sample_mapping.csv'
        ),
        (
            'input_v113_counts_logCPM.npz',
            'input_counts_logCPM.npz'
        )
    ):
        shutil.copy2(run/src,OUT/dst)

    (
        OUT/'12_manifest.json'
    ).write_text(
        json.dumps(dict(
            version='v115',
            source_v114_manifest_sha256=sha(
                run/'10_manifest.json'
            ),
            script_sha256=sha(__file__),
            n_real_samples=39,
            n_outcomes=len(pred),
            independent_external_validation=False,
            unseen_dataset_required=True,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            no_git=True,
            no_network=True
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[BOUNDARY] Privileged heldout DMSO reference grid; '
        'not deployable for an isolated query'
    )
    say(
        '[GATE] No new biological delay detection claim; '
        'novelty and independent validation NOT ESTABLISHED'
    )
    say(
        '[COMPLETE] Real-data paired-vehicle pilot archived'
    )

def finish(ok):
    say(
        '[RESULT] '+(
            'SUCCESS' if ok else 'FAILED'
        )
    )
    say(
        '[REPORT] '+str(
            OUT/'console_summary.txt'
        )
    )

    archive = OUT/'portable_report_v115.zip'
    say('[ZIP] '+str(archive))

    (
        OUT/'console_summary.txt'
    ).write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    entries = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            archive.name,
            '13_sha256_inventory.csv'
        ):
            entries.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(entries).to_csv(
        OUT/'13_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive,'w',zipfile.ZIP_DEFLATED,
        compresslevel=4
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

    say(
        '[ZIP COMPLETE] code, input counts, '
        'decisions, checksums, logs'
    )

if __name__=='__main__':
    ok=True
    try:
        main()
    except Exception:
        ok=False
        say(
            '[ERROR]\n'+traceback.format_exc()
        )
    try:
        finish(ok)
    except Exception:
        ok=False
        print(
            '[ARCHIVE ERROR]\n'+traceback.format_exc(),
            flush=True
        )
    if not ok:
        sys.exit(1)
