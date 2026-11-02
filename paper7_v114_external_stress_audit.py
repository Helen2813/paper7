#!/usr/bin/env python3
"""Paper 7 V114: GSE194200 external stress audit, no model novelty claim."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, platform, sys, traceback, zipfile, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper7_gse194200_stress_v114'/(
    'run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)

LOG=[]
DRUGS=('SW016789','Thapsigargin')
TIMES=(1,2,6,24)
PAIRS=((1,2),(2,6),(6,24))
MODES=('raw_logCPM','within_experiment_baseline')
TOP=200

XLSX_SHA='7d220b4ef8e02deda74c1f51295e9636634796a086580323ee315e58d8f5ea7b'
SERIES_SHA='7aef2271b0d193a059cf95ab0356d32ea3b643495fdf97d7f01267d5f24f6543'

def say(s):
    s=str(s)
    print(s,flush=True)
    LOG.append(s)

def need(ok,msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def save(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(
        OUT/name,index=False,
        encoding='utf-8-sig',float_format='%.15g'
    )
    say(f'[OUTPUT] {name} | {len(d)} rows')
    return d

def load():
    home=ROOT/'paper7_gse194200_real_baseline_v113'
    required=(
        '11_sha256_inventory.csv',
        '09_protocol_manifest.json',
        '01_verified_sample_mapping.csv',
        '02_gene_annotations.csv',
        '04_gene_counts_and_logCPM.npz',
        '05_heldout_sample_predictions.csv'
    )
    runs=sorted(
        (
            r for r in home.glob('run_*')
            if r.is_dir()
            and all((r/f).is_file() for f in required)
        ),
        reverse=True
    )
    need(runs,'Complete V113 archive missing')
    run=runs[0]

    inv=pd.read_csv(
        run/'11_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    need(
        {'filename','bytes','sha256'}.issubset(inv.columns)
        and inv.filename.is_unique,
        'Invalid archived inventory'
    )

    for z in inv.itertuples(index=False):
        n=str(z.filename)
        p=run/n
        need(
            n==Path(n).name
            and p.is_file()
            and p.stat().st_size==int(z.bytes)
            and sha(p)==str(z.sha256),
            'V113 source changed: '+n
        )

    man=json.loads(
        (run/'09_protocol_manifest.json').read_text('utf-8')
    )
    need(
        man.get('version')=='v113'
        and man.get('samples')==39
        and man.get('raw_xlsx_sha256')==XLSX_SHA
        and man.get('geo_matrix_sha256')==SERIES_SHA,
        'V113 source provenance mismatch'
    )

    mapping=pd.read_csv(
        run/'01_verified_sample_mapping.csv',
        encoding='utf-8-sig',dtype={'gsm':str}
    )
    annotation=pd.read_csv(
        run/'02_gene_annotations.csv',
        encoding='utf-8-sig',
        dtype={'gene_symbol':str,'entrez_id':str}
    )
    historical=pd.read_csv(
        run/'05_heldout_sample_predictions.csv',
        encoding='utf-8-sig',dtype={'gsm':str}
    )

    with np.load(
        run/'04_gene_counts_and_logCPM.npz',
        allow_pickle=False
    ) as npz:
        counts=npz['counts'].astype(np.float64)
        stored=npz['log2CPM'].astype(np.float64)

    need(
        counts.shape==(24421,39)
        and stored.shape==counts.shape,
        'Counts shape mismatch'
    )
    need(
        len(mapping)==39
        and mapping.gsm.is_unique
        and len(annotation)==24421,
        'Sample/gene mapping mismatch'
    )
    need(
        np.isfinite(counts).all()
        and (counts>=0).all()
        and np.max(np.abs(counts-np.rint(counts)))<1e-6,
        'Counts are not nonnegative integers'
    )

    X=np.log2(
        1.0+counts*1e6/counts.sum(axis=0)[None,:]
    )
    need(
        np.max(np.abs(X-stored))<3e-6,
        'Saved V113 logCPM differs'
    )

    ex=mapping.experiment.to_numpy(int)
    tm=mapping.hours.to_numpy(int)
    treatment=mapping.treatment.to_numpy(str)

    expected={
        (e,'Baseline',0) for e in (1,2,3)
    } | {
        (e,t,h)
        for e in (1,2,3)
        for t in ('DMSO',)+DRUGS
        for h in TIMES
    }

    need(
        len(set(zip(ex,treatment,tm)))==39
        and set(zip(ex,treatment,tm))==expected,
        'Experiment/treatment/time cells mismatch'
    )
    need(
        len(historical)==96
        and len(set(historical.method))==3,
        'V113 predictions missing'
    )

    act=historical[
        historical.is_active.astype(str).str.lower()=='true'
    ]

    for method,ncorrect in (
        ('constant_2h',6),
        ('pooled_centroid',11),
        ('drug_aware_centroid',12)
    ):
        part=act[act.method==method]
        need(
            len(part)==24
            and (
                part.predicted_hours
                ==part.true_exposure_hours
            ).sum()==ncorrect,
            'Historical metrics mismatch: '+method
        )

    say(
        f'[V113 VERIFIED] inventory={len(inv)} '
        '| 24421 genes x 39 real samples'
    )
    say(
        '[V113 REPLAY] baseline correct=6/24 '
        '| pooled=11/24 | drug-aware=12/24'
    )
    return run,counts,X,mapping,annotation,historical

def indices(meta):
    d={}
    for i,r in enumerate(meta.itertuples(index=False)):
        d[(
            int(r.experiment),
            str(r.treatment),
            int(r.hours)
        )]=i
    return d

def train_transform(X,index,mode):
    if mode=='raw_logCPM':
        return X

    Z=X.copy()
    for exp in (1,2,3):
        b=X[:,index[(exp,'Baseline',0)]]
        for (e,_,_),i in index.items():
            if e==exp:
                Z[:,i]=X[:,i]-b
    return Z

def top_features(counts,V,train,n_min):
    eligible=np.flatnonzero(
        (
            (counts[:,train]>=10).sum(axis=1)>=n_min
        ) & (
            np.var(V[:,train],axis=1)>1e-10
        )
    )
    need(
        len(eligible)>=TOP,
        'Insufficient eligible training genes'
    )
    vari=np.var(V[eligible][:,train],axis=1)
    selected=eligible[
        np.argsort(-vari,kind='stable')[:TOP]
    ]
    return selected,len(eligible)

def z_transform(V,selected,train):
    mu=V[selected][:,train].mean(axis=1)
    sd=V[selected][:,train].std(axis=1)
    return (
        (V[selected].T-mu)/np.maximum(sd,.5)
    ).astype(np.float64)

def predict_centroid(Z,train,test,times):
    centroids={
        h:Z[train[times[train]==h]].mean(axis=0)
        for h in TIMES
    }
    need(
        all(np.isfinite(v).all()
            for v in centroids.values()),
        'Centroid not finite'
    )
    out=[]
    for i in test:
        distances={
            h:float(np.sum(
                (Z[i]-centroids[h])**2
            ))
            for h in TIMES
        }
        pred=min(
            TIMES,key=lambda t:(distances[t],t)
        )
        out.append((i,pred,distances))
    return out

def cross_drug(counts,X,meta,ix):
    ex=meta.experiment.to_numpy(int)
    treat=meta.treatment.to_numpy(str)
    tm=meta.hours.to_numpy(int)

    outputs=[]
    binary=[]
    features=[]

    for mode in MODES:
        V=train_transform(X,ix,mode)

        for hold in (1,2,3):
            for source in DRUGS:
                train=np.flatnonzero(
                    (ex!=hold)&(treat==source)
                )
                need(
                    len(train)==8,
                    'Source training size mismatch'
                )

                selected,n_eligible=top_features(
                    counts,V,train,2
                )
                Z=z_transform(V,selected,train)

                for target in DRUGS:
                    test=np.flatnonzero(
                        (ex==hold)&(treat==target)
                    )
                    need(
                        len(test)==4,
                        'Target test size mismatch'
                    )

                    pred=predict_centroid(
                        Z,train,test,tm
                    )

                    for i,y,d in pred:
                        outputs.append(dict(
                            representation=mode,
                            heldout_experiment=hold,
                            source_drug=source,
                            target_drug=target,
                            transfer=(
                                'same_drug'
                                if source==target
                                else 'cross_drug'
                            ),
                            gsm=meta.iloc[i].gsm,
                            true_hours=int(tm[i]),
                            predicted_hours=int(y),
                            correct=bool(y==tm[i]),
                            absolute_error_hours=abs(
                                int(y)-int(tm[i])
                            ),
                            selected_training_genes=TOP,
                            n_eligible_training_genes=n_eligible,
                            margin_sse=(
                                sorted(d.values())[1]
                                -sorted(d.values())[0]
                            )
                        ))

                    for a,b in PAIRS:
                        ca=Z[
                            train[tm[train]==a]
                        ].mean(axis=0)
                        cb=Z[
                            train[tm[train]==b]
                        ].mean(axis=0)

                        for i in test:
                            if int(tm[i]) not in (a,b):
                                continue

                            da=float(np.sum(
                                (Z[i]-ca)**2
                            ))
                            db=float(np.sum(
                                (Z[i]-cb)**2
                            ))
                            y=a if da<=db else b

                            binary.append(dict(
                                representation=mode,
                                heldout_experiment=hold,
                                source_drug=source,
                                target_drug=target,
                                transfer=(
                                    'same_drug'
                                    if source==target
                                    else 'cross_drug'
                                ),
                                true_hours=int(tm[i]),
                                pair=f'{a}v{b}',
                                predicted_hours=y,
                                correct=bool(y==tm[i]),
                                gsm=meta.iloc[i].gsm
                            ))

                    features.extend(
                        dict(
                            representation=mode,
                            heldout_experiment=hold,
                            source_drug=source,
                            rank=j+1,
                            gene_index=int(g),
                            n_eligible=n_eligible
                        )
                        for j,g in enumerate(selected)
                    )

        say(
            f'[CROSS-DRUG] {mode} | train 2 experiments, '
            'hold out experiment AND drug'
        )

    result=save(
        '02_cross_drug_predictions.csv',
        outputs
    )
    pair=save(
        '03_adjacent_pair_binary_predictions.csv',
        binary
    )
    save(
        '04_source_drug_training_features.csv',
        features
    )

    need(
        len(result)==96 and len(pair)==144,
        'Cross-drug sample count mismatch'
    )

    for mode in MODES:
        for transfer in (
            'same_drug','cross_drug'
        ):
            part=result[
                (result.representation==mode)
                &(result.transfer==transfer)
            ]
            say(
                '[TRANSFER | {} | {}] correct={}/24 '
                '| MAE={:.3f}h'.format(
                    mode,transfer,
                    int(part.correct.sum()),
                    part.absolute_error_hours.mean()
                )
            )

            subset=pair[
                (pair.representation==mode)
                &(pair.transfer==transfer)
            ]
            for _,r in subset.groupby('pair'):
                say(
                    '[PAIR | {} | {} | {}] correct={}/12'
                    .format(
                        mode,transfer,
                        r.pair.iloc[0],
                        int(r.correct.sum())
                    )
                )

    return result,pair

def vehicle_gate(counts,X,meta,ix,historical):
    ex=meta.experiment.to_numpy(int)
    treat=meta.treatment.to_numpy(str)
    tm=meta.hours.to_numpy(int)

    archived=historical[
        historical.method=='drug_aware_centroid'
    ].set_index('gsm')

    rows=[]
    marker=[]

    for mode in MODES:
        V=train_transform(X,ix,mode)

        for hold in (1,2,3):
            train=np.flatnonzero(
                (ex!=hold)&(treat!='Baseline')
            )
            active=train[
                np.isin(treat[train],DRUGS)
            ]
            vehicle=train[
                treat[train]=='DMSO'
            ]
            test=np.flatnonzero(
                (ex==hold)&(treat!='Baseline')
            )

            need(
                len(train)==24
                and len(active)==16
                and len(vehicle)==8
                and len(test)==12,
                'Gate split mismatch'
            )

            deltas=[]

            for exp in (1,2,3):
                if exp==hold:
                    continue
                for drug in DRUGS:
                    for h in TIMES:
                        deltas.append(
                            V[:,ix[(exp,drug,h)]]
                            -V[:,ix[(exp,'DMSO',h)]]
                        )

            D=np.column_stack(deltas)
            need(
                D.shape==(24421,16),
                'Training matched contrasts missing'
            )

            sd=np.maximum(
                np.std(V[:,train],axis=1),.5
            )
            score=np.abs(D.mean(axis=1))/sd

            eligible=(
                (counts[:,train]>=10).sum(axis=1)>=3
            ) & np.isfinite(score)

            candidates=np.flatnonzero(eligible)
            need(
                len(candidates)>=TOP,
                'Not enough eligible active/vehicle genes'
            )

            genes=candidates[
                np.argsort(
                    -score[candidates],
                    kind='stable'
                )[:TOP]
            ]

            Z=z_transform(V,genes,train)
            ca=Z[active].mean(axis=0)
            cv=Z[vehicle].mean(axis=0)

            for j,g in enumerate(genes):
                marker.append(dict(
                    representation=mode,
                    heldout_experiment=hold,
                    rank=j+1,
                    gene_index=int(g),
                    training_pair_effect=score[g]
                ))

            for i in test:
                d_veh=float(np.sum(
                    (Z[i]-cv)**2
                ))
                d_act=float(np.sum(
                    (Z[i]-ca)**2
                ))

                call_active=d_act<d_veh
                actual_active=treat[i] in DRUGS

                prior_time=(
                    int(
                        archived.loc[
                            meta.iloc[i].gsm
                        ].predicted_hours
                    )
                    if actual_active
                    else None
                )

                rows.append(dict(
                    representation=mode,
                    heldout_experiment=hold,
                    gsm=meta.iloc[i].gsm,
                    treatment=treat[i],
                    hours=int(tm[i]),
                    true_active=bool(actual_active),
                    called_active=bool(call_active),
                    correct_activity=bool(
                        actual_active==call_active
                    ),
                    active_margin_vehicle_minus_active=(
                        d_veh-d_act
                    ),
                    predicted_duration_if_active=prior_time,
                    accepted_active_and_correct_duration=bool(
                        actual_active
                        and call_active
                        and prior_time==tm[i]
                    ),
                    rejected_active=bool(
                        actual_active and not call_active
                    ),
                    false_active_DMSO=bool(
                        (not actual_active) and call_active
                    )
                ))

        say(
            f'[VEHICLE GATE] {mode} | '
            'training-only contrasts, '
            'pre-fixed score margin 0'
        )

    table=save(
        '05_vehicle_gate_predictions.csv',
        rows
    )
    save(
        '06_vehicle_gate_train_features.csv',
        marker
    )
    need(
        len(table)==72,
        'Gate results count mismatch'
    )

    for mode in MODES:
        part=table[
            table.representation==mode
        ]
        act=part[part.true_active]
        neg=part[~part.true_active]
        selected=act[act.called_active]

        say(
            '[DMSO GATE | {}] DMSO rejected={}/12 '
            '| active accepted={}/24 '
            '| accepted time correct={}/{} '
            '| total correct-and-accepted={}/24'
            .format(
                mode,
                int((~neg.called_active).sum()),
                int(act.called_active.sum()),
                int(
                    selected.accepted_active_and_correct_duration.sum()
                ),
                len(selected),
                int(
                    act.accepted_active_and_correct_duration.sum()
                )
            )
        )
    return table

def variability(counts,X,meta,ix):
    batches=np.asarray((1,2,3),int)
    designs=[
        ('Baseline',0)
    ]+[
        (drug,t)
        for drug in ('DMSO',)+DRUGS
        for t in TIMES
    ]

    variances=[]

    for drug,t in designs:
        vals=np.column_stack([
            X[:,ix[(e,drug,t)]]
            for e in batches
        ])
        variances.append(
            np.var(vals,axis=1,ddof=1)
        )

    var=np.column_stack(variances)
    all_sd=np.sqrt(
        np.mean(var,axis=1)
    )

    diffvar=[]
    bygroup=[]

    for drug in DRUGS:
        for h in TIMES:
            contrasts=np.column_stack([
                X[:,ix[(e,drug,h)]]
                -X[:,ix[(e,'DMSO',h)]]
                for e in batches
            ])

            s=np.std(
                contrasts,axis=1,ddof=1
            )
            diffvar.append(s*s)

            bygroup.append(dict(
                drug=drug,
                hours=h,
                median_abs_effect_log2CPM=float(
                    np.median(np.abs(
                        np.mean(contrasts,axis=1)
                    ))
                ),
                median_between_experiment_SD_log2CPM=float(
                    np.median(s)
                ),
                n_experiments=3
            ))

    diff_sd=np.sqrt(
        np.mean(
            np.column_stack(diffvar),
            axis=1
        )
    )
    mask=(counts>=10).sum(axis=1)>=6

    gene_rows=pd.DataFrame(dict(
        gene_index=np.arange(
            counts.shape[0],dtype=int
        ),
        all_conditions_between_experiment_rms_SD=all_sd,
        drug_vs_DMSO_between_experiment_rms_SD=diff_sd,
        counts_ge10_in_at_least6=mask
    ))

    save(
        '07_between_experiment_gene_variability.csv',
        gene_rows
    )
    save(
        '08_matched_drug_vehicle_variability.csv',
        bygroup
    )

    need(
        mask.sum()>1000,
        'Too few expressed genes for SD audit'
    )

    say(
        '[REPLICATE VARIABILITY] expressed genes={} '
        '| median matched-condition SD={:.4f} '
        '| median drug-minus-DMSO SD={:.4f} log2CPM'
        .format(
            int(mask.sum()),
            float(np.median(all_sd[mask])),
            float(np.median(diff_sd[mask]))
        )
    )

    say(
        '[NOISE LIMIT] Three independent experiments '
        'estimate combined reproducibility, '
        'NOT technical noise or validated sigma covariance'
    )

def main():
    say('='*94)
    say(
        'PAPER 7 | V114 | REAL GSE194200 DRUG TRANSFER, '
        'DMSO GATE & BATCH VARIABILITY'
    )
    say(
        'OFFLINE | FROZEN V113 | NO SYNTHETIC SHIFTS '
        '| NO METHOD NOVELTY CLAIM | NO GIT'
    )
    say('='*94)

    need(
        len(sys.argv)==1,
        'Run Python filename without flags'
    )

    run,counts,X,meta,annotation,historical=load()
    ix=indices(meta)

    protocol=dict(
        version='v114',
        GEO='GSE194200',
        source_v113_manifest_sha256=sha(
            run/'09_protocol_manifest.json'
        ),
        top_genes=TOP,
        modes=MODES,
        stress_test=(
            'train on one drug and two experiments, '
            'test on other drug in heldout experiment'
        ),
        gate=(
            'drug vs DMSO nearest centroids, '
            'train-only paired contrasts, zero threshold'
        ),
        baseline_centered=(
            'uses one existing baseline sample '
            'from each train/test experiment'
        ),
        pairwise_resolution=PAIRS,
        known_target=(
            'time since compound addition, '
            'NOT biological onset'
        ),
        do_not_use_time_matched_DMSO_from_test_for_normalization=True,
        preselected_settings_no_holdout_tuning=True,
        empirical_95pct_coverage_claim=False,
        script_sha256=sha(__file__),
        no_network=True,
        no_git=True
    )

    (OUT/'01_protocol.json').write_text(
        json.dumps(protocol,indent=2)+'\n',
        encoding='utf-8'
    )

    cross_drug(counts,X,meta,ix)
    vehicle_gate(
        counts,X,meta,ix,historical
    )
    variability(
        counts,X,meta,ix
    )

    (OUT/'09_methods_and_limits.md').write_text(
        """# V114 — real-data diagnostic stress tests

Target: known duration of drug exposure,
NOT gene-specific onset.

Three independent MIN6 experiments are grouped
in heldout tests.

Cross-drug test trains on one compound in two
experiments and tests on the other compound in
the third experiment.

Same-drug test has identical experiment splits.
No hyperparameter tuning from test labels.

Raw log2CPM and per-experiment baseline-centered
log2CPM are separate analysis regimes.

The baseline-centered regime requires a measured
0h baseline for every heldout experiment.

Baseline is always allowed as a reference input
and has no duration label of an active treatment.

It cannot be assumed available for a single
unpaired clinical sample.

Vehicle gate detects whether heldout expression
resembles an active compound or DMSO, using
training-only drug-minus-time-matched-DMSO features.

Heldout duration is never used to find a
time-matched DMSO.

The predeclared threshold is a zero difference
in centroid squared distances.

The gate is not a calibrated false-positive bound.
A vehicle duration guess is not a valid assessment
of false activation.

The baseline time prediction is reused unchanged
from frozen V113; this does not constitute a newly
validated algorithm.

With 3 independent experiments, between-experiment
sample variance can be described per condition,
but its uncertainty is high, and it combines
biological and technical factors.

No full gene covariance or nonparametric 95%
coverage is claimed.

Multiple genes are not independent biological
replication units.

V114 is a diagnostic, NOT a novel method evaluation
or proof of biological onset detection.

Comparisons with existing temporal methods and
an additional independent study are needed.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v114.py'
    )

    for f in (
        '09_protocol_manifest.json',
        '01_verified_sample_mapping.csv',
        '05_heldout_sample_predictions.csv',
        '02_gene_annotations.csv'
    ):
        shutil.copy2(
            run/f,
            OUT/('input_v113_'+f)
        )

    shutil.copy2(
        run/'04_gene_counts_and_logCPM.npz',
        OUT/'input_v113_counts_logCPM.npz'
    )

    (OUT/'10_manifest.json').write_text(
        json.dumps(dict(
            version='v114',
            script_sha256=sha(__file__),
            source_v113_sha256=sha(
                run/'09_protocol_manifest.json'
            ),
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            real_onset_truth=False,
            raw_counts_already_local=True,
            noise_covariance_known=False,
            no_git=True
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[GATE] DMSO specificity, cross-drug transfer, '
        'and 1v2/2v6/6v24 diagnostics only'
    )
    say(
        '[COMPLETE] Frozen V114 external '
        'stress tests archived'
    )

def finish(ok):
    say(
        '[RESULT] '+(
            'SUCCESS' if ok else 'FAILED'
        )
    )
    say(
        '[REPORT] '+
        str(OUT/'console_summary.txt')
    )

    archive=OUT/'portable_report_v114.zip'
    say('[ZIP] '+str(archive))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    entries=[]

    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.name not in (
            archive.name,
            '11_sha256_inventory.csv'
        ):
            entries.append(dict(
                filename=f.name,
                bytes=f.stat().st_size,
                sha256=sha(f)
            ))

    pd.DataFrame(entries).to_csv(
        OUT/'11_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive,'w',zipfile.ZIP_DEFLATED,
        compresslevel=5
    ) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file() and f!=archive:
                z.write(f,f.name)

    say(
        '[ZIP COMPLETE] Outputs, source archive '
        'references, logs, code, hashes'
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
