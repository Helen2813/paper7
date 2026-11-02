#!/usr/bin/env python3
"""V111: archived V110 tie audit and direct observed-knot contrast."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import sys, json, zipfile, traceback, shutil, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper7_v111_design_tie_audit' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)
LOG = []

CULTURES = ('S169','S170','S506','S508')
GENES = ('PRL','IGFBP1','FOXO1','HAND2',
         'WNT4','PGR','IL11','LEFTY2')
EXTRAS = ('plus_3','plus_6','plus_9','plus_18')

SOURCE_SHA = (
    'eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a'
)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def digest(p):
    h = sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def save(name, data):
    df = data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(
        OUT/name,
        index=False,
        encoding='utf-8-sig',
        float_format='%.15g'
    )
    say('[OUTPUT] {} | {} rows'.format(name,len(df)))
    return df

def load():
    base = ROOT/'paper7_three_delay_design_v110'
    files = (
        '13_sha256_inventory.csv',
        '12_manifest.json',
        '02_three_pair_distances.csv',
        '03_gene_direction_SSE.csv',
        '04_three_pair_bottlenecks.csv',
        '07_global_minimax_design.csv',
        '08_leave_one_gene_out_all_options.csv',
        '09_leave_one_gene_out_winners.csv',
        '10_leave_one_culture_out_descriptive.csv',
        'input_measured_source.csv'
    )

    runs = sorted(
        (
            p for p in base.glob('run_*')
            if p.is_dir()
            and all((p/name).is_file() for name in files)
        ),
        reverse=True
    )
    require(runs,'V110 archived run unavailable')
    run = runs[0]

    inventory = pd.read_csv(
        run/'13_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    require(
        {'filename','bytes','sha256'}.issubset(inventory.columns)
        and inventory.filename.is_unique,
        'V110 SHA inventory invalid'
    )

    for item in inventory.itertuples(index=False):
        name = str(item.filename)
        p = run/name
        require(
            name == Path(name).name
            and p.is_file()
            and p.stat().st_size == int(item.bytes)
            and digest(p) == str(item.sha256),
            'Modified V110 archive: '+name
        )

    man = json.loads(
        (run/'12_manifest.json').read_text('utf-8')
    )
    require(
        man.get('version') == 'v110'
        and man.get('pairwise_rows') == 240
        and man.get('gene_direction_rows') == 1920,
        'V110 provenance mismatch'
    )

    source = run/'input_measured_source.csv'
    require(
        digest(source) == SOURCE_SHA,
        'Measured source SHA256 differs'
    )

    filenames = {
        'distance':'02_three_pair_distances.csv',
        'genes':'03_gene_direction_SSE.csv',
        'bottlenecks':'04_three_pair_bottlenecks.csv',
        'global':'07_global_minimax_design.csv',
        'loo':'08_leave_one_gene_out_all_options.csv',
        'loo_winners':'09_leave_one_gene_out_winners.csv',
        'loco':'10_leave_one_culture_out_descriptive.csv',
        'source':'input_measured_source.csv'
    }

    data = {
        key:pd.read_csv(run/name,encoding='utf-8-sig')
        for key,name in filenames.items()
    }

    for key,n in (
        ('distance',240),('genes',1920),
        ('bottlenecks',40),('global',10),
        ('loo',160),('loo_winners',32),
        ('loco',4),('source',320)
    ):
        require(
            len(data[key]) == n,
            'Unexpected V110 table size: '+key
        )

    say(
        '[V110 VERIFIED] {} | {} checksum-verified files'
        .format(run,len(inventory))
    )
    return run,data

def audit_identity(data):
    checks = []

    for level,keys,columns,expected in (
        (
            'culture_pairs',
            ('culture_target','culture_alternative'),
            ('forward_SSE','reverse_SSE','min_SSE'),
            16
        ),
        (
            'gene_components',
            ('culture_target','culture_alternative','gene'),
            ('forward_SSE','reverse_SSE'),
            128
        )
    ):
        frame = (
            data['distance']
            if level == 'culture_pairs'
            else data['genes']
        )

        part = frame[
            (frame.tau_a == 6)
            & (frame.tau_b == 12)
        ]

        left = part[
            part.schedule == 'plus_3'
        ][list(keys)+list(columns)].copy()

        right = part[
            part.schedule == 'plus_6'
        ][list(keys)+list(columns)].copy()

        require(
            len(left) == len(right) == expected,
            'Incomplete 6v12 comparison'
        )

        paired = left.merge(
            right,
            on=list(keys),
            validate='one_to_one',
            suffixes=('_3','_6')
        )
        require(len(paired)==expected,'Missing culture/gene pair')

        maximum = max(
            float(np.max(np.abs(
                paired[col+'_3'].to_numpy()
                -paired[col+'_6'].to_numpy()
            )))
            for col in columns
        )

        require(
            maximum <= 1e-9,
            'Unexpected 6v12 identity failure'
        )

        checks.append(dict(
            level=level,
            n=expected,
            max_abs_SSE_difference=maximum,
            identical_within_tolerance=True
        ))

        say(
            '[V111 IDENTITY | {}] +3 vs +6 for 6v12: '
            '{}/{} match | max difference={:.3g}'
            .format(level,expected,expected,maximum)
        )

    save('02_exact_6v12_identity.csv',checks)

def audit_direct_difference(source):
    require(
        not source.duplicated([
            'culture_label','gene','time_hours'
        ]).any(),
        'Duplicate source measurements'
    )

    records = []

    for culture in CULTURES:
        for gene in GENES:
            df = source[
                (source.culture_label == culture)
                & (source.gene == gene)
            ]
            require(len(df)==10,'Incomplete measured trajectory')

            values = {
                int(t):float(y)
                for t,y in zip(
                    df.time_hours,
                    df.log2CPM
                )
            }
            require(
                all(t in values for t in (0,3,6,12)),
                'Missing measured early knot'
            )

            for hour in (3,6,9,18):
                t6 = max(hour-6,0)
                t12 = max(hour-12,0)

                delta = values[t6]-values[t12]

                records.append(dict(
                    culture=culture,
                    gene=gene,
                    additional_hour=hour,
                    source_hour_tau6=t6,
                    source_hour_tau12=t12,
                    direct_delta_log2CPM=delta,
                    abs_delta=abs(delta),
                    identical_prediction=abs(delta)<=1e-12
                ))

    result = save(
        '03_direct_6v12_gene_differences.csv',
        records
    )

    for hour,group in result.groupby('additional_hour'):
        say(
            '[V111 DIRECT 6v12 | +{}h] identical={}/32 '
            '| mean absolute difference={:.5f} log2CPM'
            .format(
                hour,
                int(group.identical_prediction.sum()),
                group.abs_delta.mean()
            )
        )

def audit_minimax(data):
    b = data['bottlenecks']

    require(
        not b.duplicated([
            'culture','objective','schedule'
        ]).any(),
        'Duplicate bottleneck result'
    )

    results = []

    for culture in CULTURES:
        for objective in (
            'known_culture','unknown_culture'
        ):
            sub = b[
                (b.culture == culture)
                & (b.objective == objective)
            ]
            require(len(sub)==5,'Incomplete schedule scores')

            values = sub[
                sub.schedule.isin(EXTRAS)
            ].set_index('schedule').min_three_pair_SSE.astype(float)

            maximum = float(values.max())
            eps = 1e-9*max(1.,abs(maximum))
            tied = sorted(
                values.index[
                    (maximum-values)<=eps
                ]
            )

            result = dict(
                culture=culture,
                objective=objective,
                best_SSE=maximum,
                tie_tolerance=eps,
                equivalent_best=';'.join(tied),
                n_equivalent=len(tied),
                plus3_SSE=float(values['plus_3']),
                plus6_SSE=float(values['plus_6']),
                plus9_SSE=float(values['plus_9']),
                plus18_SSE=float(values['plus_18']),
                difference_3_minus_6=float(
                    values['plus_3']-values['plus_6']
                )
            )
            results.append(result)

            say(
                '[V111 MINIMAX | {} | {}] best={} '
                '| difference 3 minus 6={:+.7f}'
                .format(
                    culture,objective,
                    result['equivalent_best'],
                    result['difference_3_minus_6']
                )
            )

    save('04_minimax_ties_and_margins.csv',results)

    global_results=[]

    for objective in ('known_culture','unknown_culture'):
        sub=b[b.objective==objective]
        archived=data['global'][
            data['global'].objective==objective
        ]

        require(len(archived)==5,'Global score incomplete')

        column = (
            'global_worst_SSE'
            if 'global_worst_SSE' in archived.columns
            else 'global_worst_case_SSE'
        )

        for schedule in ('base_0_12_24',)+EXTRAS:
            score=float(
                sub[
                    sub.schedule==schedule
                ].min_three_pair_SSE.min()
            )
            prior=float(
                archived[
                    archived.schedule==schedule
                ][column].iloc[0]
            )
            require(
                abs(score-prior)<1e-8,
                'Global ranking replay mismatch'
            )

            global_results.append(dict(
                objective=objective,
                schedule=schedule,
                global_min_SSE=score
            ))

        choices=[
            r for r in global_results
            if r['objective']==objective
            and r['schedule'] in EXTRAS
        ]
        best=max(r['global_min_SSE'] for r in choices)
        eps=1e-9*max(1.,abs(best))

        tied=sorted(
            r['schedule']
            for r in choices
            if best-r['global_min_SSE']<=eps
        )

        say(
            '[V111 GLOBAL | {}] best={} | SSE={:.6f}'
            .format(objective,';'.join(tied),best)
        )

    save('05_global_minimax_ties.csv',global_results)

def audit_gene_deletion(data):
    loo=data['loo']
    previous=data['loo_winners']
    rows=[]

    for culture in CULTURES:
        for gene in GENES:
            part=loo[
                (loo.culture==culture)
                & (loo.deleted_gene==gene)
            ]
            require(len(part)==5,'Gene deletion incomplete')

            scores=part.set_index(
                'schedule'
            ).unknown_culture_min_SSE.loc[
                list(EXTRAS)
            ].astype(float)

            best=float(scores.max())
            eps=1e-9*max(1.,abs(best))
            tied=sorted(
                scores.index[
                    (best-scores)<=eps
                ]
            )

            old=previous[
                (previous.culture==culture)
                & (previous.deleted_gene==gene)
            ]
            require(len(old)==1,'Archived gene winner missing')

            selected=str(old.iloc[0].selected)
            require(
                selected in tied,
                'Archived winner not optimal'
            )

            rows.append(dict(
                culture=culture,
                removed_gene=gene,
                previous_winner=selected,
                tie_aware_winners=';'.join(tied),
                n_equivalent=len(tied)
            ))

    result=save('06_gene_deletion_tie_audit.csv',rows)

    say(
        '[V111 GENE LOO] tied winners in {}/32 removals'
        .format(int((result.n_equivalent>1).sum()))
    )

def audit_culture_holdout(data):
    b=data['bottlenecks']

    matrix=b[
        b.objective=='unknown_culture'
    ].pivot(
        index='culture',
        columns='schedule',
        values='min_three_pair_SSE'
    )

    old=data['loco']
    rows=[]

    for culture in CULTURES:
        options=matrix[list(EXTRAS)]
        training=options.drop(index=culture).mean(axis=0)
        selected=training.idxmax()

        held=options.loc[culture]
        best=float(held.max())
        eps=1e-9*max(1.,abs(best))

        tied=sorted(
            held.index[
                (best-held)<=eps
            ]
        )

        previous=old[
            old.held_out_culture==culture
        ]
        require(
            len(previous)==1
            and str(previous.iloc[0].selected_from_three)==selected,
            'V110 LOCO choice replay failed'
        )

        regret_column=(
            'regret_SSE'
            if 'regret_SSE' in previous.columns
            else 'heldout_regret_SSE'
        )
        regret=float(best-held[selected])

        require(
            abs(
                regret-float(
                    previous.iloc[0][regret_column]
                )
            )<1e-8,
            'V110 LOCO regret replay failed'
        )

        rows.append(dict(
            heldout_culture=culture,
            selected_from_three=selected,
            heldout_best_with_ties=';'.join(tied),
            strict_agreement=bool(
                previous.iloc[0].agreement
            ),
            tie_aware_agreement=selected in tied,
            regret_SSE=regret
        ))

        say(
            '[V111 LOCO | {}] selected={} | best={} '
            '| regret={:.7f}'
            .format(
                culture,selected,';'.join(tied),regret
            )
        )

    result=save('07_loco_tie_aware.csv',rows)

    say(
        '[V111 LOCO] strict={}/4; tie-aware={}/4'
        .format(
            int(result.strict_agreement.sum()),
            int(result.tie_aware_agreement.sum())
        )
    )

def main():
    say('='*84)
    say('PAPER 7 | V111 | MINIMAX TIE / PRE-ONSET IDENTITY AUDIT')
    say(
        'OFFLINE | FROZEN V110 | NO TRAINING '
        '| NO DELAY TRUTH | NO GIT'
    )
    say('='*84)

    require(len(sys.argv)==1,'Run without flags')

    run,data=load()

    (OUT/'01_protocol.json').write_text(
        json.dumps(dict(
            version='v111',
            v110_manifest_sha256=digest(
                run/'12_manifest.json'
            ),
            source_sha256=SOURCE_SHA,
            tie_tolerance='1e-9*max(1,abs(best_SSE))',
            no_new_data=True,
            no_training=True,
            no_delay_truth=True,
            script_sha256=digest(__file__)
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    audit_identity(data)
    audit_direct_difference(data['source'])
    audit_minimax(data)
    audit_gene_deletion(data)
    audit_culture_holdout(data)

    (OUT/'08_methods_limitations.md').write_text(
        """# V111: evidence and interpretation

V110 contains model-generated response shifts of
0, 6 and 12 hours.

For tau=6 and tau=12, measurements at 3 and 6
hours both use the same source baseline value.

Consequently, adding +3 or +6 creates identical
modelled 6-vs-12 pairwise SSE after nuisance fitting.

The complete minimax ranking can nevertheless
differ because it also includes 0-vs-6 and
0-vs-12 comparisons.

At 9 and 18 hours, the shifted responses may
differ directly, depending on the measured curve.

A repeated baseline observation can affect
nuisance fitting but does not itself distinguish
the tau=6 and tau=12 response onsets.

Tie-aware rankings use a declared numerical
tolerance and do not constitute an independent
experimental validation.

No known delayed biological samples, independent
technical noise estimates or gene covariance
were introduced.

The V104 shape-free structural ambiguity remains.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v111.py'
    )

    for src,dst in (
        ('12_manifest.json','input_v110_manifest.json'),
        ('02_three_pair_distances.csv','input_v110_pairwise.csv'),
        ('03_gene_direction_SSE.csv','input_v110_genes.csv'),
        ('04_three_pair_bottlenecks.csv','input_v110_bottlenecks.csv'),
        ('07_global_minimax_design.csv','input_v110_global.csv'),
        ('08_leave_one_gene_out_all_options.csv','input_v110_loo.csv'),
        ('09_leave_one_gene_out_winners.csv','input_v110_loo_winners.csv'),
        ('10_leave_one_culture_out_descriptive.csv','input_v110_loco.csv'),
        ('input_measured_source.csv','input_measured_source.csv')
    ):
        shutil.copy2(run/src,OUT/dst)

    (OUT/'09_manifest.json').write_text(
        json.dumps(dict(
            version='v111',
            code_sha256=digest(__file__),
            v110_manifest_sha256=digest(
                run/'12_manifest.json'
            ),
            source_sha256=SOURCE_SHA,
            no_new_data=True,
            noise_sigma_available=False,
            biological_delay_truth=False,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            no_git=True
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[GATE] Conditional template geometry only; '
        'no demonstrated biological delay detection'
    )
    say('[COMPLETE] Identity and numerical tie audit archived')

def finish(ok):
    say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
    say('[REPORT] '+str(OUT/'console_summary.txt'))

    archive=OUT/'portable_report_v111.zip'
    say('[ZIP] '+str(archive))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    items=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '10_sha256_inventory.csv',archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT/'10_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive,'w',zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

if __name__=='__main__':
    ok=True
    try:
        main()
    except Exception:
        ok=False
        say('[ERROR]\n'+traceback.format_exc())
    finally:
        finish(ok)

    if not ok:
        sys.exit(1)
