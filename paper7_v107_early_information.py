from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
from itertools import combinations
import json,sys,traceback,zipfile,shutil,platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper7_early_gene_design_v107'/(
    'run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True,exist_ok=False)
LOG=[]
GENES=('PRL','IGFBP1','FOXO1','HAND2','WNT4','PGR','IL11','LEFTY2')
CULTURES=('S169','S170','S506','S508')
BASE=(0,12,24)
EXTRA=(3,6,9,18)
SOURCE_SHA='eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a'

def say(v):
    v=str(v)
    print(v,flush=True)
    LOG.append(v)

def need(ok,msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):
            h.update(x)
    return h.hexdigest()

def save(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(
        OUT/name,index=False,
        encoding='utf-8-sig',float_format='%.15g'
    )
    say('[OUTPUT] {} | rows={}'.format(name,len(d)))
    return d

def read_source():
    base=ROOT/'paper7_direct_early_grid_v106'
    required=(
        '10_sha256_inventory.csv',
        'source_v67_measured.csv',
        '02_measured_0_3_6_gene_contrasts.csv',
        '04_direct_0v6_profile_distances.csv',
        '09_manifest.json'
    )
    runs=sorted(
        (r for r in base.glob('run_*')
         if r.is_dir()
         and all((r/n).is_file() for n in required)),
        reverse=True
    )
    need(runs,'V106 complete archive not found')
    run=runs[0]

    inv=pd.read_csv(
        run/'10_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    need(
        {'filename','bytes','sha256'}.issubset(inv.columns)
        and inv.filename.is_unique,
        'V106 inventory invalid'
    )

    for r in inv.itertuples(index=False):
        name=str(r.filename)
        p=run/name
        need(
            Path(name).name==name
            and p.is_file()
            and p.stat().st_size==int(r.bytes)
            and sha(p)==str(r.sha256),
            'V106 archive changed: '+name
        )

    src=run/'source_v67_measured.csv'
    need(sha(src)==SOURCE_SHA,'Source bytes changed')

    data=pd.read_csv(src,encoding='utf-8-sig')
    need(
        len(data)==320
        and {'culture_label','gene','time_hours','log2CPM'}.issubset(data.columns),
        'V67 source schema mismatch'
    )
    need(
        not data.duplicated(
            ['culture_label','gene','time_hours']
        ).any(),
        'Source duplicate'
    )

    curves={}
    for cult in CULTURES:
        for gene in GENES:
            g=data[
                (data.culture_label==cult)
                &(data.gene==gene)
            ].sort_values('time_hours')
            need(
                len(g)==10
                and np.allclose(
                    g.time_hours.to_numpy(float),
                    [0,3,6,9,12,18,24,36,48,96]
                ),
                'Unexpected measured schedule'
            )
            curves[(cult,gene)]={
                int(t):float(y)
                for t,y in zip(g.time_hours,g.log2CPM)
            }

    say(
        '[V106 VERIFIED] {} | archived files={} | measured curves=32'
        .format(run,len(inv))
    )
    return run,curves

def vector(curve,sched,tau):
    return np.asarray(
        [curve[0 if t<=tau else t-tau] for t in sched],
        float
    )

def profiled(a,b):
    ac=a-a.mean()
    bc=b-b.mean()
    ss=float(bc@bc)
    amp=max(0.,float(ac@bc)/ss) if ss>1e-14 else 0.
    residual=a-(a.mean()-amp*b.mean()+amp*b)
    return float(residual@residual)

def early(curves):
    rows=[]
    for cult in CULTURES:
        for gene in GENES:
            y=curves[(cult,gene)]
            d03=y[3]-y[0]
            d36=y[6]-y[3]
            curvature=y[0]-2*y[3]+y[6]
            t=np.array([y[0],y[3],y[6]])
            constant=float(((t-t.mean())**2).sum())
            drift=curvature**2/6.
            dyn=(
                max(y[t] for t in (0,3,6,9,12,18,24))
                -min(y[t] for t in (0,3,6,9,12,18,24))
            )

            rows.append(dict(
                culture=cult,gene=gene,
                delta0to3=d03,
                delta3to6=d36,
                delta0to6=y[6]-y[0],
                second_difference=curvature,
                flat_SSE=constant,
                linear_drift_SSE=drift,
                observed_range_0to24=dyn
            ))

    d=save('02_measured_gene_early_contrasts.csv',rows)
    signs=[]

    for gene,g in d.groupby('gene'):
        v=g.delta0to3.to_numpy(float)
        k=g.second_difference.to_numpy(float)

        signs.append(dict(
            gene=gene,
            mean_delta0to3=float(v.mean()),
            min_delta0to3=float(v.min()),
            max_delta0to3=float(v.max()),
            same_sign_0to3=bool(
                np.all(v>0) or np.all(v<0)
            ),
            same_sign_curvature=bool(
                np.all(k>0) or np.all(k<0)
            ),
            mean_abs_curvature=float(np.abs(k).mean()),
            mean_linear_drift_SSE=float(
                g.linear_drift_SSE.mean()
            )
        ))

    sign=save('03_gene_consistency_four_cultures.csv',signs)

    say(
        '[EARLY CONSISTENCY] 0->3h same direction in all four cultures: {}/8 genes'
        .format(int(sign.same_sign_0to3.sum()))
    )
    say(
        '[EARLY CURVATURE] same sign in all four cultures: {}/8 genes'
        .format(int(sign.same_sign_curvature.sum()))
    )
    return d

def design(curves,prior):
    plans=[('base_0_12_24',BASE)]

    for k in (1,2):
        for added in combinations(EXTRA,k):
            plans.append((
                'plus_'+('_'.join(str(x) for x in added)),
                tuple(sorted(BASE+added))
            ))

    rows=[]
    contrib=[]

    for name,grid in plans:
        for ca in CULTURES:
            for cb in CULTURES:
                sf=0.
                sr=0.
                pieces=[]

                for gene in GENES:
                    a=curves[(ca,gene)]
                    b=curves[(cb,gene)]

                    f=profiled(
                        vector(a,grid,0),
                        vector(b,grid,6)
                    )
                    r=profiled(
                        vector(a,grid,6),
                        vector(b,grid,0)
                    )

                    sf+=f
                    sr+=r
                    pieces.append((gene,f,r))

                m=min(sf,sr)

                rows.append(dict(
                    schedule=name,
                    hours=';'.join(map(str,grid)),
                    extra_hours=len(grid)-len(BASE),
                    culture_target=ca,
                    culture_alternative=cb,
                    same_culture=ca==cb,
                    n_times=len(grid),
                    min_SSE=m,
                    normalised_RMS=float(
                        np.sqrt(m/(8*len(grid)))
                    ),
                    sqrtSSE=float(np.sqrt(m))
                ))

                if ca==cb:
                    for gene,f,r in pieces:
                        reduced=min(sf-f,sr-r)
                        drop=max(0.,m-reduced)

                        contrib.append(dict(
                            schedule=name,
                            culture=ca,
                            gene=gene,
                            full_min_SSE=m,
                            SSE_without_gene=reduced,
                            drop_SSE=drop,
                            relative_drop_when_removed=(
                                float(drop/m)
                                if m>1e-12 else np.nan
                            )
                        ))

    table=save('04_equal_cost_conditional_designs.csv',rows)
    save('05_gene_leave_one_out.csv',contrib)

    # Reproduce the original three-point V106 values.
    archive=prior[
        (prior.contrast=='same_culture')
        &(prior.schedule=='sparse_0_12_24')
    ].set_index('culture_target')

    for ca in CULTURES:
        now=table[
            (table.schedule=='base_0_12_24')
            &(table.culture_target==ca)
            &(table.culture_alternative==ca)
        ].iloc[0]

        need(
            abs(
                float(
                    archive.loc[ca].conservative_direction_sqrtSSE
                )-now.sqrtSSE
            )<1e-8,
            'V106 sparse-grid replay differs for '+ca
        )

    comparison=[]

    for cult in CULTURES:
        sub=table[table.culture_target==cult]
        baseline=sub[
            (sub.schedule=='base_0_12_24')
            &(sub.culture_alternative==cult)
        ].iloc[0].min_SSE

        for row in sub.itertuples(index=False):
            if row.same_culture:
                others=sub[
                    (sub.schedule==row.schedule)
                    &(~sub.same_culture)
                ].sort_values('min_SSE')

                alt=others.iloc[0]

                comparison.append(dict(
                    culture=cult,
                    schedule=row.schedule,
                    extra_hours=row.extra_hours,
                    within_SSE=row.min_SSE,
                    incremental_SSE=row.min_SSE-baseline,
                    within_RMS=row.normalised_RMS,
                    nearest_other_culture=str(
                        alt.culture_alternative
                    ),
                    nearest_other_SSE=float(
                        alt.min_SSE
                    ),
                    cross_culture_looks_closer=bool(
                        alt.min_SSE<row.min_SSE-1e-10
                    )
                ))

    comp=save('06_additional_timepoint_value.csv',comparison)

    for cult in CULTURES:
        g=comp[
            (comp.culture==cult)
            &(comp.extra_hours==1)
        ]
        best=g.sort_values(
            'incremental_SSE',
            ascending=False
        ).iloc[0]

        s3=g[g.schedule=='plus_3'].iloc[0]
        s6=g[g.schedule=='plus_6'].iloc[0]

        say(
            '[V107 EARLY VALUE | {}] add 3h SSE={:.4f}; '
            'add 6h SSE={:.4f}; best one extra={} (conditional)'
            .format(
                cult,
                s3.incremental_SSE,
                s6.incremental_SSE,
                best.schedule
            )
        )

    count=int(comp.cross_culture_looks_closer.sum())
    say(
        '[CROSS CULTURE] alternative culture closer than same culture: {}/{} comparisons'
        .format(count,len(comp))
    )
    return comp

def main():
    say('='*92)
    say('PAPER 7 | V107 | REAL 3h/6h GENE CONTRIBUTIONS & EQUAL-COST DESIGNS')
    say('NO NEW DELAY TRUTH | NO INTERPOLATION | NO NOISE ASSUMPTIONS | NO GIT')
    say('='*92)

    need(
        len(sys.argv)==1,
        'Use Python and script filename only'
    )

    run,curves=read_source()

    protocol=dict(
        version='v107',
        source_sha256=SOURCE_SHA,
        base_schedule=BASE,
        additional_candidate_hours=EXTRA,
        max_added_times=2,
        same_culture_vs_other_culture_descriptive=True,
        assumed_tau6='source reindexed and flat before onset',
        noise_sd='not estimated',
        no_git=True,
        code_sha256=sha(__file__)
    )

    (OUT/'01_protocol.json').write_text(
        json.dumps(protocol,indent=2)+'\n',
        encoding='utf-8'
    )

    early(curves)

    prior=pd.read_csv(
        run/'04_direct_0v6_profile_distances.csv',
        encoding='utf-8-sig'
    )
    design(curves,prior)

    (OUT/'07_limits.md').write_text(
        """# V107 limitations

Measured RNA-seq values from GSE246591:
4 cultures x 8 genes x 10 timepoints.

The 6h-shifted alternative is reindexed, not
independently experimentally observed.

Flat pre-onset and fixed source shape are
assumptions. Gene amplitude and intercept are profiled.

Adding a timepoint cannot decrease minimized
total SSE for nested schedules.

Compare candidates with the same number of
extra timepoints; no measured biological power
is claimed.

Four cultures are not technical replicates.
Cross-culture distance is not measurement noise.

Equal timepoint cost is not actual sequencing cost.
No independent known delay truth.

Gene deletion influence depends on the profiled
comparison direction.

Real 3h/6h differences may also reflect culture,
stimulus-independent time effects or technical changes.
No replicate-based SD is established here.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v107.py'
    )
    shutil.copy2(
        run/'source_v67_measured.csv',
        OUT/'source_measured_v67.csv'
    )

    (OUT/'08_manifest.json').write_text(
        json.dumps(dict(
            version='v107',
            source_sha256=SOURCE_SHA,
            code_sha256=sha(__file__),
            source_v106_manifest_sha256=sha(
                run/'09_manifest.json'
            ),
            n_cultures=4,
            n_genes=8,
            no_noise_estimate=True,
            no_biological_delay_truth=True,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform()
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[GATE] Conditional template contrasts only; '
        'gene consistency descriptive; '
        'biological delay NOT VALIDATED'
    )
    say(
        '[COMPLETE] Measured early-gene '
        'design comparison archived'
    )

def finish(ok):
    say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
    say('[REPORT] '+str(OUT/'console_summary.txt'))
    z=OUT/'portable_report_v107.zip'
    say('[ZIP] '+str(z))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    items=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '09_sha256_inventory.csv',z.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT/'09_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        z,'w',zipfile.ZIP_DEFLATED
    ) as out:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=z:
                out.write(p,p.name)

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
