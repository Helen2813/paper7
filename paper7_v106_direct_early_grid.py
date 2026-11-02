from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, zipfile, shutil, sys, traceback, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper7_direct_early_grid_v106' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
GENES = (
    'PRL','IGFBP1','FOXO1','HAND2',
    'WNT4','PGR','IL11','LEFTY2'
)
CULTURES = ('S169','S170','S506','S508')
SOURCE_SHA = 'eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a'

SCHEDULES = {
    'sparse_0_12_24': (0,12,24),
    'early_0_3_6': (0,3,6),
    'early_0_3_6_12_24': (0,3,6,12,24),
    'later_0_6_12_18_24': (0,6,12,18,24),
    'early_0_3_6_9_12_24': (0,3,6,9,12,24),
    'all_direct_0_3_6_9_12_18_24': (0,3,6,9,12,18,24),
}

def say(msg):
    msg = str(msg)
    print(msg, flush=True)
    LOG.append(msg)

def check(test, message):
    if not bool(test):
        raise RuntimeError(message)

def checksum(p):
    h = sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def csvout(name, data):
    d = data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(
        OUT/name,
        index=False,
        encoding='utf-8-sig',
        float_format='%.15g'
    )
    say(f'[OUTPUT] {name} | rows={len(d)}')
    return d

def verified_run(folder, files, inventory):
    base = ROOT/folder
    candidates = sorted(
        (
            p for p in base.glob('run_*')
            if p.is_dir()
            and all((p/f).is_file() for f in files)
        ),
        reverse=True
    )
    check(bool(candidates), f'Missing {folder} run')

    r = candidates[0]
    inv = pd.read_csv(r/inventory, encoding='utf-8-sig')
    check(
        {'filename','bytes','sha256'}.issubset(inv.columns)
        and inv.filename.is_unique,
        'Bad archive inventory'
    )

    for item in inv.itertuples(index=False):
        n = str(item.filename)
        p = r/n
        check(
            Path(n).name == n
            and p.is_file()
            and p.stat().st_size == int(item.bytes)
            and checksum(p) == str(item.sha256),
            f'Integrity error: {n}'
        )

    return r, len(inv)

def source():
    r104,n104 = verified_run(
        'paper7_shape_free_onset_v104',
        (
            '09_sha256_inventory.csv',
            'input_V67_source.csv',
            '05_shared_culture_tau_feasibility.csv'
        ),
        '09_sha256_inventory.csv'
    )

    r105,n105 = verified_run(
        'paper7_early_onset_constraints_v105',
        (
            '05_sha256_inventory.csv',
            '01_measured_early_constraints.csv',
            '02_core_vs_all_measured_0v6.csv'
        ),
        '05_sha256_inventory.csv'
    )

    check(
        checksum(r104/'input_V67_source.csv') == SOURCE_SHA,
        'Frozen V67 source changed'
    )

    m104 = json.loads(
        (r104/'08_manifest.json').read_text('utf-8')
    )
    m105 = json.loads(
        (r105/'04_manifest.json').read_text('utf-8')
    )

    check(
        m104.get('version') == 'v104'
        and m105.get('version') == 'v105',
        'Prior version mismatch'
    )

    raw = pd.read_csv(
        r104/'input_V67_source.csv',
        encoding='utf-8-sig'
    )
    prior = pd.read_csv(
        r105/'02_core_vs_all_measured_0v6.csv',
        encoding='utf-8-sig'
    )

    check(
        len(raw) == 320 and len(prior) == 16,
        'Unexpected source dimensions'
    )
    check(
        not raw.duplicated(
            ['culture_label','gene','time_hours']
        ).any(),
        'Duplicate timepoint'
    )

    curves = {}

    for cult in CULTURES:
        for gene in GENES:
            g = raw[
                (raw.culture_label == cult)
                & (raw.gene == gene)
            ].sort_values('time_hours')

            check(
                len(g) == 10
                and not g.log2CPM.isna().any(),
                'Incomplete curve'
            )

            t = g.time_hours.to_numpy(float)
            y = g.log2CPM.to_numpy(float)

            check(
                np.allclose(
                    t,
                    [0,3,6,9,12,18,24,36,48,96]
                ),
                f'Unexpected grid {cult} {gene}'
            )

            curves[(cult,gene)] = {
                int(ti):float(yi)
                for ti,yi in zip(t,y)
            }

    say(
        f'[SOURCE VERIFIED] V104={n104} artifacts | '
        f'V105={n105} artifacts | 32 real curves'
    )
    return curves,prior,r104,r105

def source_vector(curve,grid,tau):
    values = []
    for t in grid:
        knot = 0 if t <= tau else t-tau
        check(
            knot in curve,
            'Unobserved shifted source knot '+str(knot)
        )
        values.append(curve[knot])

    return np.asarray(values,float)

def profiled_distance(target, alternative):
    # Fit a free intercept and a nonnegative response amplitude.
    xc = alternative-alternative.mean()
    yc = target-target.mean()
    denom = float(xc@xc)

    amplitude = (
        max(0.,float(xc@yc)/denom)
        if denom > 1e-14 else 0.
    )
    intercept = float(
        target.mean()-amplitude*alternative.mean()
    )
    residual = target-intercept-amplitude*alternative

    return float(residual@residual)

def contrast(curves,ca,cb,grid):
    forward = 0.
    reverse = 0.
    per_gene = []

    for gene in GENES:
        curve_a = curves[(ca,gene)]
        curve_b = curves[(cb,gene)]

        a0 = source_vector(curve_a,grid,0)
        a6 = source_vector(curve_a,grid,6)
        b0 = source_vector(curve_b,grid,0)
        b6 = source_vector(curve_b,grid,6)

        s0 = profiled_distance(a0,b6)
        s1 = profiled_distance(a6,b0)

        forward += s0
        reverse += s1

        per_gene.append(dict(
            gene=gene,
            sse_forward=s0,
            sse_reverse=s1
        ))

    return forward,reverse,per_gene

def measured_pre(curves,prior):
    rows = []

    for cult in CULTURES:
        flat = 0.
        drift = 0.

        for gene in GENES:
            curve = curves[(cult,gene)]
            t = np.array([0.,3.,6.])
            y = np.array([
                curve[int(v)] for v in t
            ])

            X0 = np.ones((3,1))
            X1 = np.column_stack((np.ones(3),t))

            r0 = y-X0@np.linalg.lstsq(
                X0,y,rcond=None
            )[0]
            r1 = y-X1@np.linalg.lstsq(
                X1,y,rcond=None
            )[0]

            second = float(y[0]-2*y[1]+y[2])

            check(
                abs(float(r1@r1)-second**2/6) < 1e-10,
                'Second difference identity failed'
            )

            flat += float(r0@r0)
            drift += float(r1@r1)

            rows.append(dict(
                culture=cult,
                gene=gene,
                y0_log2CPM=y[0],
                y3_log2CPM=y[1],
                y6_log2CPM=y[2],
                delta_3h=y[1]-y[0],
                delta_6h=y[2]-y[0],
                second_difference_0_3_6=second,
                flat_pre_RMS=float(
                    np.sqrt((r0@r0)/3)
                ),
                drift_pre_RMS=float(
                    np.sqrt((r1@r1)/3)
                ),
                empirical_noise_SD_available=False
            ))

        rms0 = float(np.sqrt(flat/24))
        rms1 = float(np.sqrt(drift/24))

        subset = prior[
            (prior.culture == cult)
            & (prior.design == 'all_measured')
        ]

        for kind,got in (
            ('flat',rms0),
            ('linear_drift',rms1)
        ):
            old = subset[subset.baseline == kind]
            check(
                len(old) == 1
                and abs(
                    float(old.iloc[0].tau6_pre_RMS)-got
                ) < 1e-10,
                f'V105 parity failure: {cult} {kind}'
            )

        say(
            f'[EARLY DIRECT | {cult}] '
            f'flat={rms0:.5f} drift={rms1:.5f} '
            f'log2CPM | V105 exact'
        )

    csvout('02_measured_0_3_6_gene_contrasts.csv',rows)

def run():
    say('='*90)
    say('PAPER 7 | V106 | 0 vs 6h | DIRECT MEASURED EARLY-GRID AUDIT')
    say('NO IMPUTATION | NO DELAY TRUTH | NO ESTIMATED NOISE | NO GIT')
    say('='*90)

    check(len(sys.argv)==1,'Run without flags')

    curves,prior,r104,r105 = source()

    protocol = dict(
        version='v106',
        purpose='conditional six-hour shift using measured source knots',
        source_sha256=SOURCE_SHA,
        source_grid=[0,3,6,9,12,18,24,36,48,96],
        delays=[0,6],
        schedules=SCHEDULES,
        nuisance='gene-specific nonnegative amplitude and free baseline',
        tau6_is_hypothetical_not_observed=True,
        no_interpolated_time_inputs=True,
        sigma_available=False,
        no_network=True,
        no_git=True,
        script_sha256=checksum(__file__)
    )

    (OUT/'01_protocol.json').write_text(
        json.dumps(protocol,indent=2)+'\n',
        encoding='utf-8'
    )

    measured_pre(curves,prior)

    mapping = []
    for name,grid in SCHEDULES.items():
        for tt in grid:
            src_t = 0 if tt <= 6 else tt-6
            check(
                src_t in (
                    0,3,6,9,12,18,24,36,48,96
                ),
                'Missing direct source knot'
            )
            mapping.append(dict(
                schedule=name,
                acquisition_hour=tt,
                tau0_source_hour=tt,
                tau6_source_hour=src_t,
                tau6_pre_onset_baseline_assumed=bool(tt<=6)
            ))

    csvout('03_direct_reindex_mapping.csv',mapping)

    rows = []
    genes = []

    for name,grid in SCHEDULES.items():
        for ca in CULTURES:
            for cb in CULTURES:
                s0,s6,parts = contrast(
                    curves,ca,cb,grid
                )

                if ca == cb:
                    for item in parts:
                        genes.append(dict(
                            schedule=name,
                            culture=ca,
                            n_hours=len(grid),
                            **item
                        ))

                rows.append(dict(
                    schedule=name,
                    culture_target=ca,
                    culture_alternative=cb,
                    contrast=(
                        'same_culture'
                        if ca==cb else 'cross_culture'
                    ),
                    n_genes=8,
                    n_times=len(grid),
                    n_gene_timepoints=8*len(grid),
                    total_sse_0_from_6=s0,
                    total_sse_6_from_0=s6,
                    sqrtSSE_forward=np.sqrt(s0),
                    sqrtSSE_reverse=np.sqrt(s6),
                    conservative_direction_sqrtSSE=np.sqrt(
                        min(s0,s6)
                    ),
                    conservative_RMS_log2CPM=np.sqrt(
                        min(s0,s6)/(8*len(grid))
                    ),
                    delay_truth_available=False,
                    empirical_sigma_available=False
                ))

        say(
            '[GRID] {} | {} times | '
            'only measured reference knots'.format(
                name,len(grid)
            )
        )

    rows = csvout(
        '04_direct_0v6_profile_distances.csv',rows
    )
    csvout(
        '05_within_culture_gene_distances.csv',genes
    )

    same = rows[
        rows.contrast == 'same_culture'
    ]

    pivot = same.pivot(
        index='culture_target',
        columns='schedule',
        values='conservative_direction_sqrtSSE'
    )

    base = pivot['sparse_0_12_24']
    full = pivot[
        'all_direct_0_3_6_9_12_18_24'
    ]

    check(
        np.all(full+1e-10 >= base),
        'Adding measured points reduced minimized SSE'
    )

    comparisons = []
    for cult in CULTURES:
        comparisons.append(dict(
            culture=cult,
            sparse_sqrtSSE=float(base.loc[cult]),
            seven_time_sqrtSSE=float(full.loc[cult]),
            incremental_sqrtSSE=float(
                full.loc[cult]-base.loc[cult]
            ),
            additional_timepoints=4,
            sigma_available=False
        ))

        say(
            '[V106 0v6 | {}] sparse sqrtSSE={:.5f} | '
            'seven-time sqrtSSE={:.5f} | '
            'sigma UNKNOWN'.format(
                cult,base.loc[cult],full.loc[cult]
            )
        )

    csvout(
        '06_early_grid_information_comparison.csv',
        comparisons
    )

    summary = []
    for grid,group in rows.groupby('schedule'):
        within = group[
            group.contrast == 'same_culture'
        ]
        across = group[
            group.contrast == 'cross_culture'
        ]

        summary.append(dict(
            schedule=grid,
            n_times=int(group.n_times.iloc[0]),
            min_within_RMS=float(
                within.conservative_RMS_log2CPM.min()
            ),
            median_within_RMS=float(
                within.conservative_RMS_log2CPM.median()
            ),
            min_cross_culture_RMS=float(
                across.conservative_RMS_log2CPM.min()
            ),
            median_cross_culture_RMS=float(
                across.conservative_RMS_log2CPM.median()
            ),
            sigma_available=False
        ))

    csvout(
        '07_schedule_culture_heterogeneity.csv',
        summary
    )

    (OUT/'08_methods_and_limits.md').write_text(
        """# V106: Direct measured early-grid audit

The original source has observed 0, 3, 6, 9, 12,
18, 24, 36, 48 and 96 hour measurements.

V106 uses only measured reference values. A
hypothetical tau=6 shift is constructed by
reindexing existing source values at t-6 and
assuming a flat response before onset.

This is NOT an experimentally observed delay.

Each gene is allowed its own nuisance baseline
and nonnegative amplitude. The square-root SSE
and RMS are conditional template differences.

Neither value is normalized by experimental noise.
No empirical replicate SD or covariance exists
in the frozen source snapshot.

Cross-culture comparisons are descriptive and
must not be treated as technical noise estimates.

The 0,3,6h contrast reconstructs the earlier
V105 flat and linear-drift errors exactly.

Extra measured times mechanically cannot decrease
the total best-fit SSE for nested designs.
This alone does not demonstrate biological power.

Onset-0 with arbitrary response shape can imitate
an onset-6 response with internal latency.
The result is NOT global delay identifiability.

No new data are downloaded or generated.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v106.py'
    )
    shutil.copy2(
        r104/'input_V67_source.csv',
        OUT/'source_v67_measured.csv'
    )
    shutil.copy2(
        r105/'02_core_vs_all_measured_0v6.csv',
        OUT/'source_v105_comparison.csv'
    )

    (OUT/'09_manifest.json').write_text(
        json.dumps(dict(
            version='v106',
            script_sha256=checksum(__file__),
            source_v67_sha256=SOURCE_SHA,
            n_source_curves=32,
            n_schedules=len(SCHEDULES),
            no_interpolation=True,
            measured_delayed_samples=False,
            sigma_available=False,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            no_git=True
        ),indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[GATE] 0v6 separation conditional; '
        'measured 3h/6h available; '
        'noise UNKNOWN; delayed truth UNAVAILABLE'
    )
    say('[COMPLETE] V106 direct-grid audit archived')

def finish(ok):
    say(
        '[RESULT] '+(
            'SUCCESS' if ok else 'FAILED'
        )
    )

    say('[REPORT] '+str(
        OUT/'console_summary.txt'
    ))

    archive = OUT/'portable_report_v106.zip'
    say('[ZIP] '+str(archive))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    entries = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '10_sha256_inventory.csv',archive.name
        ):
            entries.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=checksum(p)
            ))

    pd.DataFrame(entries).to_csv(
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

if __name__ == '__main__':
    ok = True
    try:
        run()
    except Exception:
        ok = False
        say('[ERROR]\n'+traceback.format_exc())
    finally:
        finish(ok)
    if not ok:
        sys.exit(1)
