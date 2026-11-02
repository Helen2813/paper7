#!/usr/bin/env python3
"""Paper 7 V108. Read-only gene/culture robustness audit of V107."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, sys, traceback, zipfile, shutil, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper7_early_design_robustness_v108' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
GENES = ('PRL','IGFBP1','FOXO1','HAND2',
         'WNT4','PGR','IL11','LEFTY2')
CULTURES = ('S169','S170','S506','S508')
EXTRAS = ('plus_3','plus_6','plus_9','plus_18')
BASE = 'base_0_12_24'
SOURCE_SHA = (
    'eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a'
)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def save(name, data):
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(
        OUT/name, index=False,
        encoding='utf-8-sig',
        float_format='%.15g'
    )
    say('[OUTPUT] {} | rows={}'.format(name,len(df)))
    return df

def as_bool(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin(('true','1'))
    )

def load_v107():
    parent = ROOT / 'paper7_early_gene_design_v107'
    required = (
        '09_sha256_inventory.csv',
        '08_manifest.json',
        '02_measured_gene_early_contrasts.csv',
        '03_gene_consistency_four_cultures.csv',
        '04_equal_cost_conditional_designs.csv',
        '05_gene_leave_one_out.csv',
        '06_additional_timepoint_value.csv',
        'script_snapshot_v107.py'
    )

    runs = sorted(
        (
            p for p in parent.glob('run_*')
            if p.is_dir()
            and all((p/n).is_file() for n in required)
        ),
        reverse=True
    )
    need(runs, 'Complete V107 archive not found')
    run = runs[0]

    inv = pd.read_csv(
        run/'09_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    need(
        {'filename','bytes','sha256'}.issubset(inv.columns)
        and inv.filename.is_unique,
        'Invalid V107 inventory'
    )

    for r in inv.itertuples(index=False):
        name = str(r.filename)
        path = run/name
        need(
            Path(name).name == name
            and path.is_file()
            and path.stat().st_size == int(r.bytes)
            and sha(path) == str(r.sha256),
            'V107 archive mismatch: '+name
        )

    manifest = json.loads(
        (run/'08_manifest.json').read_text('utf-8')
    )
    need(
        manifest.get('version') == 'v107'
        and manifest.get('source_sha256') == SOURCE_SHA,
        'V107 manifest/source mismatch'
    )

    source_code = ROOT/'paper7_v107_early_information.py'
    need(
        source_code.is_file()
        and sha(source_code) == sha(run/'script_snapshot_v107.py'),
        'V107 source code differs from archived snapshot'
    )

    early = pd.read_csv(
        run/'02_measured_gene_early_contrasts.csv',
        encoding='utf-8-sig'
    )
    genes = pd.read_csv(
        run/'03_gene_consistency_four_cultures.csv',
        encoding='utf-8-sig'
    )
    design = pd.read_csv(
        run/'04_equal_cost_conditional_designs.csv',
        encoding='utf-8-sig'
    )
    loo = pd.read_csv(
        run/'05_gene_leave_one_out.csv',
        encoding='utf-8-sig'
    )
    compare = pd.read_csv(
        run/'06_additional_timepoint_value.csv',
        encoding='utf-8-sig'
    )

    need(
        len(early)==32 and len(genes)==8
        and len(design)==176 and len(loo)==352
        and len(compare)==44,
        'Unexpected V107 table size'
    )

    need(
        set(early.culture)==set(CULTURES)
        and set(early.gene)==set(GENES),
        'Unexpected cultures or genes'
    )
    need(
        set(loo.schedule)==set(compare.schedule)
        and len(set(loo.schedule))==11,
        'Design library incomplete'
    )
    need(
        not compare.duplicated(['culture','schedule']).any()
        and not loo.duplicated(
            ['culture','schedule','gene']
        ).any(),
        'Duplicated archived design results'
    )

    for col in ('same_sign_0to3','same_sign_curvature'):
        genes[col] = as_bool(genes[col])
    design['same_culture'] = as_bool(design['same_culture'])
    compare['cross_culture_looks_closer'] = as_bool(
        compare['cross_culture_looks_closer']
    )

    say(
        '[V107 VERIFIED] {} | inventory={} files'
        .format(run,len(inv))
    )
    return run,early,genes,design,loo,compare

def audit(early,genes,design,loo,compare):
    # Reconstruct the nearest other-culture comparisons.
    cross_rows = []

    for r in compare.itertuples(index=False):
        d = design[
            (design.schedule==r.schedule)
            & (design.culture_target==r.culture)
        ]

        need(
            len(d)==4 and int(d.same_culture.sum())==1,
            'Incomplete culture comparison'
        )

        same = d[d.same_culture].iloc[0]
        other = (
            d[~d.same_culture]
            .sort_values('min_SSE').iloc[0]
        )

        need(
            abs(float(same.min_SSE)-float(r.within_SSE))<1e-9
            and abs(
                float(other.min_SSE)-float(r.nearest_other_SSE)
            )<1e-9,
            'Archived cross-culture distances differ'
        )
        need(
            str(other.culture_alternative)
            ==str(r.nearest_other_culture),
            'Nearest culture changed'
        )
        need(
            bool(other.min_SSE<same.min_SSE-1e-10)
            ==bool(r.cross_culture_looks_closer),
            'Cross-culture comparison flag mismatch'
        )

        cross_rows.append(dict(
            culture=r.culture,
            schedule=r.schedule,
            within_SSE=float(same.min_SSE),
            nearest_other=str(other.culture_alternative),
            nearest_other_SSE=float(other.min_SSE),
            cross_closer=bool(r.cross_culture_looks_closer),
            margin_other_minus_within=float(
                other.min_SSE-same.min_SSE
            )
        ))

    checked = save(
        '02_cross_culture_nearest_audit.csv',
        cross_rows
    )

    by_schedule = (
        checked.groupby('schedule')
        .agg(
            n_targets=('cross_closer','size'),
            other_closer=('cross_closer','sum'),
            mean_margin=('margin_other_minus_within','mean'),
            min_margin=('margin_other_minus_within','min')
        )
        .reset_index()
    )
    by_schedule['other_closer_fraction'] = (
        by_schedule.other_closer/by_schedule.n_targets
    )
    save('03_cross_culture_by_schedule.csv',by_schedule)

    need(
        int(checked.cross_closer.sum())
        ==int(compare.cross_culture_looks_closer.sum()),
        'Total cross-culture parity failed'
    )

    say(
        '[CROSS CULTURE REPLAY] other closer={}/44'
        .format(int(checked.cross_closer.sum()))
    )

    # Compare equal-cost schedules; repeat after deleting a gene.
    singles = compare[
        compare.schedule.isin(EXTRAS)
    ].copy()

    need(
        len(singles)==16
        and (singles.extra_hours==1).all(),
        'Single-extra designs incomplete'
    )

    baseline_rank = []
    deletion = []
    contributions = []

    for culture in CULTURES:
        values = singles[
            singles.culture==culture
        ].set_index('schedule').incremental_SSE.astype(float)

        need(
            len(values)==4 and values.index.is_unique,
            'Missing one-time candidates'
        )

        winner = values.idxmax()
        next_best = float(values.drop(winner).max())

        baseline_rank.append(dict(
            culture=culture,
            best_one_extra=winner,
            gain_best=float(values[winner]),
            gain_plus3=float(values['plus_3']),
            gain_plus6=float(values['plus_6']),
            margin_6_minus_3=float(
                values['plus_6']-values['plus_3']
            ),
            winner_margin=float(
                values.max()-next_best
            ),
            plus6_wins=bool(winner=='plus_6')
        ))

        for gene in GENES:
            sub = loo[
                (loo.culture==culture)
                & (loo.gene==gene)
            ].set_index('schedule')

            need(len(sub)==11,'Incomplete leave-one-gene-out table')

            base = float(sub.loc[BASE,'SSE_without_gene'])

            gains = {
                name:float(
                    sub.loc[name,'SSE_without_gene']
                )-base
                for name in EXTRAS
            }

            need(
                min(gains.values())>-1e-8,
                'Nested-design SSE condition failed'
            )

            best = max(
                EXTRAS,key=lambda name:gains[name]
            )
            second = max(
                gains[name]
                for name in EXTRAS if name!=best
            )

            deletion.append(dict(
                culture=culture,
                excluded_gene=gene,
                original_best=winner,
                best_one_extra=best,
                plus6_wins=bool(best=='plus_6'),
                gain_plus3=gains['plus_3'],
                gain_plus6=gains['plus_6'],
                gain_plus9=gains['plus_9'],
                gain_plus18=gains['plus_18'],
                margin_6_minus_3=(
                    gains['plus_6']-gains['plus_3']
                ),
                winner_margin=gains[best]-second,
                plus6_loses_to_best=(
                    gains[best]-gains['plus_6']
                )
            ))

            contributions.append(dict(
                culture=culture,
                gene=gene,
                full_gain_plus6=float(values['plus_6']),
                leave_one_out_gain_plus6=gains['plus_6'],
                apparent_gain_contribution=float(
                    values['plus_6']-gains['plus_6']
                )
            ))

    ranking = save(
        '04_equal_cost_single_extra_ranking.csv',
        baseline_rank
    )
    deletion = save(
        '05_gene_deletion_schedule_stability.csv',
        deletion
    )
    contributions = save(
        '06_gene_influence_on_plus6_gain.csv',
        contributions
    )

    gene_summary = (
        contributions.groupby('gene')
        .agg(
            mean_apparent_gain_contribution=(
                'apparent_gain_contribution','mean'
            ),
            min_apparent_gain_contribution=(
                'apparent_gain_contribution','min'
            ),
            max_apparent_gain_contribution=(
                'apparent_gain_contribution','max'
            )
        )
        .reset_index()
        .sort_values(
            'mean_apparent_gain_contribution',
            ascending=False
        )
    )
    save(
        '07_genes_ranked_by_plus6_influence.csv',
        gene_summary
    )

    need(
        ranking.plus6_wins.all(),
        'V107 original +6h ranking mismatch'
    )

    say(
        '[LOO GENES] plus6 remains best in {}/32 gene deletions'
        .format(int(deletion.plus6_wins.sum()))
    )

    for r in deletion[
        ~deletion.plus6_wins
    ].itertuples(index=False):
        say(
            '[LOO REVERSAL] {} remove {} -> {}'
            .format(
                r.culture,r.excluded_gene,r.best_one_extra
            )
        )

    # Diagnostic leave-one-culture-out, no real delay truth.
    wide = singles.pivot(
        index='culture',
        columns='schedule',
        values='incremental_SSE'
    )

    holdout = []

    for target in CULTURES:
        train = wide.drop(index=target).mean(axis=0)
        selected = train.idxmax()

        heldout = wide.loc[target]
        best = heldout.idxmax()

        holdout.append(dict(
            held_out_culture=target,
            selected_by_three_cultures=selected,
            heldout_winner=best,
            agreement=bool(selected==best),
            heldout_gain_selected=float(heldout[selected]),
            heldout_gain_best=float(heldout[best]),
            regret_in_SSE=float(
                heldout[best]-heldout[selected]
            )
        ))

    holdout = save(
        '08_leave_one_culture_out_selection_diagnostic.csv',
        holdout
    )

    say(
        '[LOCO DESIGN] winner agrees in {}/4 held-out cultures'
        ' (not independent validation)'
        .format(int(holdout.agreement.sum()))
    )

    gene_signs = genes[[
        'gene','mean_delta0to3','same_sign_0to3',
        'same_sign_curvature','mean_abs_curvature'
    ]].merge(
        gene_summary,on='gene',validate='one_to_one'
    )

    save(
        '09_gene_consistency_and_design_impact.csv',
        gene_signs
    )

    say(
        '[GENE SIGNS] 0->3 same sign {}/8; '
        'early curvature same sign {}/8'
        .format(
            int(genes.same_sign_0to3.sum()),
            int(genes.same_sign_curvature.sum())
        )
    )

    for r in baseline_rank:
        say(
            '[ONE EXTRA | {}] best={} | '
            '+6 minus +3={:+.4f} SSE'
            .format(
                r['culture'],
                r['best_one_extra'],
                r['margin_6_minus_3']
            )
        )

    for r in by_schedule.sort_values(
        'schedule'
    ).itertuples(index=False):
        say(
            '[OTHER-CULTURE | {}] closer={}/4 | '
            'mean_margin={:+.4f}'
            .format(
                r.schedule,r.other_closer,r.mean_margin
            )
        )

    return deletion,holdout

def main():
    say('='*88)
    say('PAPER 7 | V108 | MEASURED EARLY DESIGN: GENE / CULTURE ROBUSTNESS')
    say(
        'OFFLINE | ARCHIVED V107 ONLY | '
        'NO NEW SIMULATION | NO DELAY TRUTH | NO GIT'
    )
    say('='*88)

    need(
        len(sys.argv)==1,
        'Run without flags'
    )

    run,early,genes,design,loo,compare = load_v107()

    protocol = dict(
        version='v108',
        source_v107_manifest_sha256=sha(
            run/'08_manifest.json'
        ),
        tests=[
            'equal-cost single-extra comparison',
            'leave-one-gene-out',
            'leave-one-culture-out',
            'nearest alternative culture'
        ],
        no_training=True,
        no_new_simulation=True,
        no_biological_delay_truth=True,
        measured_noise_sigma_available=False,
        no_network=True,
        no_git=True,
        script_sha256=sha(__file__)
    )

    (OUT/'01_protocol.json').write_text(
        json.dumps(protocol,indent=2)+'\n',
        encoding='utf-8'
    )

    deletion,holdout = audit(
        early,genes,design,loo,compare
    )

    (OUT/'10_methods_and_limitations.md').write_text(
        """# Paper 7 V108 — design robustness audit

Only frozen V107 tables are used. No new data,
synthetic trajectories, model fitting or selection
of a deployable classifier.

The score is the minimum of two direction-specific
sums of gene-wise squared residuals after profiling
gene-specific baselines and nonnegative amplitudes.

The six-hour delay is hypothetical and generated
by reindexing measured source values. It is not an
independently observed delayed biological response.

Baseline time grid: 0,12,24h.
Equal-cost additions: 3,6,9,18h.

Leave-one-gene-out is a sensitivity analysis, not
independent biological validation. Gene contributions
can be negative because the minimizing direction
may switch after deletion.

Leave-one-culture-out uses only four already-studied
cultures and is not independent external validation.

Cross-culture nearest-template counts are NOT
classification error rates.

Additional measured points cannot decrease minimum
total SSE for nested schedules; raw gains are
model-dependent and not measured statistical power.

Practical detectability is unknown without a
replicate-derived noise model and covariance.

V104 nonidentifiability with arbitrary smooth
post-onset responses remains valid.

All prior results stay unchanged.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v108.py'
    )

    for source,target in (
        ('08_manifest.json','input_v107_manifest.json'),
        ('09_sha256_inventory.csv','input_v107_inventory.csv'),
        ('04_equal_cost_conditional_designs.csv',
         'input_v107_equal_cost.csv'),
        ('05_gene_leave_one_out.csv',
         'input_v107_gene_deletion.csv'),
        ('06_additional_timepoint_value.csv',
         'input_v107_candidates.csv'),
        ('03_gene_consistency_four_cultures.csv',
         'input_v107_gene_signs.csv')
    ):
        shutil.copy2(run/source,OUT/target)

    (OUT/'11_manifest.json').write_text(
        json.dumps(dict(
            version='v108',
            script_sha256=sha(__file__),
            source_v107_manifest_sha256=sha(
                run/'08_manifest.json'
            ),
            gene_deletions=len(deletion),
            culture_holdouts=len(holdout),
            no_new_data=True,
            noise_sigma_available=False,
            no_delay_truth=True,
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
        '[GATE] DESCRIPTIVE ONLY; '
        'no independent delayed cultures or noise SD'
    )
    say('[COMPLETE] V107 robustness audit archived')

def finish(ok):
    say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
    say('[REPORT] '+str(OUT/'console_summary.txt'))

    archive = OUT/'portable_report_v108.zip'
    say('[ZIP] '+str(archive))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    inventory = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '12_sha256_inventory.csv',archive.name
        ):
            inventory.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(inventory).to_csv(
        OUT/'12_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive,'w',zipfile.ZIP_DEFLATED
    ) as bundle:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                bundle.write(p,p.name)

if __name__=='__main__':
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say('[ERROR]\n'+traceback.format_exc())
    finally:
        finish(ok)

    if not ok:
        sys.exit(1)
