#!/usr/bin/env python3
"""Paper 7 V103. Measured-time identifiability audit.

Reads archived V67 source values, no synthetic delay labels.
Distances are geometry of model-conditional templates,
NOT validated delay estimates.

No measured replicate-based noise:
no claims of practical signal/noise separation.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import platform
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper7_measured_identifiability_v103' / (
    'run_' +
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []

GENES = (
    'PRL', 'IGFBP1', 'FOXO1', 'HAND2',
    'WNT4', 'PGR', 'IL11', 'LEFTY2'
)

CULTURES = ('S169', 'S170', 'S506', 'S508')

DELAYS = (0., 6., 12., 24.)
CORE = (0., 12., 24., 36., 48.)
ADD = (2., 4., 6.)

SPEEDS = (.70, .85, 1., 1.15, 1.30)
EPS = .02


def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)


def ensure(condition, why):
    if not bool(condition):
        raise RuntimeError(why)


def digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(filename, data):
    frame = (
        data if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data)
    )
    frame.to_csv(
        OUT / filename,
        index=False,
        encoding='utf-8-sig',
        float_format='%.15g'
    )
    say('[OUTPUT] {} | {} rows'.format(
        filename, len(frame)
    ))
    return frame


def find_archive():
    folder = ROOT / 'paper7_gse403_locked_external_v67'

    needed = (
        'frozen_v31_source.csv',
        '08_manifest.json',
        'script_snapshot.py'
    )

    runs = sorted(
        (
            p for p in folder.glob('run_*')
            if p.is_dir()
            and all((p / n).is_file() for n in needed)
        ),
        reverse=True
    )

    for run in runs:
        try:
            man = json.loads(
                (run / '08_manifest.json').read_text('utf-8')
            )

            if man.get('version') != 'v67':
                continue

            if man.get('geo_accession') != 'GSE403':
                continue

            if digest(
                run / 'frozen_v31_source.csv'
            ).lower() != str(
                man.get('source_sha256')
            ).lower():
                continue

            if digest(
                run / 'script_snapshot.py'
            ).lower() != str(
                man.get('script_sha256')
            ).lower():
                continue

            say(
                '[V67 VERIFIED] source SHA256={}'.format(
                    digest(run / 'frozen_v31_source.csv')
                )
            )
            return run, man

        except (
            OSError, KeyError, ValueError,
            TypeError, json.JSONDecodeError
        ):
            continue

    raise RuntimeError(
        'No SHA256-verified V67 source archive found'
    )


def verify_algebraic_aliases():
    """Exact algebraic witnesses, not fitted biology."""
    t = np.array([0., 2., 5., 12., 24., 36., 48.])
    h = 6.

    functions = {
        'pure_exponential':
            lambda x: np.exp(.04 * x),

        'affine':
            lambda x: 2. + .12 * x,

        'exponential_times_periodic_for_fixed_6h':
            lambda x: (
                np.exp(.04 * x)
                * (1. + .2 * np.sin(2. * np.pi * x / 6.))
            )
    }

    rows = []

    for name, func in functions.items():
        f0 = func(t)
        f6 = func(t - h)

        design = np.column_stack((
            np.ones(len(t)), f0
        ))

        coef = np.linalg.lstsq(
            design, f6, rcond=None
        )[0]

        rmse = float(
            np.linalg.norm(f6 - design @ coef)
            / np.sqrt(len(t))
        )

        ensure(
            rmse < 1e-10,
            'Analytic alias witness failed: ' + name
        )

        rows.append(dict(
            shape=name,
            delay_shift_hours=h,
            profiled_RMS_log2CPM_illustrative=rmse,
            all_time_domain_no_causal_clamping=True,
            numerical_identity_verified=True
        ))

    save(
        '00_analytic_alias_identity_checks.csv',
        rows
    )

    say(
        '[ANALYTIC ALIASES] exponential, affine, '
        '6h-periodic-modulated exponential: 3/3 exact'
    )


def load_curves(run):
    raw = pd.read_csv(
        run / 'frozen_v31_source.csv',
        encoding='utf-8-sig'
    )

    required = {
        'culture_label', 'gene',
        'time_hours', 'log2CPM'
    }

    ensure(
        required.issubset(raw.columns),
        'Frozen source schema changed'
    )

    raw = raw[
        raw.gene.isin(GENES)
        & raw.culture_label.isin(CULTURES)
    ].copy()

    ensure(
        len(raw) == 4 * 8 * 10,
        'Expected 320 measured entries'
    )

    ensure(
        not raw.duplicated([
            'culture_label', 'gene', 'time_hours'
        ]).any(),
        'Duplicate gene/culture/time'
    )

    curves = {}
    inventory = []
    noise = []
    bump = []

    for culture in CULTURES:
        for gene in GENES:
            part = raw[
                (raw.culture_label == culture)
                & (raw.gene == gene)
            ].sort_values('time_hours')

            ensure(
                len(part) == 10,
                'Missing gene/culture time course'
            )

            t = part.time_hours.to_numpy(float)
            y = part.log2CPM.to_numpy(float)

            ensure(
                np.isfinite(t).all()
                and np.isfinite(y).all()
                and np.all(np.diff(t) > 0),
                'Invalid time/expression values'
            )

            ensure(
                np.isclose(t, 0).any()
                and np.isclose(t, 36).any()
                and not np.isclose(t, 2).any(),
                'Unexpected 0h/2h/36h assayability'
            )

            curves[(culture, gene)] = (t, y)

            left = t[t < 2].max()
            right = t[t > 2].min()

            u = (2 - left) / (right - left)
            bump_basis = float(
                np.sin(np.pi * u)**2
            )

            ensure(
                bump_basis > 0,
                'Invalid 2h interpolation interval'
            )

            inventory.append(dict(
                culture=culture,
                gene=gene,
                n_measured=len(t),
                observed_hours=';'.join(
                    '%g' % x for x in t
                ),
                measured_2h=False,
                measured_36h=True,
                core_measured_count=int(
                    np.isin(CORE, t).sum()
                ),
                source_sha256=digest(
                    run / 'frozen_v31_source.csv'
                )
            ))

            bump.append(dict(
                culture=culture,
                gene=gene,
                left_hour=left,
                right_hour=right,
                basis_value_at_2h=bump_basis,
                amplitude_for_plus_0p5_log2CPM=(
                    .5 / bump_basis
                ),
                amplitude_for_minus_0p5_log2CPM=(
                    -.5 / bump_basis
                ),
                measured_knots_unchanged=True,
                C1_smooth_interval_bump=True
            ))

            noise.append(dict(
                culture=culture,
                gene=gene,
                n_unique_gene_culture_time=10,
                distinct_technical_replicates_available=False,
                empirical_sigma_log2CPM=np.nan,
                practical_identifiability_certified=False,
                reason=(
                    'One value per gene/culture/time; '
                    'no replicate-level residuals in snapshot'
                )
            ))

    save('01_measured_time_inventory.csv', inventory)
    save('02_empirical_noise_availability.csv', noise)
    save('03_unobserved_2h_smooth_bump_witnesses.csv', bump)

    say(
        '[ASSAYABILITY] observed source 2h=0/32; '
        'observed source 36h=32/32'
    )

    say(
        '[NOISE GATE] No gene/culture/time replicate '
        'series in frozen snapshot: sigma=NOT ESTIMABLE'
    )

    return curves


def response(interp, t, delay, speed=1.):
    """Conditional causal-onset shifted template."""
    grid = np.asarray(t, float)

    x = speed * np.maximum(
        grid - delay, 0.
    )

    x = np.minimum(
        x, interp.x[-1]
    )

    return np.asarray(interp(x), float)


def make_interpolator(t, y):
    baseline = float(
        y[np.argmin(abs(t))]
    )

    return PchipInterpolator(
        t,
        y - baseline,
        extrapolate=False
    )


def schedules(t):
    measured = lambda hour: bool(
        np.isclose(t, hour, atol=1e-8).any()
    )

    base = tuple(
        x for x in CORE if measured(x)
    )

    ensure(
        len(base) >= 3,
        'Too few measured core timepoints'
    )

    ans = {
        'all_measured': tuple(t),
        'core_measured': base
    }

    for x in ADD:
        ans[
            'core_plus_%gh' % x
        ] = tuple(sorted(set(base + (x,))))

    return ans


def nuisance_projection(y, X):
    coef = np.linalg.lstsq(
        X, y, rcond=1e-10
    )[0]

    residual = y - X @ coef

    _, sing, _ = np.linalg.svd(
        X, full_matrices=False
    )

    rank = (
        int(np.sum(sing > sing[0] * 1e-10))
        if len(sing) and sing[0] > 0
        else 0
    )

    return residual, rank


def local_geometry(interp, t, delay, with_speed):
    f = response(interp, t, delay)

    grad = (
        response(interp, t, delay + EPS) - f
    ) / EPS

    columns = [
        np.ones(len(t)),
        f
    ]

    if with_speed:
        speed_grad = (
            response(interp, t, delay, 1. + EPS)
            - response(interp, t, delay, 1. - EPS)
        ) / (2 * EPS)

        columns.append(speed_grad)

    X = np.column_stack(columns)

    leftover, rank = nuisance_projection(
        grad, X
    )

    raw = float(np.linalg.norm(grad))
    projected = float(np.linalg.norm(leftover))

    return dict(
        model=(
            'baseline_amplitude_speed'
            if with_speed else
            'baseline_amplitude'
        ),
        nuisance_rank=rank,
        raw_derivative_norm_log2CPM_per_hour=raw,
        residual_derivative_norm_log2CPM_per_hour=projected,
        local_geometry_squared=projected**2,
        fractional_unconfounded_sensitivity=(
            projected / raw
            if raw > 1e-12 else np.nan
        ),
        local_numerical_alias=bool(
            projected < 1e-7 * max(raw, 1.)
        ),
        sigma_available=False
    )


def profile_direction(
    interp, t, tau_target, tau_competitor,
    speed_search
):
    target = response(
        interp, t, tau_target
    )

    speeds = (
        SPEEDS if speed_search else (1.,)
    )

    best = None

    for speed in speeds:
        alt = response(
            interp, t, tau_competitor, speed
        )

        alt_c = alt - alt.mean()
        target_c = target - target.mean()

        aa = max(
            0.,
            float(
                np.dot(alt_c, target_c)
                / max(
                    np.dot(alt_c, alt_c), 1e-14
                )
            )
        )

        bb = float(
            target.mean() - aa * alt.mean()
        )

        residual = target - (bb + aa * alt)
        ss = float(np.dot(residual, residual))

        if best is None or ss < best['sse']:
            best = dict(
                sse=ss,
                rmse=float(
                    np.sqrt(ss / len(t))
                ),
                amplitude=aa,
                baseline=bb,
                speed=float(speed),
                target_range=float(
                    np.ptp(target)
                )
            )

    return best


def fit_degenerate_classes(t, y):
    target = np.asarray(y, float)

    X = np.column_stack([
        np.ones(len(t)), t
    ])

    res, _ = nuisance_projection(
        target, X
    )

    affine = float(
        np.linalg.norm(res) / np.sqrt(len(t))
    )

    best = (np.inf, np.nan)

    # Approximate family b + a*exp(k*t).
    for k in np.r_[
        np.linspace(-.12, -.003, 45),
        np.linspace(.003, .12, 45)
    ]:
        z = np.expm1(k * t)

        Z = np.column_stack([
            np.ones(len(t)), z
        ])

        rr, _ = nuisance_projection(
            target, Z
        )

        s = float(
            np.linalg.norm(rr) / np.sqrt(len(t))
        )

        if s < best[0]:
            best = (s, float(k))

    return dict(
        affine_fit_RMS_log2CPM=affine,
        exponential_plus_baseline_fit_RMS_log2CPM=best[0],
        exponential_rate_per_hour=best[1],
        observed_dynamic_range_log2CPM=float(np.ptp(y)),
        fitted_shape_is_not_true_biological_mechanism=True
    )


def audit(curves):
    local = []
    pairs = []
    fitrows = []
    design = []

    allpairs = tuple(
        combinations(DELAYS, 2)
    )

    for (culture, gene), (times, expr) in curves.items():
        interpolator = make_interpolator(
            times, expr
        )

        fitrows.append(dict(
            culture=culture,
            gene=gene,
            **fit_degenerate_classes(times, expr)
        ))

        for sched, grid in schedules(times).items():
            t = np.asarray(grid, float)

            hypothetical = [
                float(x) for x in t
                if not np.isclose(times, x).any()
            ]

            m = dict(
                culture=culture,
                gene=gene,
                schedule=sched,
                n_times=len(t),
                times_hours=';'.join(
                    '%g' % x for x in t
                ),
                all_times_measured=(
                    len(hypothetical) == 0
                ),
                hypothetical_times=';'.join(
                    '%g' % x for x in hypothetical
                )
            )

            for tau in (0., 6.):
                for speed in (False, True):
                    local.append(dict(
                        **m,
                        tau_hours=tau,
                        **local_geometry(
                            interpolator, t, tau, speed
                        )
                    ))

            for tau0, tau1 in allpairs:
                for speed in (False, True):
                    forward = profile_direction(
                        interpolator, t, tau0, tau1, speed
                    )

                    reverse = profile_direction(
                        interpolator, t, tau1, tau0, speed
                    )

                    minimum = min(
                        forward['sse'],
                        reverse['sse']
                    )

                    pairs.append(dict(
                        **m,
                        tau_a=tau0,
                        tau_b=tau1,
                        nuisance=(
                            'baseline_amplitude_speedgrid'
                            if speed else
                            'baseline_amplitude'
                        ),
                        sqrtSSE_forward_log2CPM=np.sqrt(
                            forward['sse']
                        ),
                        sqrtSSE_reverse_log2CPM=np.sqrt(
                            reverse['sse']
                        ),
                        minimum_direction_sqrtSSE_log2CPM=(
                            np.sqrt(minimum)
                        ),
                        minimum_direction_RMS_log2CPM=(
                            np.sqrt(minimum / len(t))
                        ),
                        amp_forward=forward['amplitude'],
                        amp_reverse=reverse['amplitude'],
                        speed_forward=forward['speed'],
                        speed_reverse=reverse['speed'],
                        empirical_noise_sigma_log2CPM=np.nan,
                        RMS_over_empirical_sigma=np.nan,
                        interpretation=(
                            'Conditional template geometry, '
                            'not observed delay truth'
                        )
                    ))

        say(
            '[CURVE] {} {} | measured={} hours; '
            '2h missing'.format(
                culture, gene, len(times)
            )
        )

    local = save(
        '04_local_projected_sensitivity.csv',
        local
    )

    pairs = save(
        '05_pairwise_profile_distances.csv',
        pairs
    )

    fitrows = save(
        '06_affine_exponential_shape_fit.csv',
        fitrows
    )

    joint = []

    for (
        culture, sched, ta, tb, nuisance
    ), d in pairs.groupby([
        'culture', 'schedule',
        'tau_a', 'tau_b', 'nuisance'
    ]):
        ensure(
            len(d) == len(GENES),
            'Joint culture missing gene scores'
        )

        # Same shift direction for all eight genes.
        a = float(np.sum(
            d.sqrtSSE_forward_log2CPM.to_numpy()**2
        ))

        b = float(np.sum(
            d.sqrtSSE_reverse_log2CPM.to_numpy()**2
        ))

        sample_count = int(
            d.n_times.sum()
        )

        joint.append(dict(
            culture=culture,
            schedule=sched,
            tau_a=ta,
            tau_b=tb,
            nuisance=nuisance,
            genes=8,
            total_time_gene_observations=sample_count,
            min_direction_joint_sqrtSSE_log2CPM=float(
                np.sqrt(min(a, b))
            ),
            joint_RMS_log2CPM=float(
                np.sqrt(min(a, b) / sample_count)
            ),
            full_covariance_available=False,
            empirical_sigma_available=False,
            interpretation=(
                'Shared-tau eight-gene geometry; '
                'no measured noise or correlation'
            )
        ))

    joint = save(
        '07_culture_shared_delay_pairwise.csv',
        joint
    )

    for (
        culture, sched, model, tau
    ), d in local.groupby([
        'culture', 'schedule',
        'model', 'tau_hours'
    ]):
        ensure(
            len(d) == 8,
            'Local culture information incomplete'
        )

        design.append(dict(
            culture=culture,
            schedule=sched,
            model=model,
            tau_hours=tau,
            independent_gene_unit_sigma_information=float(
                d.local_geometry_squared.sum()
            ),
            per_gene_min_information=float(
                d.local_geometry_squared.min()
            ),
            n_genes=8,
            correlation_known=False,
            empirical_noise_known=False,
            interpretation=(
                'Information additive only with independent '
                'equal-variance gene errors'
            )
        ))

    save(
        '08_shared_vs_gene_specific_local.csv',
        design
    )

    target = joint[
        (joint.tau_a == 0)
        & (joint.tau_b == 6)
        & (joint.nuisance == 'baseline_amplitude')
    ].copy()

    columns = [
        'culture', 'schedule',
        'min_direction_joint_sqrtSSE_log2CPM',
        'joint_RMS_log2CPM',
        'genes'
    ]

    target = target[columns]

    baseline = target[
        target.schedule == 'core_measured'
    ].set_index('culture')

    ref = baseline[
        'min_direction_joint_sqrtSSE_log2CPM'
    ].to_dict()

    target['incremental_sqrtSSE_vs_core_log2CPM'] = (
        target.apply(
            lambda r: float(
                r.min_direction_joint_sqrtSSE_log2CPM
                - ref[r.culture]
            ),
            axis=1
        )
    )

    target['2h_purely_modelled_if_added'] = (
        target.schedule.str.contains('plus_2')
    )

    save(
        '09_0_vs_6_schedule_screen.csv',
        target
    )

    best = joint[
        (joint.schedule == 'core_measured')
        & (joint.tau_a == 0)
        & (joint.tau_b == 6)
        & (joint.nuisance == 'baseline_amplitude')
    ]

    for r in best.itertuples(index=False):
        say(
            '[PROFILE 0v6 | {}] 8 genes measured-grid '
            'RMS={:.5f} log2CPM; sigma UNAVAILABLE'.format(
                r.culture,
                r.joint_RMS_log2CPM
            )
        )

    alias_count = int(
        local[
            (local.schedule == 'core_measured')
            & (local.tau_hours == 0)
            & (local.model == 'baseline_amplitude_speed')
        ].local_numerical_alias.sum()
    )

    say(
        '[LOCAL ALIAS] tau0 core, allowing speed nuisance: '
        '{}/32 near-zero projected sensitivities'.format(
            alias_count
        )
    )

    return local, pairs, joint


def write_methods(source_run):
    txt = r'''
# V103: actual-time identifiability audit

## Measured source and measurement boundary

Input: SHA256-checked V67 frozen_v31_source.csv,
four cultures x eight genes x ten recorded hours each.

The 2h value is absent from all 32 trajectories.

Real replicate-resolved measurement error cannot be
estimated from one value per gene/culture/time in
this frozen snapshot.

No RMS-to-SD ratio, statistical power, Bayes error
or practical-identifiability certificate is reported.

## Theory and qualifications

For every shift in a neighborhood of zero, a continuously
differentiable nonvanishing function satisfying
f(t-h)=c(h)f(t) must have exponential form under
regularity assumptions.

With a free baseline, f(t-h)=c(h)f(t)+d(h)
admits exponential-plus-baseline or affine responses.

These are not the only solutions for one fixed shift
or for a finite observed time grid.

An exponential multiplied by a 6h-periodic function
also aliases under a fixed 6h shift.

An unknown additional onset phase can confound tau.
A response with zero amplitude also aliases.

The V96/V100 linear 2h value is interpolation of
one coordinate, not a globally linear biological curve.

## Conditional measured-template geometry

Each gene/culture trajectory is represented by a PCHIP
interpolator through ten measured timepoints.

Candidate delays are represented using a causal
flat-pre-onset response q(max(t-tau,0)).

This is an additional modelling assumption, not
an empirically proven kinetic law.

The pairwise calculation profiles unknown baseline
and nonnegative amplitude.

A second model profiles a small prespecified
time-scale grid from 0.70 to 1.30.

Both shift directions are reported.

Distances are descriptive, not hypothesis-test p-values
and not globally optimized over all possible kinetics.

Local time-delay sensitivity is projected away from
baseline/amplitude and optionally time-scale directions.

The resulting information-like quantity has not been
divided by measured noise variance, so it is NOT
empirical Fisher information.

For shared culture-level tau and independent equal
variance errors, information contributions from
eight genes can be added.

If each gene has its own tau, the interpretation differs.

Correlated gene errors require an estimated covariance.

## Mathematical 2h ambiguity

Let a,b bracket the unmeasured 2h value.

Inside (a,b), construct:

h(t) = A sin^2(pi*(t-a)/(b-a)).

Outside that interval, set h(t)=0.

The curve and first derivative match at the endpoints.
Every measured point remains unchanged.

By choosing A, the unobserved 2h expression can be
changed by an arbitrary specified amount.

This is a mathematical C1 ambiguity witness, not
a biologically justified waveform or curvature bound.

## Interpretation

File 01: observed measurement times.
File 02: missing empirical noise.
File 03: exact 2h smooth ambiguity witnesses.
File 04: projected local sensitivity.
File 05: pairwise template distances.
File 06: proximity to affine/exponential model classes.
File 07: common-tau eight-gene distances.
File 08: shared versus gene-specific tau.
File 09: hypothetical acquisition designs.

Hypothetical 2h values in file 09 come from interpolation;
they are not additional observed biological evidence.

Practical delay detectability remains NOT EVALUABLE
without appropriate replicate data and a defended
biological response family.

V67-V102 historical findings remain unchanged.
'''

    (
        OUT / '10_assumptions_and_theory.md'
    ).write_text(txt, encoding='utf-8')

    shutil.copy2(
        source_run / '08_manifest.json',
        OUT / 'input_v67_verified_manifest.json'
    )

    shutil.copy2(
        source_run / 'frozen_v31_source.csv',
        OUT / 'input_v67_measured_source.csv'
    )

    shutil.copy2(
        Path(__file__),
        OUT / 'script_snapshot_v103.py'
    )


def main():
    say('=' * 94)
    say(
        'PAPER 7 | V103 | MEASURED GRID / '
        'LOCAL AND DISCRETE-SHIFT IDENTIFIABILITY'
    )
    say(
        'OFFLINE | REAL SOURCE KNOTS | '
        'NO DELAY TRUTH | NO FABRICATED NOISE | NO GIT'
    )
    say('=' * 94)

    ensure(
        len(sys.argv) == 1,
        'Run without arguments'
    )

    run, manifest = find_archive()

    config = dict(
        version='v103',
        source_sha256=digest(
            run / 'frozen_v31_source.csv'
        ),
        eight_genes=list(GENES),
        four_cultures=list(CULTURES),
        delays=list(DELAYS),
        source_hours_actual=(
            'Read from V67 for each gene/culture'
        ),
        designs={
            'core_requested_hours': CORE,
            'added_times': ADD
        },
        time_shift=(
            'Additive tau, not temporal rescaling'
        ),
        model=(
            'PCHIP through measured knots, '
            'causal flat pre-onset assumption'
        ),
        nuisance=(
            'Free baseline, nonnegative amplitude, '
            'optional fixed-grid speed'
        ),
        empirical_noise=(
            'NOT_ESTIMABLE_FROM_FROZEN_SOURCE'
        ),
        no_new_data=True,
        no_git=True,
        no_network=True,
        code_sha256=digest(Path(__file__))
    )

    (
        OUT / '00_protocol_before_analysis.json'
    ).write_text(
        json.dumps(config, indent=2) + '\n',
        encoding='utf-8'
    )

    verify_algebraic_aliases()

    curves = load_curves(run)
    local, pairs, joint = audit(curves)

    write_methods(run)

    (
        OUT / '11_manifest.json'
    ).write_text(
        json.dumps(dict(
            version='v103',
            code_sha256=digest(Path(__file__)),
            v67_source_sha256=digest(
                run / 'frozen_v31_source.csv'
            ),
            n_curves=len(curves),
            n_local_rows=len(local),
            n_pairwise_rows=len(pairs),
            n_joint_rows=len(joint),
            noise_sd_available=False,
            mathematical_nonidentifiability_examples_proven=True,
            no_empirical_biological_delay_validation=True,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            no_git=True,
            no_network=True
        ), indent=2) + '\n',
        encoding='utf-8'
    )

    say(
        '[GATE] Structural degeneracy examples = analytic; '
        'local template geometry = conditional; '
        'practical detectability = NOT EVALUABLE'
    )

    say(
        '[COMPLETE] Measured-grid identifiability '
        'geometry and witness archive complete'
    )


def finish(success):
    say(
        '[RESULT] ' + (
            'SUCCESS' if success else 'FAILED'
        )
    )

    say(
        '[REPORT] ' + str(
            OUT / 'console_summary.txt'
        )
    )

    archive = OUT / 'portable_report_v103.zip'
    say('[ZIP] ' + str(archive))

    (
        OUT / 'console_summary.txt'
    ).write_text(
        '\n'.join(LOG) + '\n',
        encoding='utf-8'
    )

    files = []

    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '12_sha256_inventory.csv',
            archive.name
        ):
            files.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(files).to_csv(
        OUT / '12_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive, 'w',
        zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)


if __name__ == '__main__':
    success = True

    try:
        main()
    except Exception:
        success = False
        say('[ERROR]\n' + traceback.format_exc())
    finally:
        finish(success)

    if not success:
        sys.exit(1)
