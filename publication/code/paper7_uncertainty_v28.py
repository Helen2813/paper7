#!/usr/bin/env python3
"""Paper 7 v28: reference-ensemble delay assessment.
Read-only, offline, no flags.
Synthetic calibration is exploratory, not clinical inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_reference_ensemble_v28" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
TIMES = np.array([0., 2., 4., 6., 8.])
DELAYS = np.arange(0., 2.00001, .25)

N_CAL = 250
N_EVAL = 250
SEED = 20271030
STABILITY_GATE = .50

# Precompute interpolation operators.
BASIS = np.eye(len(TIMES))
WEIGHTS = np.array([
    [
        np.interp(t - delay, TIMES, BASIS[:, j], left=0.)
        for j in range(len(TIMES))
    ]
    for delay in DELAYS
    for t in TIMES[1:]
]).reshape(len(DELAYS), 4, len(TIMES))


def say(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def save(name, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(
        k for r in rows for k in r
    )) or ["note"]

    with (OUT / name).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    say(f"[OUTPUT] {name} | rows={len(rows)}")


def latest_v25():
    base = ROOT / "paper7_panel_delay_v25"

    if not base.is_dir():
        raise FileNotFoundError(str(base))

    required = (
        "05_panel_unit_trajectories.csv",
        "09_model_comparison_summary.csv",
        "11_manifest.json",
    )

    for folder in sorted(
        base.glob("run_*"), reverse=True
    ):
        if all(
            (folder / f).is_file()
            for f in required
        ):
            return folder

    raise FileNotFoundError(
        "No complete v25 run"
    )


def load_data(path):
    df = pd.read_csv(path)

    required = {
        "panel", "group", "replicate_label",
        "day", "score"
    }

    if (
        not required.issubset(df.columns)
        or df[list(required)].isna().any().any()
    ):
        raise ValueError("Incomplete v25 score table")

    if df.duplicated([
        "panel", "group", "replicate_label", "day"
    ]).any():
        raise ValueError("Duplicate unit/day")

    if len(df) != 80 or df.panel.nunique() != 2:
        raise ValueError(
            "Expected two panels, 80 rows"
        )

    result = {}

    for panel in sorted(df.panel.unique()):
        traces = {}
        labels = {"N": [], "E": []}

        part = df[df.panel == panel]

        for (group, rep), unit in part.groupby([
            "group", "replicate_label"
        ]):
            if group not in labels:
                raise ValueError(
                    "Unknown group " + str(group)
                )

            unit = unit.sort_values("day")

            if unit.day.tolist() != [
                0, 2, 4, 6, 8
            ]:
                raise ValueError(
                    "Incomplete time course"
                )

            values = unit.score.to_numpy(float)

            if (
                not np.isfinite(values).all()
                or abs(values[0]) > 1e-7
            ):
                raise ValueError(
                    "Invalid baseline or score"
                )

            key = f"{group}:{int(rep)}"
            traces[key] = values
            labels[group].append(key)

        if (
            len(labels["N"]) != 4
            or len(labels["E"]) != 4
        ):
            raise ValueError(
                "Expected 4 N and 4 E units"
            )

        result[panel] = (
            traces,
            sorted(labels["N"]),
            sorted(labels["E"])
        )

    return result


def fit_reference(reference, target):
    """Fit amplitude-only and amplitude+delay.
    Uses training target only.
    """
    shapes = np.einsum(
        "dtk,k->dt", WEIGHTS, reference
    )

    norms = np.einsum(
        "dt,dt->d", shapes, shapes
    )

    numerator = shapes @ target[1:]

    amps = np.divide(
        numerator,
        norms,
        out=np.zeros_like(numerator),
        where=norms > 1e-12
    )

    amps = np.maximum(0., amps)

    predictions = shapes * amps[:, None]

    losses = np.sum(
        (predictions - target[1:][None, :]) ** 2,
        axis=1
    )

    winner = int(np.argmin(losses))

    if losses[winner] > losses[0] + 1e-9:
        raise RuntimeError(
            "Nested objective contradiction"
        )

    return (
        predictions[0].copy(),
        predictions[winner].copy(),
        float(DELAYS[winner])
    )


def compare(traces, controls, cases, detailed=False):
    """16 overlapping double-holdout folds.

    For each fold:
    - Train on 3 N and 3 E units.
    - Construct 4 training-only reference curves.
    - Average predictions separately by model.
    - Evaluate on excluded E trajectory.
    """

    rows = []
    ref_rows = []

    totals = {
        "fixed_A": 0.,
        "fixed_D": 0.,
        "ensemble_A": 0.,
        "ensemble_D": 0.,
    }

    stable = 0
    wins = 0

    for excluded_n in controls:
        train_n = [
            n for n in controls
            if n != excluded_n
        ]

        training_n = np.array([
            traces[n] for n in train_n
        ])

        # Reference 1: mean of 3 training controls.
        references = [
            training_n.mean(axis=0)
        ]

        # References 2-4:
        # all leave-one-out means of 2 controls.
        references.extend(
            np.delete(
                training_n, j, axis=0
            ).mean(axis=0)
            for j in range(3)
        )

        for excluded_e in cases:
            train_e = [
                e for e in cases
                if e != excluded_e
            ]

            target = np.mean([
                traces[e] for e in train_e
            ], axis=0)

            observed = traces[excluded_e][1:]

            amp_predictions = []
            delay_predictions = []
            fitted_delays = []

            for k, reference in enumerate(references):
                amp, delayed, delay = fit_reference(
                    reference, target
                )

                amp_predictions.append(amp)
                delay_predictions.append(delayed)
                fitted_delays.append(delay)

                if detailed:
                    ref_rows.append({
                        "excluded_N": excluded_n,
                        "excluded_E": excluded_e,
                        "reference": (
                            "mean_3N" if k == 0
                            else f"mean_2N_{k}"
                        ),
                        "training_delay_days": delay,
                        "test_amp_SSE": float(
                            np.sum((observed - amp) ** 2)
                        ),
                        "test_delay_SSE": float(
                            np.sum((observed - delayed) ** 2)
                        ),
                    })

            fixed_a = float(np.sum(
                (observed - amp_predictions[0]) ** 2
            ))

            fixed_d = float(np.sum(
                (observed - delay_predictions[0]) ** 2
            ))

            ensemble_a = np.mean(
                amp_predictions, axis=0
            )

            ensemble_d = np.mean(
                delay_predictions, axis=0
            )

            ens_a = float(np.sum(
                (observed - ensemble_a) ** 2
            ))

            ens_d = float(np.sum(
                (observed - ensemble_d) ** 2
            ))

            delays = np.array(fitted_delays)

            # Descriptive consensus rule.
            stable_fold = bool(
                np.count_nonzero(delays >= .25) >= 3
                and np.ptp(delays) <= .5
            )

            stable += int(stable_fold)
            wins += int(ens_d < ens_a - 1e-9)

            totals["fixed_A"] += fixed_a
            totals["fixed_D"] += fixed_d
            totals["ensemble_A"] += ens_a
            totals["ensemble_D"] += ens_d

            if detailed:
                rows.append({
                    "excluded_N": excluded_n,
                    "excluded_E": excluded_e,
                    "fixed_amp_SSE": fixed_a,
                    "fixed_delay_SSE": fixed_d,
                    "ensemble_amp_SSE": ens_a,
                    "ensemble_delay_SSE": ens_d,
                    "ensemble_gain": ens_a - ens_d,
                    "delays": ";".join(
                        f"{x:.2f}" for x in delays
                    ),
                    "consensus_stable": stable_fold,
                })

    gain = (
        100 * (
            totals["ensemble_A"]
            - totals["ensemble_D"]
        ) / max(totals["ensemble_A"], 1e-12)
    )

    result = {
        "fixed_amp_SSE": totals["fixed_A"],
        "fixed_delay_SSE": totals["fixed_D"],
        "ensemble_amp_SSE": totals["ensemble_A"],
        "ensemble_delay_SSE": totals["ensemble_D"],
        "ensemble_gain_percent": gain,
        "stable_fold_fraction": stable / 16.,
        "ensemble_delay_fold_wins": wins,
    }

    return result, rows, ref_rows


def wilson_interval(successes, total):
    """Descriptive Monte Carlo binomial interval."""
    z = 1.95996398454
    p = successes / total
    q = z*z / total

    center = (p + q / 2) / (1 + q)

    radius = (
        z * math.sqrt(
            p*(1-p)/total
            + z*z/(4*total*total)
        ) / (1+q)
    )

    return (
        max(0., center-radius),
        min(1., center+radius)
    )


def simulate(rng, baseline, residuals, sd, truth):
    """Create one independent simulated 4N+4E experiment.

    Residuals are drawn from observed controls.
    This is a scenario generator, NOT a model of
    real patient sampling.
    """

    traces = {}
    extra_noise = .15 * sd

    delayed_truth = np.interp(
        TIMES - 1.,
        TIMES,
        baseline,
        left=0.
    )

    # No explicit shift; different response shape.
    shape_change = (
        .65
        * max(float(np.max(np.abs(baseline))), .1)
        * np.sin(np.pi * TIMES / 8.)
    )

    for group in ("N", "E"):
        for j in range(4):

            if group == "N":
                signal = baseline

            elif truth == "delay_one_day":
                signal = .85 * delayed_truth

            elif truth == "null_shape_changed":
                signal = (
                    .85 * baseline + shape_change
                )

            else:
                signal = .85 * baseline

            residual = residuals[
                int(rng.integers(0, 4))
            ]

            noise = rng.normal(
                0., extra_noise, size=5
            )

            curve = signal + residual + noise
            curve[0] = 0.

            traces[f"{group}:{j}"] = curve

    return traces


def make_svg(results):
    """Compact graph with no matplotlib dependency."""
    values = [
        r["ensemble_gain_percent"]
        for r in results
    ]

    low = min(-25., min(values) - 5.)
    high = max(15., max(values) + 5.)

    def Y(value):
        return (
            285
            - 210 * (value - low) / (high - low)
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 900 390">',
        '<rect width="900" height="390" fill="white"/>',
        '<text x="32" y="35" font-family="Arial" '
        'font-size="21" font-weight="bold">'
        'Paper 7 v28 | Reference ensemble</text>',
        '<text x="32" y="56" font-family="Arial" '
        'font-size="12">'
        'Held-out SSE improvement from adding delay'
        '</text>',
    ]

    for tick in (-20, -10, 0, 10):
        if low <= tick <= high:
            y = Y(tick)

            parts.append(
                f'<line x1="80" x2="850" '
                f'y1="{y:.1f}" y2="{y:.1f}" '
                'stroke="#e5e7eb"/>'
            )

            parts.append(
                f'<text x="68" y="{y+4:.1f}" '
                'text-anchor="end" font-family="Arial" '
                f'font-size="11">{tick}%</text>'
            )

    colors = ("#2463aa", "#c05a32")

    for i, (row, color) in enumerate(
        zip(results, colors)
    ):
        x = 170 + i * 350
        val = row["ensemble_gain_percent"]

        zero = Y(0.)
        y = Y(val)

        parts.append(
            f'<rect x="{x}" '
            f'y="{min(zero,y):.1f}" '
            f'width="105" '
            f'height="{abs(y-zero):.1f}" '
            f'fill="{color}"/>'
        )

        parts.append(
            f'<text x="{x+53}" '
            f'y="{max(zero,y)+18:.1f}" '
            'text-anchor="middle" '
            'font-family="Arial" font-size="13">'
            f'{val:+.2f}%</text>'
        )

        label = (
            "Eight-gene panel"
            if "all8" in row["panel"]
            else "Six-gene panel"
        )

        parts.append(
            f'<text x="{x+53}" y="333" '
            'font-family="Arial" font-size="12" '
            f'text-anchor="middle">{label}</text>'
        )

    parts.append(
        '<text x="32" y="371" '
        'font-family="Arial" font-size="11">'
        'Overlapping folds; descriptive comparison only.'
        '</text>'
    )

    parts.append("</svg>")

    (OUT / "08_real_comparison.svg").write_text(
        "\n".join(parts),
        encoding="utf-8"
    )

    say("[FIGURE] 08_real_comparison.svg")


def main():
    say(
        "PAPER 7 | V28 | "
        "REFERENCE-ENSEMBLE DELAY ASSESSMENT"
    )

    say("[ROOT] " + str(ROOT))
    say("[OUTPUT] " + str(OUT))

    say(
        "[POLICY] No feature re-selection, "
        "no GEO downloads, no donor-level p-values."
    )

    prior = latest_v25()

    say("[SOURCE V25] " + str(prior))

    inputs = [
        prior / "05_panel_unit_trajectories.csv",
        prior / "09_model_comparison_summary.csv",
        prior / "11_manifest.json",
    ]

    save("00_source_manifest.csv", [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": sha(p),
        }
        for p in inputs
    ])

    datasets = load_data(inputs[0])

    previous = pd.read_csv(inputs[1])

    real_results = []
    all_folds = []
    all_references = []

    for panel, (
        traces, controls, cases
    ) in datasets.items():

        result, folds, refs = compare(
            traces, controls, cases,
            detailed=True
        )

        result["panel"] = panel
        real_results.append(result)

        for row in folds:
            row["panel"] = panel

        for row in refs:
            row["panel"] = panel

        all_folds.extend(folds)
        all_references.extend(refs)

        old = previous[
            previous.panel == panel
        ]

        if len(old) != 1:
            raise RuntimeError(
                "V25 panel summary missing"
            )

        parity = (
            np.isclose(
                result["fixed_amp_SSE"],
                float(
                    old.iloc[0]["amp_heldout_sse"]
                ),
                atol=1e-6, rtol=0
            )
            and np.isclose(
                result["fixed_delay_SSE"],
                float(
                    old.iloc[0][
                        "amp_delay_heldout_sse"
                    ]
                ),
                atol=1e-6, rtol=0
            )
        )

        if not parity:
            raise RuntimeError(
                "V25 PARITY FAILED: " + panel
            )

        say(
            "[PARITY PASS] {} | "
            "fixed A={:.6f} | "
            "fixed A+D={:.6f}".format(
                panel,
                result["fixed_amp_SSE"],
                result["fixed_delay_SSE"]
            )
        )

        say(
            "[REAL ENSEMBLE] {} | "
            "A={:.6f} | A+D={:.6f} | "
            "gain={:+.2f}% | "
            "stable folds={:.2f} | "
            "delay wins={}/16".format(
                panel,
                result["ensemble_amp_SSE"],
                result["ensemble_delay_SSE"],
                result["ensemble_gain_percent"],
                result["stable_fold_fraction"],
                result["ensemble_delay_fold_wins"]
            )
        )

    save(
        "01_real_double_holdout_folds.csv",
        all_folds
    )

    save(
        "02_real_training_reference_variants.csv",
        all_references
    )

    save(
        "03_real_panel_summary.csv",
        real_results
    )

    # The simulator uses the primary eight-gene panel.
    primary = next(
        (
            name for name in datasets
            if "all8" in name
        ),
        None
    )

    if primary is None:
        raise RuntimeError(
            "Eight-gene panel not found"
        )

    traces, controls, cases = datasets[primary]

    N = np.stack([
        traces[key] for key in controls
    ])

    baseline = N.mean(axis=0)
    residuals = N - baseline
    sd = N.std(axis=0, ddof=1)

    say(
        "[SIMULATOR] Control-derived "
        "reference from four observed N curves."
    )

    say(
        "[SIMULATOR] Independent experiments: "
        f"calibration={N_CAL}; "
        f"evaluation={N_EVAL} per condition."
    )

    synthetic_n = [
        f"N:{j}" for j in range(4)
    ]

    synthetic_e = [
        f"E:{j}" for j in range(4)
    ]

    # Independent calibration simulations.
    rng = np.random.default_rng(SEED + 1)

    calibration = []

    for trial in range(N_CAL):
        data = simulate(
            rng, baseline, residuals, sd,
            "null_same_shape"
        )

        result, _, _ = compare(
            data, synthetic_n, synthetic_e
        )

        calibration.append({
            "trial": trial,
            "gain_percent": result[
                "ensemble_gain_percent"
            ],
            "stable_fold_fraction": result[
                "stable_fold_fraction"
            ]
        })

    threshold = float(
        np.quantile(
            [
                row["gain_percent"]
                for row in calibration
            ],
            .95,
            method="higher"
        )
    )

    say(
        "[INDEPENDENT NULL CALIBRATION] "
        f"95th percentile gain={threshold:+.4f}%"
    )

    save(
        "04_synthetic_null_calibration_trials.csv",
        calibration
    )

    evaluations = []
    gate_summaries = []

    conditions = [
        ("null_same_shape", SEED + 2),
        ("delay_one_day", SEED + 3),
        ("null_shape_changed", SEED + 4),
    ]

    for truth, seed in conditions:
        rng = np.random.default_rng(seed)

        positives = 0
        gains = []

        for trial in range(N_EVAL):
            data = simulate(
                rng, baseline, residuals, sd,
                truth
            )

            result, _, _ = compare(
                data, synthetic_n, synthetic_e
            )

            gain = result[
                "ensemble_gain_percent"
            ]

            stable = result[
                "stable_fold_fraction"
            ]

            # Both conditions must hold.
            decision = bool(
                gain > max(0., threshold)
                and stable >= STABILITY_GATE
            )

            positives += int(decision)
            gains.append(gain)

            evaluations.append({
                "truth": truth,
                "trial": trial,
                "gain_percent": gain,
                "stable_fold_fraction": stable,
                "passes_prototype_gate": decision,
            })

        low, high = wilson_interval(
            positives, N_EVAL
        )

        gate_summaries.append({
            "truth": truth,
            "n_experiments": N_EVAL,
            "positive_decisions": positives,
            "fraction_positive": (
                positives / N_EVAL
            ),
            "conditional_wilson_95_low": low,
            "conditional_wilson_95_high": high,
            "median_gain_percent": float(
                np.median(gains)
            ),
        })

        say(
            "[SIMULATED EVALUATION] {} | "
            "{}/{} ({:.1%}) | "
            "conditional Wilson 95% "
            "[{:.1%}, {:.1%}]".format(
                truth,
                positives, N_EVAL,
                positives / N_EVAL,
                low, high
            )
        )

    save(
        "05_synthetic_evaluation_trials.csv",
        evaluations
    )

    save(
        "06_synthetic_gate_summary.csv",
        gate_summaries
    )

    for row in real_results:
        row["synthetic_calibration_cutoff"] = threshold

        # Illustrative only, NOT empirical significance.
        row["illustrative_real_gate_pass"] = bool(
            row["ensemble_gain_percent"]
            > max(0., threshold)
            and row["stable_fold_fraction"]
            >= STABILITY_GATE
        )

        say(
            "[REAL ILLUSTRATIVE GATE] {} | {} "
            "(NOT donor-level inference)".format(
                row["panel"],
                row["illustrative_real_gate_pass"]
            )
        )

    save(
        "07_real_gate_context.csv",
        real_results
    )

    make_svg(real_results)

    notes = [
        "# Paper 7 v28 — Reference-ensemble method",
        "",
        "## Method",
        "Both previously selected gene panels were retained.",
        "",
        "Each double-holdout fold excluded one "
        "control and one endometriosis unit.",
        "",
        "Three remaining controls generated four "
        "reference curves: the three-control mean "
        "and three two-control means.",
        "",
        "Both competing response models were fitted "
        "on each reference separately.",
        "",
        "Within each model, predictions across four "
        "reference curves were averaged.",
        "",
        "Predictive performance was evaluated on "
        "the excluded endometriosis trajectory.",
        "",
        "The stability indicator requires at least "
        "three of four nonzero delay estimates and "
        "a range no greater than 0.5 days.",
        "",
        "## Real data",
    ]

    for row in real_results:
        notes.append(
            "- {}: fixed A/A+D SSE {:.3f}/{:.3f}; "
            "ensemble A/A+D SSE {:.3f}/{:.3f}; "
            "ensemble gain {:+.2f}%; "
            "stable folds {:.1%}.".format(
                row["panel"],
                row["fixed_amp_SSE"],
                row["fixed_delay_SSE"],
                row["ensemble_amp_SSE"],
                row["ensemble_delay_SSE"],
                row["ensemble_gain_percent"],
                row["stable_fold_fraction"]
            )
        )

    notes.extend([
        "",
        "## Simulation experiment",
        "Independent synthetic null calibration "
        f"used {N_CAL} experiments.",
        "",
        f"Null gain threshold: {threshold:+.4f}%.",
        "",
    ])

    for row in gate_summaries:
        notes.append(
            "- {}: {}/{} positive decisions ({:.1%}).".format(
                row["truth"],
                row["positive_decisions"],
                N_EVAL,
                row["fraction_positive"]
            )
        )

    notes.extend([
        "",
        "## Limitations",
        "This is an exploratory methodological prototype.",
        "",
        "The reference estimates share training controls "
        "and do not constitute an independent bootstrap "
        "distribution or confidence interval.",
        "",
        "The 16 folds reuse the same biological units.",
        "",
        "Synthetic calibration is conditional on "
        "a simulator informed by four observed controls.",
        "",
        "The shape-perturbed no-delay scenario "
        "tests violation of calibration assumptions.",
        "",
        "Reported Monte Carlo intervals exclude "
        "uncertainty in the calibration threshold.",
        "",
        "No clinical inference or validated "
        "patient-specific delay claim is made.",
        "",
        "Comparison with established functional "
        "registration methods is still required.",
    ])

    (OUT / "09_methods_results_draft.md").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8"
    )

    say("[OUTPUT] 09_methods_results_draft.md")

    manifest = {
        "version": "v28",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_v25": str(prior.relative_to(ROOT)),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in inputs
        },
        "delay_grid_days": DELAYS.tolist(),
        "reference_ensemble": (
            "mean_3N plus three mean_2N references"
        ),
        "calibration_trials": N_CAL,
        "evaluation_trials_per_condition": N_EVAL,
        "calibration_seed": SEED + 1,
        "evaluation_seeds": {
            name: seed
            for name, seed in conditions
        },
        "stability_gate": STABILITY_GATE,
        "null_cutoff": threshold,
        "script_sha256": sha(Path(__file__)),
        "interpretation": (
            "Exploratory conditional synthetic calibration; "
            "no donor-level significance"
        ),
    }

    (OUT / "10_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT / "script_snapshot.py"
    )

    say(
        "[COMPLETE] Results, manuscript notes, "
        "source hashes and code saved."
    )


if __name__ == "__main__":
    failed = False

    try:
        main()

    except Exception:
        failed = True
        say(
            "[ERROR]\n" + traceback.format_exc()
        )

    finally:
        say(
            "[REPORT] " +
            str(OUT / "console_summary.txt")
        )

        (OUT / "console_summary.txt").write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8"
        )

        archive = OUT / "portable_report_v28.zip"

        with zipfile.ZipFile(
            archive, "w", zipfile.ZIP_DEFLATED
        ) as z:
            for path in sorted(OUT.rglob("*")):
                if (
                    path.is_file()
                    and path != archive
                ):
                    z.write(
                        path,
                        path.relative_to(OUT)
                    )

        say("[ZIP] " + str(archive))

    if failed:
        sys.exit(1)