#!/usr/bin/env python3
"""
PAPER 7 | V29
PAIRED BASELINE BENCHMARK

No new gene selection.
No external downloads.
No population-level or clinical inference.
Checks exact parity with v25 and v28.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

OUT = ROOT / "paper7_benchmark_v29" / (
    "run_" +
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
T = np.array([0., 2., 4., 6., 8.])
DELAYS = np.arange(0., 2.000001, .25)

SEED = 20271030
N_CAL = 250
N_EVAL = 250

BASIS = np.eye(5)

W = np.array([
    [
        [
            np.interp(
                t - d, T, BASIS[:, j], left=0.
            )
            for j in range(5)
        ]
        for t in T[1:]
    ]
    for d in DELAYS
])


def log(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)


def hashfile(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def save(name, rows):
    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig"
    )

    log(
        f"[OUTPUT] {name} | rows={len(df)}"
    )


def latest(folder, required):
    base = ROOT / folder

    for candidate in sorted(
        base.glob("run_*"), reverse=True
    ):
        if all(
            (candidate / name).is_file()
            for name in required
        ):
            return candidate

    raise RuntimeError(
        f"Missing source run: {folder}"
    )


def read_units(path):
    df = pd.read_csv(path)

    if len(df) != 80 or df.panel.nunique() != 2:
        raise ValueError(
            "Unexpected panel structure"
        )

    if df.duplicated([
        "panel", "group",
        "replicate_label", "day"
    ]).any():
        raise ValueError("Duplicate trajectories")

    result = {}

    for panel, part in df.groupby("panel"):
        traces = {}
        ids = {"N": [], "E": []}

        for (group, rep), sub in part.groupby([
            "group", "replicate_label"
        ]):
            if group not in ids:
                raise ValueError(
                    "Unknown group"
                )

            sub = sub.sort_values("day")

            if sub.day.tolist() != [0, 2, 4, 6, 8]:
                raise ValueError(
                    "Incomplete time course"
                )

            curve = sub.score.to_numpy(float)

            if (
                not np.isfinite(curve).all()
                or abs(curve[0]) > 1e-7
            ):
                raise ValueError(
                    "Invalid trajectory"
                )

            key = f"{group}:{int(rep)}"

            ids[group].append(key)
            traces[key] = curve

        if (
            len(ids["N"]) != 4
            or len(ids["E"]) != 4
        ):
            raise ValueError(
                "Expected four units per group"
            )

        result[panel] = (
            traces,
            sorted(ids["N"]),
            sorted(ids["E"])
        )

    return result


def fit(reference, target):
    shapes = np.einsum(
        "dtk,k->dt", W, reference
    )

    norms = np.einsum(
        "dt,dt->d", shapes, shapes
    )

    amps = np.maximum(
        0.,
        np.divide(
            shapes @ target[1:],
            norms,
            out=np.zeros(len(DELAYS)),
            where=norms > 1e-12
        )
    )

    fitted = shapes * amps[:, None]

    losses = np.sum(
        (fitted - target[1:]) ** 2,
        axis=1
    )

    best = int(np.argmin(losses))

    return (
        fitted[0],
        fitted[best],
        float(DELAYS[best])
    )


def evaluate(traces, controls, cases):
    totals = {
        "fixedA": 0.,
        "fixedD": 0.,
        "ensembleA": 0.,
        "ensembleD": 0.,
    }

    stable = 0

    for heldout_n in controls:
        training_n = np.stack([
            traces[key]
            for key in controls
            if key != heldout_n
        ])

        references = [
            training_n.mean(axis=0)
        ]

        references.extend(
            np.delete(
                training_n, j, axis=0
            ).mean(axis=0)
            for j in range(3)
        )

        for heldout_e in cases:
            train_target = np.stack([
                traces[key]
                for key in cases
                if key != heldout_e
            ]).mean(axis=0)

            observed = traces[heldout_e][1:]

            fitted = [
                fit(ref, train_target)
                for ref in references
            ]

            amp = np.stack([
                item[0] for item in fitted
            ])

            delayed = np.stack([
                item[1] for item in fitted
            ])

            delays = np.array([
                item[2] for item in fitted
            ])

            totals["fixedA"] += float(
                ((observed - amp[0]) ** 2).sum()
            )

            totals["fixedD"] += float(
                ((observed - delayed[0]) ** 2).sum()
            )

            totals["ensembleA"] += float(
                (
                    (observed - amp.mean(axis=0)) ** 2
                ).sum()
            )

            totals["ensembleD"] += float(
                (
                    (observed - delayed.mean(axis=0)) ** 2
                ).sum()
            )

            stable += int(
                np.count_nonzero(delays >= .25) >= 3
                and np.ptp(delays) <= .5
            )

    fixed_gain = (
        100
        * (
            totals["fixedA"] - totals["fixedD"]
        )
        / max(totals["fixedA"], 1e-12)
    )

    ensemble_gain = (
        100
        * (
            totals["ensembleA"] - totals["ensembleD"]
        )
        / max(totals["ensembleA"], 1e-12)
    )

    return {
        **totals,
        "fixed_gain": fixed_gain,
        "ensemble_gain": ensemble_gain,
        "stable_frac": stable / 16,
    }


def simulate(rng, mean, residuals, sd, kind):
    traces = {}

    delayed_truth = np.interp(
        T - 1., T, mean, left=0.
    )

    shape_change = (
        .65
        * max(float(np.max(np.abs(mean))), .1)
        * np.sin(np.pi * T / 8.)
    )

    for group in ("N", "E"):
        for j in range(4):

            if group == "N":
                signal = mean

            elif kind == "delay_one_day":
                signal = .85 * delayed_truth

            elif kind == "null_shape_changed":
                signal = .85 * mean + shape_change

            elif kind == "delay_plus_shape_changed":
                signal = (
                    .85 * delayed_truth + shape_change
                )

            else:
                signal = .85 * mean

            noise = (
                residuals[int(rng.integers(0, 4))]
                + rng.normal(
                    0., .15 * sd, size=5
                )
            )

            curve = (signal + noise).copy()
            curve[0] = 0.

            traces[f"{group}:{j}"] = curve

    return traces


def wilson(successes, total):
    z = 1.95996398454
    p = successes / total
    denominator = 1 + z * z / total

    center = (
        p + z * z / (2 * total)
    ) / denominator

    radius = (
        z * math.sqrt(
            (
                p * (1 - p)
                + z * z / (4 * total)
            ) / total
        )
        / denominator
    )

    return (
        max(0., center - radius),
        min(1., center + radius)
    )


def save_svg(summary):
    scenarios = [
        "null_same_shape",
        "delay_one_day",
        "null_shape_changed",
        "delay_plus_shape_changed",
    ]

    methods = [
        "fixed",
        "ensemble",
        "ensemble_with_stability",
    ]

    labels = {
        "null_same_shape": "No delay",
        "delay_one_day": "1-day delay",
        "null_shape_changed": "Shape change",
        "delay_plus_shape_changed": "Delay + shape",
    }

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1020 410">',
        '<rect width="1020" height="410" fill="white"/>',
        '<text x="28" y="36" font-size="22" '
        'font-family="Arial" font-weight="bold">'
        'Paper 7 v29 | Conditional simulation decisions'
        '</text>',
    ]

    for i, scenario in enumerate(scenarios):
        x = 230 + 190 * i

        svg.append(
            f'<text x="{x}" y="95" '
            'font-family="Arial" font-size="12" '
            'text-anchor="middle">'
            f'{labels[scenario]}</text>'
        )

        for j, method in enumerate(methods):
            row = next(
                r for r in summary
                if r["scenario"] == scenario
                and r["method"] == method
            )

            fraction = row["fraction"]
            y = 118 + j * 78

            color = (
                "#e0f2fe" if fraction < .05
                else "#bae6fd" if fraction < .20
                else "#7dd3fc" if fraction < .50
                else "#0ea5e9"
            )

            svg.append(
                f'<rect x="{x-64}" y="{y}" '
                'width="128" height="57" rx="6" '
                f'fill="{color}" stroke="#94a3b8"/>'
            )

            svg.append(
                f'<text x="{x}" y="{y+34}" '
                'font-family="Arial" font-weight="bold" '
                'font-size="19" text-anchor="middle">'
                f'{100*fraction:.1f}%</text>'
            )

    for j, label in enumerate([
        "Single reference",
        "Reference ensemble",
        "Ensemble + stability",
    ]):
        svg.append(
            f'<text x="24" y="{152+j*78}" '
            'font-family="Arial" font-size="13">'
            f'{label}</text>'
        )

    svg.append(
        '<text x="28" y="384" '
        'font-family="Arial" font-size="12">'
        'Synthetic decision frequencies only.'
        '</text>'
    )

    svg.append("</svg>")

    (OUT / "05_simulation_comparison.svg").write_text(
        "\n".join(svg),
        encoding="utf-8"
    )

    log("[OUTPUT] 05_simulation_comparison.svg")


def main():
    log(
        "PAPER 7 | V29 | "
        "PAIRED BASELINE BENCHMARK"
    )

    log(
        "[POLICY] Synthetic calibration; "
        "no donor-level or clinical inference."
    )

    v25 = latest(
        "paper7_panel_delay_v25",
        (
            "05_panel_unit_trajectories.csv",
            "09_model_comparison_summary.csv",
        ),
    )

    v28 = latest(
        "paper7_reference_ensemble_v28",
        (
            "03_real_panel_summary.csv",
            "04_synthetic_null_calibration_trials.csv",
            "05_synthetic_evaluation_trials.csv",
            "06_synthetic_gate_summary.csv",
            "07_real_gate_context.csv",
        ),
    )

    log("[SOURCE V25] " + str(v25))
    log("[SOURCE V28] " + str(v28))

    sources = [
        v25 / "05_panel_unit_trajectories.csv",
        v25 / "09_model_comparison_summary.csv",
        v28 / "03_real_panel_summary.csv",
        v28 / "04_synthetic_null_calibration_trials.csv",
        v28 / "05_synthetic_evaluation_trials.csv",
        v28 / "06_synthetic_gate_summary.csv",
        v28 / "07_real_gate_context.csv",
    ]

    save(
        "00_provenance.csv",
        [
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": hashfile(path),
                "bytes": path.stat().st_size,
            }
            for path in sources
        ],
    )

    datasets = read_units(sources[0])

    old25 = pd.read_csv(sources[1])
    old28 = pd.read_csv(sources[2])

    real_results = []

    for panel, (
        traces, controls, cases
    ) in datasets.items():

        result = evaluate(
            traces, controls, cases
        )

        result["panel"] = panel
        real_results.append(result)

        previous25 = old25[
            old25.panel == panel
        ].iloc[0]

        previous28 = old28[
            old28.panel == panel
        ].iloc[0]

        if not (
            np.isclose(
                result["fixedA"],
                previous25.amp_heldout_sse,
                atol=1e-8
            )
            and np.isclose(
                result["fixedD"],
                previous25.amp_delay_heldout_sse,
                atol=1e-8
            )
        ):
            raise RuntimeError(
                "V25 parity failed: " + panel
            )

        comparisons = [
            ("ensembleA", "ensemble_amp_SSE"),
            ("ensembleD", "ensemble_delay_SSE"),
            ("stable_frac", "stable_fold_fraction"),
        ]

        for key, field in comparisons:
            if not np.isclose(
                result[key],
                previous28[field],
                atol=1e-8
            ):
                raise RuntimeError(
                    "V28 real parity failed: "
                    + panel + " " + key
                )

        log(
            "[REAL PARITY PASS] {} | fixed={:+.2f}% "
            "| ensemble={:+.2f}% | stable={:.3f}".format(
                panel,
                result["fixed_gain"],
                result["ensemble_gain"],
                result["stable_frac"]
            )
        )

    save(
        "01_real_baseline_comparison.csv",
        real_results
    )

    primary = next(
        name for name in datasets
        if "all8" in name
    )

    traces, controls, cases = datasets[primary]

    controls_array = np.stack([
        traces[key] for key in controls
    ])

    mean = controls_array.mean(axis=0)
    residuals = controls_array - mean

    sd = controls_array.std(
        axis=0, ddof=1
    )

    synthetic_n = [
        f"N:{i}" for i in range(4)
    ]

    synthetic_e = [
        f"E:{i}" for i in range(4)
    ]

    old_cal = pd.read_csv(sources[3])
    old_eval = pd.read_csv(sources[4])
    old_summary = pd.read_csv(sources[5])

    calibration = []
    rng = np.random.default_rng(SEED + 1)

    for trial in range(N_CAL):
        data = simulate(
            rng, mean, residuals, sd,
            "null_same_shape"
        )

        result = evaluate(
            data, synthetic_n, synthetic_e
        )

        if not np.isclose(
            result["ensemble_gain"],
            old_cal.iloc[trial].gain_percent,
            rtol=0,
            atol=1e-8
        ):
            raise RuntimeError(
                f"V28 calibration parity failed: {trial}"
            )

        calibration.append({
            "trial": trial,
            **result,
        })

    save(
        "02_null_calibration_same_experiments.csv",
        calibration
    )

    threshold_fixed = float(
        np.quantile(
            [
                row["fixed_gain"]
                for row in calibration
            ],
            .95,
            method="higher"
        )
    )

    threshold_ensemble = float(
        np.quantile(
            [
                row["ensemble_gain"]
                for row in calibration
            ],
            .95,
            method="higher"
        )
    )

    log(
        "[CALIBRATION] fixed threshold={:+.4f}% | "
        "ensemble threshold={:+.4f}%".format(
            threshold_fixed,
            threshold_ensemble
        )
    )

    conditions = [
        ("null_same_shape", SEED + 2),
        ("delay_one_day", SEED + 3),
        ("null_shape_changed", SEED + 4),
        ("delay_plus_shape_changed", SEED + 5),
    ]

    methods = [
        "fixed",
        "ensemble",
        "ensemble_with_stability",
    ]

    trials = []
    summary = []

    for kind, seed in conditions:
        rng = np.random.default_rng(seed)

        wins = {
            method: 0 for method in methods
        }

        for trial in range(N_EVAL):
            data = simulate(
                rng, mean, residuals, sd, kind
            )

            result = evaluate(
                data, synthetic_n, synthetic_e
            )

            if kind != "delay_plus_shape_changed":
                old = old_eval[
                    (old_eval.truth == kind)
                    & (old_eval.trial == trial)
                ]

                if (
                    len(old) != 1
                    or not np.isclose(
                        result["ensemble_gain"],
                        old.iloc[0].gain_percent,
                        atol=1e-8,
                        rtol=0
                    )
                ):
                    raise RuntimeError(
                        "V28 evaluation parity failed: "
                        + kind
                    )

            fixed_decision = (
                result["fixed_gain"]
                > max(0., threshold_fixed)
            )

            ensemble_decision = (
                result["ensemble_gain"]
                > max(0., threshold_ensemble)
            )

            decisions = {
                "fixed": fixed_decision,
                "ensemble": ensemble_decision,
                "ensemble_with_stability": (
                    ensemble_decision
                    and result["stable_frac"] >= .5
                ),
            }

            for method in methods:
                wins[method] += int(
                    decisions[method]
                )

            trials.append({
                "scenario": kind,
                "trial": trial,
                **result,
                **{
                    "decision_" + key: int(value)
                    for key, value in decisions.items()
                },
            })

        for method in methods:
            low, high = wilson(
                wins[method], N_EVAL
            )

            summary.append({
                "scenario": kind,
                "method": method,
                "positive": wins[method],
                "n": N_EVAL,
                "fraction": wins[method] / N_EVAL,
                "wilson_conditional_low": low,
                "wilson_conditional_high": high,
            })

            log(
                "[BENCHMARK] {} | {} | "
                "{}/{} ({:.1%})".format(
                    kind,
                    method,
                    wins[method],
                    N_EVAL,
                    wins[method] / N_EVAL
                )
            )

    save(
        "03_paired_evaluation_trials.csv",
        trials
    )

    save(
        "04_matched_calibration_comparison.csv",
        summary
    )

    save_svg(summary)

    # Exact parity with previous v28.
    v28_cutoff = float(
        pd.read_csv(
            v28 / "07_real_gate_context.csv"
        ).iloc[0].synthetic_calibration_cutoff
    )

    if not np.isclose(
        threshold_ensemble,
        v28_cutoff,
        atol=1e-8,
        rtol=0
    ):
        raise RuntimeError(
            "V28 threshold parity failed"
        )

    for kind in (
        "null_same_shape",
        "delay_one_day",
        "null_shape_changed",
    ):
        old = old_summary[
            old_summary.truth == kind
        ].iloc[0]

        new = next(
            row for row in summary
            if row["scenario"] == kind
            and row["method"] == "ensemble_with_stability"
        )

        if new["positive"] != int(
            old.positive_decisions
        ):
            raise RuntimeError(
                "V28 gate parity failed: " + kind
            )

    log(
        "[PARITY PASS] V28 calibration, evaluation "
        "draws, thresholds and decisions reproduced."
    )

    notes = [
        "# Paper 7 v29 — Baseline comparison",
        "",
        "The fixed-reference model, reference ensemble "
        "and stability-gated ensemble were evaluated "
        "on identical synthetic experiments.",
        "",
        "Fixed-reference and ensemble decision "
        "thresholds were calibrated separately "
        "using independent no-delay simulations.",
        "",
        "This is conditional synthetic calibration, "
        "not a population-level significance test.",
        "",
        "## Real data",
    ]

    for row in real_results:
        notes.append(
            "- {}: fixed gain {:+.2f}%, "
            "ensemble gain {:+.2f}%, "
            "stable folds {:.1%}.".format(
                row["panel"],
                row["fixed_gain"],
                row["ensemble_gain"],
                row["stable_frac"]
            )
        )

    notes.extend([
        "",
        "## Synthetic decision fractions",
    ])

    for row in summary:
        notes.append(
            "- {} / {}: {}/{} ({:.1%}).".format(
                row["scenario"],
                row["method"],
                row["positive"],
                row["n"],
                row["fraction"]
            )
        )

    notes.extend([
        "",
        "## Limitations",
        "The 16 folds reuse the same biological units.",
        "Calibration is conditional on four observed "
        "control trajectories.",
        "Wilson intervals exclude uncertainty "
        "in the estimated calibration thresholds.",
        "The shape-change scenarios cover only "
        "specified synthetic perturbations.",
        "The decision score uses held-out outcomes "
        "and is not a prospective clinical test.",
        "Comparison with established functional "
        "registration methods remains necessary.",
    ])

    (OUT / "05_manuscript_critical_results.md").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "version": "v29",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": SEED,
        "calibration_trials": N_CAL,
        "evaluation_trials_per_scenario": N_EVAL,
        "thresholds": {
            "fixed": threshold_fixed,
            "ensemble": threshold_ensemble,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashfile(path)
            for path in sources
        },
        "script_sha256": hashfile(Path(__file__)),
        "limitations": [
            "Conditional simulator",
            "Four source control trajectories",
            "Overlapping held-out folds",
            "No donor-level significance",
            "No established clinical validity",
        ],
    }

    (OUT / "06_manifest.json").write_text(
        json.dumps(
            manifest, indent=2
        ),
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT / "script_snapshot.py"
    )

    log(
        "[COMPLETE] Matched baseline benchmark saved."
    )


if __name__ == "__main__":
    failed = False

    try:
        main()

    except Exception:
        failed = True
        log(
            "[ERROR]\n" + traceback.format_exc()
        )

    finally:
        report = OUT / "console_summary.txt"

        log("[REPORT] " + str(report))

        report.write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8"
        )

        archive = OUT / "portable_report_v29.zip"

        with zipfile.ZipFile(
            archive, "w", zipfile.ZIP_DEFLATED
        ) as z:
            for path in sorted(OUT.rglob("*")):
                if path.is_file() and path != archive:
                    z.write(
                        path,
                        path.relative_to(OUT)
                    )

        log("[ZIP] " + str(archive))

    if failed:
        sys.exit(1)