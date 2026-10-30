#!/usr/bin/env python3
"""Paper 7 v30: paired benchmark audit and manuscript evidence.
Offline, no flags, no model refitting, no new packages.
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
OUT = ROOT / "paper7_submission_evidence_v30" / (
    "run_" +
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
METHODS = {
    "fixed": "decision_fixed",
    "ensemble": "decision_ensemble",
    "ensemble_with_stability":
        "decision_ensemble_with_stability",
}
SCENARIOS = (
    "null_same_shape",
    "delay_one_day",
    "null_shape_changed",
    "delay_plus_shape_changed",
)
BOOTSTRAPS = 5000
SEED = 20271031


def log(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig",
    )
    log(f"[OUTPUT] {name} | rows={len(df)}")
    return df


def latest_valid():
    base = ROOT / "paper7_benchmark_v29"
    required = (
        "03_paired_evaluation_trials.csv",
        "04_matched_calibration_comparison.csv",
        "01_real_baseline_comparison.csv",
        "06_manifest.json",
    )

    for folder in sorted(
        base.glob("run_*"), reverse=True
    ):
        if all(
            (folder / name).is_file()
            for name in required
        ):
            return folder

    raise FileNotFoundError(
        "No complete Paper 7 v29 benchmark found"
    )


def wilson(k, n):
    z = 1.95996398454
    p = k / n
    denominator = 1 + z*z/n

    center = (
        p + z*z/(2*n)
    ) / denominator

    radius = (
        z * math.sqrt(
            p*(1-p)/n + z*z/(4*n*n)
        ) / denominator
    )

    return (
        max(0., center-radius),
        min(1., center+radius),
    )


def main():
    log(
        "PAPER 7 | V30 | PAIRED BENCHMARK "
        "EVIDENCE AND MANUSCRIPT GATE"
    )
    log("[ROOT] " + str(ROOT))
    log("[OUTPUT] " + str(OUT))
    log(
        "[POLICY] Reuses v29 results only. "
        "No clinical or donor-level inference."
    )

    source = latest_valid()
    log("[SOURCE] " + str(source))

    names = (
        "03_paired_evaluation_trials.csv",
        "04_matched_calibration_comparison.csv",
        "01_real_baseline_comparison.csv",
        "06_manifest.json",
    )

    save(
        "00_source_hashes.csv",
        [
            {
                "file": str(
                    (source / name).relative_to(ROOT)
                ),
                "sha256": sha(source / name),
                "bytes": (source / name).stat().st_size,
            }
            for name in names
        ],
    )

    trials = pd.read_csv(source / names[0])
    summary = pd.read_csv(source / names[1])
    real = pd.read_csv(source / names[2])

    required = {
        "scenario", "trial", *METHODS.values()
    }

    if not required.issubset(trials.columns):
        raise RuntimeError(
            "Missing v29 trial columns: " +
            str(sorted(required - set(trials.columns)))
        )

    if (
        len(trials) != 1000
        or len(summary) != 12
        or len(real) != 2
    ):
        raise RuntimeError(
            "Unexpected v29 table lengths"
        )

    for col in METHODS.values():
        values = pd.to_numeric(
            trials[col], errors="raise"
        )

        if not values.isin([0, 1]).all():
            raise RuntimeError(
                "Decisions not binary: " + col
            )

        trials[col] = values.astype(int)

    if trials.duplicated([
        "scenario", "trial"
    ]).any():
        raise RuntimeError(
            "Duplicate simulation trial identity"
        )

    if set(trials.scenario) != set(SCENARIOS):
        raise RuntimeError(
            "Unexpected simulation scenarios"
        )

    rng = np.random.default_rng(SEED)

    rates = []
    contrasts = []

    for scenario in SCENARIOS:
        data = trials.loc[
            trials.scenario == scenario
        ].sort_values(
            "trial"
        ).reset_index(drop=True)

        if (
            len(data) != 250
            or data.trial.tolist() != list(range(250))
        ):
            raise RuntimeError(
                "Incomplete trials: " + scenario
            )

        for method, column in METHODS.items():
            successes = int(data[column].sum())
            low, high = wilson(
                successes, len(data)
            )

            prior = summary[
                (summary.scenario == scenario)
                & (summary.method == method)
            ]

            if (
                len(prior) != 1
                or int(prior.iloc[0].positive)
                    != successes
            ):
                raise RuntimeError(
                    "V29 summary parity failed: "
                    + scenario + " / " + method
                )

            rates.append({
                "scenario": scenario,
                "method": method,
                "positives": successes,
                "n": len(data),
                "rate": successes / len(data),
                "conditional_Wilson_low": low,
                "conditional_Wilson_high": high,
            })

        fixed = data[
            METHODS["fixed"]
        ].to_numpy(dtype=int)

        for method in (
            "ensemble",
            "ensemble_with_stability",
        ):
            alternative = data[
                METHODS[method]
            ].to_numpy(dtype=int)

            paired_delta = alternative - fixed

            # Resample entire simulation experiments,
            # not independent decisions or folds.
            indices = rng.integers(
                0, len(data),
                size=(BOOTSTRAPS, len(data)),
            )

            replicates = paired_delta[
                indices
            ].mean(axis=1)

            low, high = np.quantile(
                replicates, [.025, .975]
            )

            record = {
                "scenario": scenario,
                "contrast": method + "_minus_fixed",
                "difference_percentage_points": (
                    100 * paired_delta.mean()
                ),
                "paired_bootstrap_low_pp": (
                    100 * float(low)
                ),
                "paired_bootstrap_high_pp": (
                    100 * float(high)
                ),
                "both_yes": int(
                    ((fixed == 1)
                     & (alternative == 1)).sum()
                ),
                "fixed_only": int(
                    ((fixed == 1)
                     & (alternative == 0)).sum()
                ),
                "alternative_only": int(
                    ((fixed == 0)
                     & (alternative == 1)).sum()
                ),
                "both_no": int(
                    ((fixed == 0)
                     & (alternative == 0)).sum()
                ),
                "n_independent_synthetic_experiments":
                    len(data),
            }

            contrasts.append(record)

            log(
                "[PAIRED] {} | {} - fixed = "
                "{:+.1f} pp | conditional 95% "
                "[{:+.1f}, {:+.1f}]".format(
                    scenario,
                    method,
                    record[
                        "difference_percentage_points"
                    ],
                    record["paired_bootstrap_low_pp"],
                    record["paired_bootstrap_high_pp"],
                )
            )

    save("01_scenario_rates.csv", rates)
    save(
        "02_paired_method_contrasts.csv",
        contrasts,
    )

    required_real = {
        "panel",
        "fixedA", "fixedD",
        "ensembleA", "ensembleD",
        "fixed_gain", "ensemble_gain",
        "stable_frac",
    }

    if not required_real.issubset(real.columns):
        raise RuntimeError(
            "Missing v29 real-data columns"
        )

    for _, row in real.iterrows():
        log(
            "[REAL] {} | fixed gain={:+.2f}% | "
            "ensemble gain={:+.2f}% | "
            "stable={:.3f}".format(
                row["panel"],
                float(row["fixed_gain"]),
                float(row["ensemble_gain"]),
                float(row["stable_frac"]),
            )
        )

    save(
        "03_real_model_errors.csv",
        real.to_dict("records"),
    )

    # Mathematical illustration:
    # for f(t)=exp(-t/tau), shifting the curve
    # is equivalent to rescaling its amplitude,
    # provided all observation times exceed delay.
    t = np.array([2., 4., 6., 8.])
    tau = 3.
    amplitude = .8
    delay = .75

    shifted = amplitude * np.exp(
        -(t - delay) / tau
    )

    equivalent = (
        amplitude * np.exp(delay / tau)
    ) * np.exp(-t / tau)

    discrepancy = float(
        np.max(np.abs(shifted - equivalent))
    )

    if discrepancy > 1e-12:
        raise RuntimeError(
            "Analytic symmetry test failed"
        )

    save(
        "04_analytic_counterexample.csv",
        [
            {
                "time_days": float(x),
                "shifted_exponential": float(y),
                "amplitude_only_equivalent": float(z),
                "difference": float(y-z),
            }
            for x, y, z in zip(
                t, shifted, equivalent
            )
        ],
    )

    log(
        "[IDENTIFIABILITY EXAMPLE] Exact "
        "exponential shift/amplitude equivalence | "
        "max error={:.2e}".format(discrepancy)
    )

    def get_rate(scenario, method):
        return next(
            r for r in rates
            if r["scenario"] == scenario
            and r["method"] == method
        )["rate"]

    null_fixed = get_rate(
        "null_same_shape", "fixed"
    )

    delay_fixed = get_rate(
        "delay_one_day", "fixed"
    )

    complex_fixed = get_rate(
        "delay_plus_shape_changed", "fixed"
    )

    manuscript = [
        "# Paper 7 v30 — Manuscript-safe results",
        "",
        "## Paired synthetic benchmark",
        "",
        "Three decision rules were compared on "
        "identical independently generated synthetic "
        "experiments: a fixed-reference model, "
        "a control-reference ensemble, and an "
        "ensemble with a delay-stability requirement.",
        "",
        "Each rule was calibrated using separate "
        "no-delay simulations. Evaluation included "
        "250 simulated experiments per scenario.",
        "",
        "The fixed-reference rule generated "
        "positive decisions in {:.1f}% of no-delay "
        "simulations, {:.1f}% of experiments with "
        "a one-day delay, and {:.1f}% when a "
        "one-day delay co-occurred with a "
        "response-shape perturbation.".format(
            100*null_fixed,
            100*delay_fixed,
            100*complex_fixed,
        ),
        "",
        "Reference ensembling did not provide a "
        "clear predictive or detection advantage "
        "over the simpler fixed-reference model.",
        "",
        "Paired uncertainty intervals were computed "
        "by resampling entire synthetic experiments. "
        "They condition on the fitted calibration "
        "thresholds and therefore do not account "
        "for calibration uncertainty.",
        "",
        "## Analytic identifiability example",
        "",
        "For an exponential-decay response "
        "f(t)=exp(-t/tau), a shifted response "
        "a*f(t-delay) equals "
        "a*exp(delay/tau)*f(t) for t>delay.",
        "",
        "Without an externally fixed amplitude, "
        "these two mechanisms are observationally "
        "equivalent on that response segment.",
        "",
        "This is an elementary counterexample, "
        "not a new universal identifiability theorem.",
        "",
        "## Limitations",
        "",
        "- Results are conditional on the specified "
        "synthetic data-generating mechanism.",
        "- Bootstrap intervals omit uncertainty "
        "in estimated null thresholds.",
        "- The real-data double-holdout folds "
        "reuse the same eight biological labels.",
        "- The algorithms are not validated "
        "clinical decision rules.",
        "- Amplitude-phase separation and "
        "functional registration are existing "
        "statistical research areas.",
        "",
        "## Recommendation",
        "",
        "Retain the reference ensemble as a "
        "negative benchmark rather than claiming "
        "it is a superior new algorithm.",
    ]

    (OUT / "05_manuscript_results_v30.md").write_text(
        "\n".join(manuscript) + "\n",
        encoding="utf-8",
    )

    table = [
        "| Scenario | Fixed | Ensemble | "
        "Ensemble + stability |",
        "|---|---:|---:|---:|",
    ]

    for scenario in SCENARIOS:
        values = [
            "{:.1%}".format(get_rate(
                scenario, method
            ))
            for method in METHODS
        ]

        table.append(
            "| {} | {} | {} | {} |".format(
                scenario, *values
            )
        )

    (OUT / "06_main_benchmark_table.md").write_text(
        "\n".join(table) + "\n",
        encoding="utf-8",
    )

    (OUT / "07_manifest.json").write_text(
        json.dumps(
            {
                "version": "v30",
                "utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "bootstrap_replicates": BOOTSTRAPS,
                "seed": SEED,
                "source_v29": str(
                    source.relative_to(ROOT)
                ),
                "script_sha256": sha(Path(__file__)),
                "source_hashes": {
                    name: sha(source / name)
                    for name in names
                },
                "conditional_intervals_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.copy2(
        Path(__file__),
        OUT / "script_snapshot.py",
    )

    log(
        "[COMPLETE] Paired comparisons, "
        "analytic example and manuscript tables saved."
    )


if __name__ == "__main__":
    failed = False

    try:
        main()
    except Exception:
        failed = True
        log("[ERROR]\n" + traceback.format_exc())
    finally:
        log(
            "[REPORT] " +
            str(OUT / "console_summary.txt")
        )

        (OUT / "console_summary.txt").write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8",
        )

        archive = OUT / "portable_report_v30.zip"

        with zipfile.ZipFile(
            archive, "w", zipfile.ZIP_DEFLATED
        ) as z:
            for item in sorted(OUT.rglob("*")):
                if item.is_file() and item != archive:
                    z.write(
                        item,
                        item.relative_to(OUT),
                    )

        log("[ZIP] " + str(archive))

    if failed:
        sys.exit(1)