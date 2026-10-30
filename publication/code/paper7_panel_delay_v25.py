#!/usr/bin/env python3
"""PAPER 7 v25 | Frozen gene-panel analysis.
Offline. No flags. No donor-level inference.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import hashlib
import json
import re
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

RUN = ROOT / "paper7_panel_delay_v25" / (
    "run_" +
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
RUN.mkdir(parents=True, exist_ok=True)

GEO = ROOT / "geo_results/metadata/GSE243039_samples.csv"

MATRIX = ROOT / (
    "geo_results/raw/GSE243039/"
    "GSE243039_normalized_all_counts.csv.gz"
)

HGNC = ROOT / "geo_results/reference/hgnc_complete_set.txt"

EXPECTED_HGNC_SHA256 = (
    "12363916be36f8f6f91879b0faade4eeaac973722"
    "de7136ec61329b77f2c2f91"
)

DAYS = (0, 2, 4, 6, 8)

PANEL8 = (
    "PRL", "IGFBP1", "FOXO1", "HAND2",
    "WNT4", "PGR", "IL11", "LEFTY2"
)

PANEL6 = (
    "PRL", "IGFBP1", "FOXO1",
    "HAND2", "WNT4", "PGR"
)

# Signs based on earlier external descriptive data.
# NOT predefined biological truths.
DIRECTION = {
    "PRL": 1,
    "IGFBP1": 1,
    "FOXO1": 1,
    "HAND2": 1,
    "WNT4": 1,
    "PGR": -1,
    "IL11": 1,
    "LEFTY2": 1,
}

LOG = []


def say(s):
    print(str(s), flush=True)
    LOG.append(str(s))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(
        RUN / name,
        index=False,
        encoding="utf-8-sig"
    )
    say("[OUTPUT] {} | rows={}".format(name, len(df)))
    return df


def source_map():
    md = pd.read_csv(GEO, dtype=str).fillna("")

    if (
        len(md) != 40
        or not {"gsm", "title"}.issubset(md.columns)
    ):
        raise ValueError(
            "Expected 40 GEO rows with gsm/title."
        )

    pattern = re.compile(
        r"^\s*(Normal endometrial stroma|"
        r"Endometriosis stroma),\s*"
        r"Replicate\s*(\d+),\s*"
        r"Day\s*(0|2|4|6|8)\s*$",
        re.I,
    )

    rows = []

    for _, rec in md.iterrows():
        match = pattern.fullmatch(rec["title"])

        if not match:
            raise ValueError(
                "Unrecognized GEO title: " +
                repr(rec["title"])
            )

        group = (
            "N"
            if match.group(1).lower().startswith("normal")
            else "E"
        )

        rep = int(match.group(2))
        day = int(match.group(3))

        rows.append({
            "gsm": rec["gsm"],
            "title": rec["title"],
            "group": group,
            "replicate_label": rep,
            "day": day,
            "matrix_column": (
                f"{group}{rep:02d}_{day}"
            ),
        })

    design = pd.DataFrame(rows)

    if (
        design.gsm.duplicated().any()
        or design.matrix_column.duplicated().any()
    ):
        raise ValueError("Duplicate source identifiers.")

    for group in ("N", "E"):
        part = design[design.group == group]
        units = sorted(part.replicate_label.unique())

        if len(units) != 4:
            raise ValueError(
                "Expected four replicate labels per group."
            )

        for rep in units:
            observed = set(
                part[
                    part.replicate_label == rep
                ].day
            )

            if observed != set(DAYS):
                raise ValueError(
                    f"Incomplete time course: {group}{rep}"
                )

    save("01_original_GEO_title_design.csv", rows)
    return design


def mapping():
    actual_hash = sha(HGNC)

    if actual_hash != EXPECTED_HGNC_SHA256:
        raise ValueError(
            "HGNC changed since v23; "
            "annotation review required. "
            "Actual SHA256: " + actual_hash
        )

    h = pd.read_csv(
        HGNC,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "symbol", "ensembl_gene_id", "status"
    }

    if not required.issubset(h.columns):
        raise ValueError("Unexpected HGNC schema.")

    h = h[
        (h.status.str.lower() == "approved")
        & (h.symbol.isin(PANEL8))
    ]

    result = []
    gene_ids = {}

    for gene in PANEL8:
        candidates = h[
            h.symbol == gene
        ].ensembl_gene_id

        ids = sorted(set(
            re.sub(r"\.\d+$", "", value.strip())
            for value in candidates
            if re.fullmatch(
                r"ENSG\d{11}(?:\.\d+)?",
                value.strip()
            )
        ))

        result.append({
            "gene": gene,
            "ensembl_ids": ";".join(ids),
            "n_ids": len(ids),
            "external_orientation": DIRECTION[gene],
            "secondary_six_gene_panel": (
                gene in PANEL6
            ),
        })

        if len(ids) == 1:
            gene_ids[gene] = ids[0]

    save("02_annotated_frozen_panel.csv", result)

    if len(gene_ids) != 8:
        raise ValueError(
            "Not all eight genes map uniquely."
        )

    return gene_ids


def load_expression(design, gene_ids):
    matrix = pd.read_csv(
        MATRIX,
        compression="gzip",
        index_col=0,
        low_memory=False,
    )

    if (
        matrix.shape[1] != 40
        or matrix.columns.duplicated().any()
        or matrix.index.duplicated().any()
    ):
        raise ValueError("Unexpected matrix structure.")

    if set(matrix.columns) != set(
        design.matrix_column
    ):
        raise ValueError(
            "GEO labels and matrix columns disagree."
        )

    identifiers = pd.Index([
        re.sub(r"\.\d+$", "", str(value))
        for value in matrix.index
    ])

    if identifiers.duplicated().any():
        raise ValueError(
            "Duplicate stable Ensembl identifiers."
        )

    matrix.index = identifiers

    missing = [
        gene
        for gene, eid in gene_ids.items()
        if eid not in matrix.index
    ]

    if missing:
        raise ValueError(
            "Missing genes in primary cohort: " +
            str(missing)
        )

    selected = matrix.loc[
        [gene_ids[g] for g in PANEL8],
        design.matrix_column.tolist()
    ]

    values = selected.to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(
            "Nonfinite or negative panel expression."
        )

    # Explicit exploratory transform.
    # These are published normalized counts,
    # not raw RNA-seq counts.
    transformed = np.log2(1.0 + values)

    samples = {}

    for j, rec in enumerate(design.to_dict("records")):
        key = (
            rec["group"],
            int(rec["replicate_label"]),
            int(rec["day"]),
        )
        samples[key] = transformed[:, j].copy()

    rows = []

    for j, rec in enumerate(design.to_dict("records")):
        group = rec["group"]
        rep = int(rec["replicate_label"])
        day = int(rec["day"])

        baseline = samples[group, rep, 0]
        current = samples[group, rep, day]

        for k, gene in enumerate(PANEL8):
            delta = float(current[k] - baseline[k])

            rows.append({
                "gsm": rec["gsm"],
                "group": group,
                "replicate_label": rep,
                "day": day,
                "gene": gene,
                "ensembl_id": gene_ids[gene],
                "normalized_value": float(values[k, j]),
                "log2_1plus_normalized": float(current[k]),
                "within_unit_log2_change": delta,
                "externally_oriented_change": (
                    delta * DIRECTION[gene]
                ),
            })

    raw = save(
        "03_frozen_gene_per_sample_trajectories.csv",
        rows
    )

    summary = (
        raw.groupby(
            ["gene", "group", "day"],
            sort=False
        )["within_unit_log2_change"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0)

    save(
        "04_gene_group_descriptive.csv",
        summary.to_dict("records")
    )

    say(
        "[MATRIX] features={} | samples=40 | "
        "panel genes=8/8".format(matrix.shape[0])
    )

    say(
        "[SCALE] Exploratory log2(1+published "
        "normalized count), baseline-centered."
    )

    return samples, summary


def panel_scores(samples, genes, name):
    positions = [
        PANEL8.index(gene) for gene in genes
    ]

    directions = np.array([
        DIRECTION[gene] for gene in genes
    ], dtype=float)

    rows = []
    values = {}

    for (group, rep, day), current in sorted(
        samples.items()
    ):
        baseline = samples[group, rep, 0]

        change = (
            current[positions] - baseline[positions]
        ) * directions

        score = float(np.mean(change))
        values[group, rep, day] = score

        rows.append({
            "panel": name,
            "group": group,
            "replicate_label": rep,
            "day": day,
            "score": score,
            "n_genes": len(genes),
        })

    return rows, values


def shifted_reference(reference, delay, times):
    return np.interp(
        np.asarray(times, dtype=float) - delay,
        np.asarray(DAYS, dtype=float),
        np.asarray(reference, dtype=float),
        left=0.0,
    )


def fit_models(reference, target):
    times = np.asarray(DAYS[1:], dtype=float)
    target = np.asarray(target, dtype=float)[1:]

    output = []

    for model, shifts in (
        ("amplitude_only", [0.0]),
        (
            "amplitude_plus_delay",
            np.arange(0, 2.00001, 0.25)
        ),
    ):
        best = None

        for delay in shifts:
            curve = shifted_reference(
                reference, float(delay), times
            )

            denominator = float(
                np.dot(curve, curve)
            )

            amplitude = (
                max(
                    0.0,
                    float(np.dot(curve, target) / denominator)
                )
                if denominator > 1e-12
                else 0.0
            )

            sse = float(
                np.sum(
                    (target - amplitude * curve) ** 2
                )
            )

            candidate = {
                "model": model,
                "amplitude": amplitude,
                "delay_days": float(delay),
                "training_sse": sse,
            }

            if (
                best is None
                or sse < best["training_sse"] - 1e-12
            ):
                best = candidate

        output.append(best)

    return output


def double_holdout(scores, panel):
    controls = sorted({
        rep for group, rep, day in scores
        if group == "N"
    })

    cases = sorted({
        rep for group, rep, day in scores
        if group == "E"
    })

    fits = []
    comparisons = []

    for heldout_n in controls:
        for heldout_e in cases:

            train_n = [
                unit for unit in controls
                if unit != heldout_n
            ]

            train_e = [
                unit for unit in cases
                if unit != heldout_e
            ]

            reference = np.array([
                np.mean([
                    scores["N", unit, day]
                    for unit in train_n
                ])
                for day in DAYS
            ])

            training_case = np.array([
                np.mean([
                    scores["E", unit, day]
                    for unit in train_e
                ])
                for day in DAYS
            ])

            heldout_case = np.array([
                scores["E", heldout_e, day]
                for day in DAYS
            ])

            heldout_control = np.array([
                scores["N", heldout_n, day]
                for day in DAYS
            ])

            reference_error = float(
                np.sum(
                    (
                        heldout_control[1:]
                        - reference[1:]
                    ) ** 2
                )
            )

            model_fits = fit_models(
                reference, training_case
            )

            errors = {}

            for fitted in model_fits:
                prediction = (
                    fitted["amplitude"]
                    * shifted_reference(
                        reference,
                        fitted["delay_days"],
                        DAYS[1:]
                    )
                )

                error = float(
                    np.sum(
                        (
                            heldout_case[1:]
                            - prediction
                        ) ** 2
                    )
                )

                errors[fitted["model"]] = error

                fits.append({
                    "panel": panel,
                    "heldout_N": heldout_n,
                    "heldout_E": heldout_e,
                    "model": fitted["model"],
                    "fitted_amplitude": fitted["amplitude"],
                    "fitted_delay_days": fitted["delay_days"],
                    "training_sse": fitted["training_sse"],
                    "heldout_E_sse": error,
                    "heldout_N_reference_sse": reference_error,
                })

            gain = (
                errors["amplitude_only"]
                - errors["amplitude_plus_delay"]
            )

            winner = (
                "delay" if gain > 1e-9
                else "amplitude" if gain < -1e-9
                else "tie"
            )

            comparisons.append({
                "panel": panel,
                "heldout_N": heldout_n,
                "heldout_E": heldout_e,
                "SSE_gain_amp_minus_delay": gain,
                "winner": winner,
            })

    return fits, comparisons


def plot_svg(filename, rows, title):
    # No matplotlib dependency.
    width, height = 760, 445
    left, top = 75, 67
    plot_w, plot_h = 651, 302

    colors = {
        "N": "#1767aa",
        "E": "#ce622b",
    }

    points = {
        group: sorted(
            [
                row for row in rows
                if row["group"] == group
            ],
            key=lambda row: row["day"]
        )
        for group in ("N", "E")
    }

    values = []

    for group in ("N", "E"):
        for row in points[group]:
            mean = float(row["mean"])
            sd = float(row.get("std", 0) or 0)
            values.extend([
                mean, mean - sd, mean + sd
            ])

    low = min(values + [0])
    high = max(values + [0])
    span = max(high - low, 0.3)

    low -= span * 0.13
    high += span * 0.13

    def X(value):
        return left + plot_w * float(value) / 8

    def Y(value):
        return (
            top
            + plot_h * (high - float(value)) / (high - low)
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 760 445">',
        '<rect width="760" height="445" fill="white"/>',
        '<text x="75" y="32" font-family="Arial" '
        'font-size="20" font-weight="bold">'
        + title + '</text>',
        '<text x="75" y="50" font-family="Arial" '
        'font-size="12">'
        'Mean ± SD across four replicate labels'
        '</text>',
    ]

    for i in range(5):
        value = low + (high - low) * i / 4
        y = Y(value)

        parts.append(
            '<line x1="75" x2="726" '
            'y1="{:.1f}" y2="{:.1f}" '
            'stroke="#e5e7eb"/>'.format(y, y)
        )

    for day in DAYS:
        parts.append(
            '<text x="{:.1f}" y="392" '
            'font-family="Arial" font-size="12" '
            'text-anchor="middle">{}</text>'.format(
                X(day), day
            )
        )

    for group in ("N", "E"):
        color = colors[group]
        data = points[group]

        coordinates = " ".join(
            "{:.1f},{:.1f}".format(
                X(row["day"]),
                Y(row["mean"])
            )
            for row in data
        )

        parts.append(
            '<polyline points="{}" fill="none" '
            'stroke="{}" stroke-width="3"/>'.format(
                coordinates, color
            )
        )

        for row in data:
            x = X(row["day"])
            value = float(row["mean"])
            sd = float(row.get("std", 0) or 0)

            parts.append(
                '<line x1="{:.1f}" x2="{:.1f}" '
                'y1="{:.1f}" y2="{:.1f}" '
                'stroke="{}" opacity=".5"/>'.format(
                    x, x,
                    Y(value - sd), Y(value + sd),
                    color
                )
            )

            parts.append(
                '<circle cx="{:.1f}" cy="{:.1f}" '
                'r="4" fill="{}"/>'.format(
                    x, Y(value), color
                )
            )

    parts.extend([
        '<text x="390" y="410" font-family="Arial" '
        'font-size="13" text-anchor="middle">'
        'Days after induction</text>',
        '<text x="75" y="428" font-family="Arial" '
        'font-size="11">'
        'Blue: control; orange: endometriosis. '
        'Descriptive, not inferential.</text>',
        '</svg>',
    ])

    folder = RUN / "figures"
    folder.mkdir(exist_ok=True)

    (folder / filename).write_text(
        "\n".join(parts),
        encoding="utf-8"
    )


def main():
    say(
        "PAPER 7 | V25 | FROZEN PANEL + "
        "EXPLORATORY DOUBLE HOLDOUT"
    )
    say("[ROOT] " + str(ROOT))
    say("[OUTPUT] " + str(RUN))

    say(
        "[POLICY] GEO study reports four normal "
        "and four endometriosis patients."
    )
    say(
        "[POLICY] Replicate-to-donor crosswalk "
        "not independently verified."
    )
    say(
        "[POLICY] 8-gene panel exploratory; "
        "6-gene panel secondary, selected after v24."
    )

    inputs = []

    for name, path in (
        ("GSE243039_metadata", GEO),
        ("GSE243039_normalized_matrix", MATRIX),
        ("HGNC_local", HGNC),
    ):
        if not path.is_file():
            raise FileNotFoundError(str(path))

        inputs.append({
            "name": name,
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        })

    save("00_input_sha256.csv", inputs)

    design = source_map()
    gene_ids = mapping()
    samples, gene_summary = load_expression(
        design, gene_ids
    )

    all_scores = []
    score_tables = {}

    panels = (
        ("all8_external_orientation", PANEL8),
        (
            "six_external_concordant_sensitivity",
            PANEL6
        ),
    )

    for name, genes in panels:
        rows, scores = panel_scores(
            samples, genes, name
        )
        all_scores.extend(rows)
        score_tables[name] = scores

    score_df = save(
        "05_panel_unit_trajectories.csv",
        all_scores
    )

    group_df = (
        score_df.groupby(
            ["panel", "group", "day"]
        ).score.agg(
            ["mean", "std", "count"]
        ).reset_index()
    ).fillna(0)

    save(
        "06_panel_group_descriptive.csv",
        group_df.to_dict("records")
    )

    all_fits = []
    all_pairs = []

    for name, scores in score_tables.items():
        fits, pairs = double_holdout(
            scores, name
        )

        all_fits.extend(fits)
        all_pairs.extend(pairs)

    save(
        "07_double_holdout_fits.csv",
        all_fits
    )

    save(
        "08_double_holdout_pair_differences.csv",
        all_pairs
    )

    summaries = []

    for name in score_tables:
        subset = [
            row for row in all_fits
            if row["panel"] == name
        ]

        amp = sum(
            row["heldout_E_sse"]
            for row in subset
            if row["model"] == "amplitude_only"
        )

        delayed = sum(
            row["heldout_E_sse"]
            for row in subset
            if row["model"] == "amplitude_plus_delay"
        )

        pairs = [
            row for row in all_pairs
            if row["panel"] == name
        ]

        delays = [
            row["fitted_delay_days"]
            for row in subset
            if row["model"] == "amplitude_plus_delay"
        ]

        gain_percent = (
            100 * (amp - delayed) / amp
            if amp > 0 else float("nan")
        )

        wins_d = sum(
            row["winner"] == "delay"
            for row in pairs
        )

        wins_a = sum(
            row["winner"] == "amplitude"
            for row in pairs
        )

        ties = sum(
            row["winner"] == "tie"
            for row in pairs
        )

        summary = {
            "panel": name,
            "n_N_units": 4,
            "n_E_units": 4,
            "overlapping_folds": 16,
            "amp_heldout_sse": amp,
            "amp_delay_heldout_sse": delayed,
            "relative_SSE_gain_percent": gain_percent,
            "delay_wins": wins_d,
            "amplitude_wins": wins_a,
            "ties": ties,
            "median_training_delay_days": float(
                np.median(delays)
            ),
        }

        summaries.append(summary)

        say(
            "[DOUBLE HOLDOUT] {} | "
            "amp SSE={:.6f} | "
            "amp+delay SSE={:.6f} | "
            "gain={:+.2f}% | "
            "wins delay/amp/tie={}/{}/{} | "
            "median delay={:.2f} d".format(
                name, amp, delayed, gain_percent,
                wins_d, wins_a, ties,
                summary["median_training_delay_days"]
            )
        )

    save(
        "09_model_comparison_summary.csv",
        summaries
    )

    for gene in PANEL8:
        rows = gene_summary[
            gene_summary.gene == gene
        ].to_dict("records")

        plot_svg(
            "gene_" + gene + ".svg",
            rows,
            "GSE243039 | " + gene,
        )

    for name in score_tables:
        rows = group_df[
            group_df.panel == name
        ].to_dict("records")

        plot_svg(
            "panel_" + name + ".svg",
            rows,
            "GSE243039 | " + name,
        )

    say(
        "[FIGURES] 10 SVG figures, "
        "no matplotlib required."
    )

    notes = [
        "# Paper 7 v25 — exploratory results",
        "",
        "## Data",
        "GSE243039: four normal and four endometriosis "
        "patients according to GEO overall design, "
        "five measured days. Individual replicate "
        "identifiers were parsed from GEO titles.",
        "",
        "## Genes",
        "All eight genes were retained. Orientation "
        "was informed by earlier external data. "
        "The six-gene subpanel was selected after "
        "v24 and is secondary, not confirmatory.",
        "",
        "## Measurement",
        "Published normalized counts transformed as "
        "log2(1+x), then centered at the day-0 value "
        "for each replicate label.",
        "",
        "## Model comparison",
        "Each fold excludes one N and one E unit. "
        "Three N units define the reference. "
        "Three E units define the training target. "
        "Prediction error is evaluated on held-out "
        "E days 2, 4, 6 and 8.",
        "",
        "Amplitude is nonnegative. Delay is limited "
        "to 0–2 days in 0.25-day steps.",
        "",
        "## Descriptive outcomes",
    ]

    for row in summaries:
        notes.append(
            "- {}: amplitude SSE {:.4f}, "
            "amplitude+delay SSE {:.4f}, "
            "relative improvement {:+.2f}%, "
            "delay/amplitude/tie {}/{}/{}.".format(
                row["panel"],
                row["amp_heldout_sse"],
                row["amp_delay_heldout_sse"],
                row["relative_SSE_gain_percent"],
                row["delay_wins"],
                row["amplitude_wins"],
                row["ties"],
            )
        )

    notes.extend([
        "",
        "## Limitations",
        "Sixteen folds share eight replicate labels "
        "and are not independent tests.",
        "No fold-derived p-values.",
        "No verified participant-level uncertainty "
        "analysis was performed.",
        "The six-gene sensitivity analysis is "
        "externally informed and post hoc.",
        "Fitted shifts are interpolation-dependent "
        "and do not prove biological delay.",
        "The published normalized-count scale is "
        "not a raw-count statistical model.",
    ])

    (RUN / "10_manuscript_notes.md").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "version": "v25",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sources": inputs,
        "panel8": PANEL8,
        "panel6_secondary": PANEL6,
        "orientation": DIRECTION,
        "overlapping_folds": True,
        "no_inferential_claims": True,
        "script_sha256": sha(Path(__file__)),
    }

    (RUN / "11_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        RUN / "script_snapshot.py"
    )

    say("[COMPLETE] Real trajectories analyzed.")
    say("[COMPLETE] Original data unchanged.")


if __name__ == "__main__":
    failure = None

    try:
        main()

    except Exception:
        failure = traceback.format_exc()
        say("[ERROR] " + failure)

    finally:
        report = RUN / "console_summary.txt"

        say("[REPORT] " + str(report))

        report.write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8"
        )

        archive = RUN / "portable_report_v25.zip"

        with zipfile.ZipFile(
            archive,
            "w",
            zipfile.ZIP_DEFLATED
        ) as z:
            for path in sorted(RUN.rglob("*")):
                if path.is_file() and path != archive:
                    z.write(
                        path,
                        path.relative_to(RUN)
                    )

        say("[ZIP] " + str(archive))

    if failure:
        sys.exit(1)