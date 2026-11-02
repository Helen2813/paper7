#!/usr/bin/env python3
"""OnsetAudit v1.0 — GSE205477 PGR-related preflight.

Not a time-course classifier, not direct PGR knockdown,
not an independent validation.

No model fitting or threshold changes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import urllib.request
import zipfile

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ACC = "GSE205477"

DATA = (
    ROOT / "onsetaudit"
    / "_generated" / "data" / ACC
)

REPORTS = (
    ROOT / "onsetaudit"
    / "_generated" / "reports"
)

PREFIX = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE205nnn/GSE205477"
)

FILES = {
    "matrix": (
        "matrix/GSE205477_series_matrix.txt.gz",
        8_000_000
    ),
    "counts": (
        "suppl/GSE205477_HESC-RNAseq-hg19-count.txt.gz",
        12_000_000
    ),
}

EXPECTED = {
    f"GSM{6213359 + i}" for i in range(12)
}

LOG = []


def log(*args):
    line = " ".join(map(str, args))
    print(line, flush=True)
    LOG.append(line)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def save_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )


def fetch(which):
    relative, size_limit = FILES[which]
    dest = DATA / Path(relative).name

    if dest.exists():
        with gzip.open(dest, "rb") as f:
            f.read(64)

        log(
            "[CACHED]", dest.name,
            "SHA256", sha(dest)
        )
        return dest

    url = PREFIX + "/" + relative
    log("[DOWNLOAD]", url)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "OnsetAudit-Research-Preflight/1.0"
        }
    )

    temp_path = None

    try:
        with urllib.request.urlopen(
            request, timeout=60
        ) as response:

            with tempfile.NamedTemporaryFile(
                dir=DATA,
                prefix=".geo_",
                delete=False
            ) as f:

                temp_path = Path(f.name)
                total = 0

                while True:
                    chunk = response.read(262144)

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > size_limit:
                        raise RuntimeError(
                            "Download exceeds size limit: "
                            + which
                        )

                    f.write(chunk)

        with gzip.open(temp_path, "rb") as f:
            f.read(64)

        temp_path.replace(dest)

        log(
            "[VERIFIED DOWNLOAD]",
            dest.name,
            dest.stat().st_size,
            "bytes",
            "SHA256", sha(dest)
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return dest


def parse_metadata(path):
    fields = {}

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:
            if line.startswith(
                "!series_matrix_table_begin"
            ):
                break

            if (
                line.startswith(
                    "!Sample_geo_accession\t"
                )
                or line.startswith(
                    "!Sample_title\t"
                )
            ):
                row = next(
                    csv.reader(
                        [line.rstrip("\r\n")],
                        delimiter="\t"
                    )
                )

                fields[row[0]] = row[1:]

    ids = fields.get(
        "!Sample_geo_accession", []
    )
    titles = fields.get(
        "!Sample_title", []
    )

    if (
        len(ids) != 12
        or len(titles) != 12
        or set(ids) != EXPECTED
    ):
        raise RuntimeError(
            "GEO sample manifest mismatch: "
            f"{len(ids)} IDs, "
            f"{len(titles)} titles"
        )

    records = []

    for gsm, title in zip(ids, titles):
        canonical = re.sub(
            r"[^a-z0-9]", "",
            title.lower()
        )

        match = re.search(
            r"p([123])", canonical
        )

        if (
            not match
            or (
                "sitrim28" not in canonical
                and "sint" not in canonical
            )
        ):
            raise RuntimeError(
                "Unrecognized donor/siRNA: "
                + gsm + " " + title
            )

        donor = "P" + match.group(1)

        sirna = (
            "siTRIM28"
            if "sitrim28" in canonical
            else "siNT"
        )

        if "predec" in canonical:
            condition = "Pre-dec"
        elif "dec" in canonical:
            condition = "Dec"
        else:
            raise RuntimeError(
                "Missing condition: " + gsm
            )

        records.append({
            "gsm": gsm,
            "geo_title": title,
            "donor": donor,
            "sirna": sirna,
            "condition": condition
        })

    df = pd.DataFrame(records)

    cells = df.groupby(
        ["donor", "sirna", "condition"]
    ).size()

    if (
        len(cells) != 12
        or not cells.eq(1).all()
    ):
        raise RuntimeError(
            "Incomplete donor × siRNA × condition design"
        )

    log(
        "[GEO METADATA]",
        "12/12 verified;",
        "3 donors × 2 siRNAs × 2 conditions"
    )

    return df


def canonical(value):
    return re.sub(
        r"[^a-z0-9]", "",
        str(value).lower()
    )


def map_column(column, metadata):
    key = canonical(column)
    matches = []

    for row in metadata.itertuples(index=False):
        if (
            canonical(row.gsm) in key
            or canonical(row.geo_title) in key
        ):
            matches.append(row.gsm)

    if len(set(matches)) == 1:
        return matches[0]

    return None


def inspect_counts(path, metadata, out):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace"
    ) as f:
        df = pd.read_csv(
            f,
            sep="\t",
            low_memory=False,
            comment="#"
        )

    if df.shape[1] == 1:
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace"
        ) as f:
            df = pd.read_csv(
                f,
                sep=",",
                low_memory=False,
                comment="#"
            )

    if df.empty or df.shape[1] < 3:
        raise RuntimeError(
            "Unusable count matrix: "
            + str(df.shape)
        )

    column_map = {}

    mapping_rows = []

    for column in df.columns:
        gsm = map_column(column, metadata)

        mapping_rows.append({
            "raw_column": str(column),
            "matched_gsm": gsm or ""
        })

        if gsm:
            column_map[gsm] = column

    pd.DataFrame(mapping_rows).to_csv(
        out / "count_column_mapping.csv",
        index=False
    )

    n_mapped = len(column_map)

    log(
        "[COUNTS]",
        len(df), "rows ×",
        len(df.columns), "columns;",
        f"matched {n_mapped}/12"
    )

    status = {
        "count_matrix_rows": int(len(df)),
        "count_matrix_columns": int(
            len(df.columns)
        ),
        "matched_sample_columns": n_mapped,
        "count_column_mapping_verified": False,
        "integer_nonnegative_counts_verified": False,
        "processed_count_qc": None,
        "count_column_preview": [
            str(x) for x in df.columns[:22]
        ],
    }

    if (
        n_mapped != 12
        or len(set(column_map.values())) != 12
    ):
        log(
            "[GATE]",
            "COUNT_COLUMN_MAPPING_NOT_EVALUABLE"
        )
        return status

    selected = [
        column_map[gsm]
        for gsm in metadata.gsm
    ]

    values = df[selected].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    counts_valid = bool(
        np.isfinite(values).all()
        and np.all(values >= 0)
        and np.allclose(
            values,
            np.rint(values),
            atol=1e-8,
            rtol=0
        )
    )

    if not counts_valid:
        log(
            "[GATE]",
            "COUNTS_NOT_VERIFIED_INTEGER"
        )
        return status

    status[
        "count_column_mapping_verified"
    ] = True

    status[
        "integer_nonnegative_counts_verified"
    ] = True

    qc = []

    for i, row in metadata.reset_index(
        drop=True
    ).iterrows():

        x = values[:, i]

        qc.append({
            "gsm": row.gsm,
            "donor": row.donor,
            "sirna": row.sirna,
            "condition": row.condition,
            "processed_total_counts": int(
                x.sum()
            ),
            "genes_positive": int(
                np.count_nonzero(x)
            )
        })

    pd.DataFrame(qc).to_csv(
        out / "sample_counts_qc.csv",
        index=False
    )

    status["processed_count_qc"] = {
        "min_sum": min(
            r["processed_total_counts"]
            for r in qc
        ),
        "max_sum": max(
            r["processed_total_counts"]
            for r in qc
        )
    }

    log(
        "[QC]",
        "12 processed count vectors verified"
    )

    return status


def find_frozen_weights():
    names = {
        "v0236_mapped_weights.csv",
        "source_weights.csv"
    }

    found = []

    for directory, subdirs, filenames in os.walk(
        ROOT
    ):
        relative = Path(directory).relative_to(
            ROOT
        )

        if len(relative.parts) > 9:
            subdirs[:] = []
            continue

        excluded = {
            ".git",
            ".venv",
            ".venv-onsetaudit",
            "node_modules",
            "__pycache__",
            "_generated"
        }

        subdirs[:] = [
            d for d in subdirs
            if d not in excluded
            and not d.startswith(".venv-")
        ]

        for name in filenames:
            if name not in names:
                continue

            path = Path(directory) / name

            found.append({
                "path": str(path),
                "filename": name,
                "sha256": sha(path),
                "bytes": path.stat().st_size
            })

    return sorted(
        found,
        key=lambda x: x["path"]
    )


def make_dashboard(metadata, readiness):
    table_rows = ""

    for row in metadata[
        [
            "gsm",
            "geo_title",
            "donor",
            "sirna",
            "condition"
        ]
    ].itertuples(index=False, name=None):

        cells = "".join(
            "<td>" + html.escape(str(v)) + "</td>"
            for v in row
        )

        table_rows += "<tr>" + cells + "</tr>"

    status = html.escape(
        readiness["eligibility_status"]
    )

    n_weights = readiness[
        "frozen_pgr_weights_found"
    ]

    n_matched = readiness[
        "counts"
    ]["matched_sample_columns"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OnsetAudit v1.0 PGR preflight</title>
<style>
body {{
  font: 15px system-ui, sans-serif;
  margin: 2rem auto;
  max-width: 1050px;
  padding: 0 1rem;
  color: #1c2e43;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
td,th {{
  border-bottom: 1px solid #ddd;
  padding: .6rem;
  text-align: left;
}}
.note {{
  padding: 1rem;
  border-left: 4px solid #bd8b31;
  background: #fff8e7;
}}
code {{
  overflow-wrap: anywhere;
}}
</style>
</head>
<body>

<h1>OnsetAudit v1.0</h1>
<h2>Orthogonal PGR-related preflight</h2>

<p><b>{status}</b></p>

<div class="note">
GSE205477 is human endometrial stroma with
TRIM28 knockdown and EPC versus pre-decidualized
cells. It is not direct PGR knockdown and not
an exposure-duration time course.

The frozen OnsetAudit model has not been applied.
No PGR transfer performance is claimed.
</div>

<p>
Verified GEO samples: {len(metadata)}<br>
Matched count columns: {n_matched}/12<br>
Frozen PGR weight candidates: {n_weights}
</p>

<h2>Verified donor design</h2>

<table>
<tr>
<th>GSM</th>
<th>Title</th>
<th>Donor</th>
<th>siRNA</th>
<th>Condition</th>
</tr>
{table_rows}
</table>

<h2>Source provenance</h2>

<p>
Metadata SHA-256:
<code>{readiness["source_files"]["matrix"]["sha256"]}</code>
</p>

<p>
Counts SHA-256:
<code>{readiness["source_files"]["counts"]["sha256"]}</code>
</p>

</body>
</html>"""


def main():
    DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    out = REPORTS / (
        "v10_pgr_orthogonal_preflight_" + stamp
    )

    out.mkdir(exist_ok=False)

    try:
        log("=" * 70)

        log(
            "PAPER 7 | ONSETAUDIT v1.0 |",
            "GSE205477 ORTHOGONAL PREFLIGHT"
        )

        log(
            "NO FLAGS | NO MODEL FITTING |",
            "NO FROZEN v0.4 CHANGES"
        )

        log("=" * 70)

        sources = {
            key: fetch(key)
            for key in FILES
        }

        metadata = parse_metadata(
            sources["matrix"]
        )

        metadata.to_csv(
            out / "geo_verified_12_samples.csv",
            index=False
        )

        qc = inspect_counts(
            sources["counts"],
            metadata,
            out
        )

        weights = find_frozen_weights()

        log(
            "[PGR FROZEN WEIGHTS]",
            len(weights),
            "candidate files (not applied)"
        )

        for item in weights[:8]:
            log(
                "[WEIGHT CANDIDATE]",
                item["filename"],
                item["sha256"],
                item["path"]
            )

        pairing = []

        for donor in ("P1", "P2", "P3"):
            for sirna in (
                "siNT",
                "siTRIM28"
            ):
                subset = metadata[
                    (metadata.donor == donor)
                    & (metadata.sirna == sirna)
                ]

                pre = subset[
                    subset.condition == "Pre-dec"
                ].iloc[0]

                dec = subset[
                    subset.condition == "Dec"
                ].iloc[0]

                pairing.append({
                    "donor": donor,
                    "sirna": sirna,
                    "predec_gsm": pre.gsm,
                    "dec_gsm": dec.gsm,
                    "planned_contrast":
                        "Dec_minus_Predec",
                    "contrast_status":
                        "NOT_COMPUTED_PGR_WEIGHTS_NOT_VALIDATED"
                })

        pd.DataFrame(pairing).to_csv(
            out / "prespecified_donor_pairs.csv",
            index=False
        )

        if not qc[
            "integer_nonnegative_counts_verified"
        ]:
            status = (
                "NOT_EVALUABLE_COUNT_SCHEMA_"
                "REQUIRES_RECONCILIATION"
            )

        elif not weights:
            status = (
                "COUNTS_READY_PGR_TRANSFER_"
                "NOT_EVALUABLE_FROZEN_WEIGHTS_MISSING"
            )

        else:
            status = (
                "COUNTS_READY_PGR_TRANSFER_"
                "NOT_EVALUABLE_WEIGHT_MAPPING_NOT_VERIFIED"
            )

        readiness = {
            "version": "v1.0.0",
            "dataset": ACC,
            "eligibility_status": status,

            "source_PGR_signature":
                "Frozen GSE94036 PGR siRNA weights",

            "hypothesis":
                "Donor-paired EPC response under "
                "TRIM28 knockdown",

            "frozen_v04_model_applied": False,
            "elapsed_duration_model_evaluated": False,
            "new_biological_performance_estimated": False,
            "weights_applied": False,

            "donor_overlap_with_source_verified": False,

            "source_files": {
                key: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                    "official_url":
                        PREFIX + "/" + FILES[key][0]
                }
                for key, path in sources.items()
            },

            "counts": qc,

            "frozen_pgr_weights_found":
                len(weights),

            "candidate_weights": weights,

            "next_gate":
                "Verify exact frozen v0236 PGR "
                "weights, gene identifiers and "
                "mapping before scoring",

            "limitations": [
                "Not a time course",
                "TRIM28 is not PGR knockdown",
                "Only three donors",
                "Source donor overlap not verified",
                "Not independent lab validation",
                "No measured molecular onset"
            ],

            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__
        }

        save_json(
            out / "readiness.json",
            readiness
        )

        (out / "README.md").write_text(
            "# GSE205477 PGR-related preflight\n\n"
            "No model fitting or score calculation.\n"
            "12 GEO samples, 3 donor-paired "
            "2×2 siRNA/EPC design.\n"
            "This is not a time-course or "
            "direct PGR knockdown.\n"
            "Frozen PGR weights must be "
            "verified before scoring.\n",
            encoding="utf-8"
        )

        dashboard = (
            out / "onsetaudit_v10_dashboard.html"
        )

        dashboard.write_text(
            make_dashboard(
                metadata,
                readiness
            ),
            encoding="utf-8"
        )

        shutil.copy2(
            __file__,
            out / Path(__file__).name
        )

        log("[GATE]", status)
        log("[OUTPUT]", out)

        (out / "console.log").write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8"
        )

        save_json(
            out / "sha256_manifest.json",
            {
                p.name: sha(p)
                for p in out.iterdir()
                if p.is_file()
            }
        )

        archive = (
            out / "onsetaudit_v10_pgr_preflight.zip"
        )

        with zipfile.ZipFile(
            archive,
            "w",
            zipfile.ZIP_DEFLATED
        ) as z:

            for p in sorted(out.iterdir()):
                if p.is_file() and p != archive:
                    z.write(p, p.name)

        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError(
                    "Corrupt output ZIP"
                )

        log("[ZIP]", archive)

        if sys.platform == "darwin":
            import subprocess

            subprocess.run(
                ["open", str(dashboard)],
                check=False
            )

    except Exception:
        (out / "failure.log").write_text(
            "\n".join(LOG)
            + "\n"
            + traceback.format_exc(),
            encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            "[STOP]",
            type(e).__name__,
            str(e),
            file=sys.stderr,
            flush=True
        )
        sys.exit(1)
