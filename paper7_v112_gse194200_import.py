#!/usr/bin/env python3
"""Paper 7 V112: download and audit real GSE194200 RNA-seq data."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import csv
import gzip
import json
import platform
import posixpath
import re
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "GSE194200"
OUT = ROOT / "paper7_gse194200_ingest_v112" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194200/"
SOURCES = {
    "GSE194200_series_matrix.txt.gz":
        BASE + "matrix/GSE194200_series_matrix.txt.gz",
    "GSE194200_Kalwat_TotalRNAseq_processeddata.xlsx":
        BASE + "suppl/GSE194200_Kalwat_TotalRNAseq_processeddata.xlsx"
}

def say(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def valid(path):
    if not path.is_file() or path.stat().st_size < 100:
        return False
    if path.suffix == ".xlsx":
        return path.stat().st_size > 1000 and zipfile.is_zipfile(path)
    if path.name.endswith(".gz"):
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    return False

def download(name, url):
    dest = DATA / name
    if valid(dest):
        say("[CACHE] {} | {:.2f} MB".format(
            name, dest.stat().st_size / 1048576
        ))
        return dest

    temporary = DATA / (name + ".partial")
    errors = []

    for attempt in range(1, 4):
        try:
            say("[DOWNLOAD] {} | attempt {}/3".format(name, attempt))

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Paper7-GEO-Research-Audit/1.0",
                    "Accept": "*/*"
                }
            )

            total = 0
            next_report = 5 * 1048576

            with urllib.request.urlopen(req, timeout=120) as response:
                with temporary.open("wb") as f:
                    while True:
                        block = response.read(1048576)
                        if not block:
                            break
                        f.write(block)
                        total += len(block)

                        if total >= next_report:
                            say("[DOWNLOADING] {:.1f} MB".format(
                                total / 1048576
                            ))
                            next_report += 5 * 1048576

            require(
                temporary.stat().st_size > 100,
                "Downloaded file is empty"
            )
            temporary.replace(dest)
            require(valid(dest), "Invalid downloaded file")

            say("[DOWNLOADED] {}".format(name))
            say("[SHA256] {}".format(sha(dest)))
            return dest

        except Exception as exc:
            errors.append(
                "{}: {}: {}".format(
                    attempt, type(exc).__name__, str(exc)[:200]
                )
            )
            say("[RETRY] " + errors[-1])

            if temporary.exists():
                temporary.unlink()
            if dest.exists() and not valid(dest):
                dest.unlink()
            if attempt < 3:
                time.sleep(attempt * 2)

    raise RuntimeError(
        "Download failed: {} | {}".format(name, " | ".join(errors))
    )

def save_csv(name, rows, columns):
    with (OUT / name).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    say("[OUTPUT] {} | {} rows".format(name, len(rows)))

def parse_geo(path):
    needed = (
        "!Sample_geo_accession",
        "!Sample_title",
        "!Sample_description"
    )
    fields = {key: [] for key in needed}
    characteristics = []

    with gzip.open(
        path, "rt", encoding="utf-8", errors="replace"
    ) as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue

            row = next(csv.reader(
                [line.rstrip("\r\n")],
                delimiter="\t",
                quotechar='"'
            ))
            key = row[0]

            if key in fields:
                fields[key] = row[1:]
            if key == "!Sample_characteristics_ch1":
                characteristics.append(row[1:])

    for key in (
        "!Sample_geo_accession",
        "!Sample_title"
    ):
        require(
            len(fields[key]) == 39,
            "GEO metadata count mismatch: " + key
        )

    rows = []

    for i, (gsm, title) in enumerate(zip(
        fields["!Sample_geo_accession"],
        fields["!Sample_title"]
    )):
        m = re.fullmatch(
            r"\s*(.*?)\s+(0|1|2|6|24)\s*h\s+N\s*=\s*([123])\s*",
            title,
            re.IGNORECASE
        )
        require(m is not None, "Unexpected sample title: " + title)

        treatment, hour, experiment = m.groups()
        normalized = treatment.lower().replace(" ", "")

        aliases = {
            "baseline": "Baseline",
            "dmso": "DMSO",
            "sw016789": "SW016789",
            "thapsigargin": "Thapsigargin"
        }
        require(normalized in aliases, "Unknown treatment: " + treatment)

        desc = (
            fields["!Sample_description"][i]
            if len(fields["!Sample_description"]) == 39
            else ""
        )
        extra = " | ".join(
            x[i] for x in characteristics
            if len(x) == 39 and x[i]
        )

        rows.append(dict(
            gsm=gsm,
            title=title,
            description=desc,
            experiment=int(experiment),
            treatment=aliases[normalized],
            hours=int(hour),
            characteristics=extra
        ))

    require(
        len(set(x["gsm"] for x in rows)) == 39,
        "Duplicate GEO sample accessions"
    )

    expected = {("Baseline", 0)}
    expected |= {
        (t, h)
        for t in ("DMSO", "SW016789", "Thapsigargin")
        for h in (1, 2, 6, 24)
    }

    for exp in (1, 2, 3):
        subset = [x for x in rows if x["experiment"] == exp]
        require(
            len(subset) == 13
            and {(x["treatment"], x["hours"]) for x in subset}
                == expected,
            "Experimental design mismatch: experiment {}".format(exp)
        )
        say(
            "[DESIGN VERIFIED] experiment {} | 13 conditions"
            .format(exp)
        )

    save_csv(
        "01_GEO_sample_metadata.csv",
        rows,
        [
            "gsm", "title", "description", "experiment",
            "treatment", "hours", "characteristics"
        ]
    )

    for x in rows[:5]:
        say(
            "[METADATA] {} | {} | description={}".format(
                x["gsm"], x["title"], x["description"][:90]
            )
        )

    return rows

def localname(tag):
    return tag.rsplit("}", 1)[-1]

def shared_strings(z):
    if "xl/sharedStrings.xml" not in z.namelist():
        return []

    strings = []

    with z.open("xl/sharedStrings.xml") as f:
        for _, element in ET.iterparse(f, events=("end",)):
            if localname(element.tag) == "si":
                value = "".join(
                    n.text or ""
                    for n in element.iter()
                    if localname(n.tag) == "t"
                )
                strings.append(value)
                element.clear()

    say("[XLSX] shared strings={}".format(len(strings)))
    return strings

def column_number(label):
    value = 0
    for c in label:
        value = value * 26 + ord(c) - 64
    return value

def sheets_in_workbook(z):
    workbook = ET.fromstring(z.read("xl/workbook.xml"))
    relationships = ET.fromstring(
        z.read("xl/_rels/workbook.xml.rels")
    )

    targets = {
        n.attrib["Id"]: n.attrib["Target"]
        for n in relationships
        if "Id" in n.attrib and "Target" in n.attrib
    }

    sheets = []

    for node in workbook.iter():
        if localname(node.tag) != "sheet":
            continue

        rid = next(
            (
                v for k, v in node.attrib.items()
                if localname(k) == "id"
            ),
            None
        )
        require(rid in targets, "Missing Excel relationship")

        target = targets[rid]

        if target.startswith("/"):
            location = target.lstrip("/")
        else:
            location = posixpath.normpath(
                posixpath.join("xl", target)
            )

        require(
            location in z.namelist(),
            "Missing Excel sheet: " + location
        )

        sheets.append((node.attrib.get("name", ""), location))

    require(sheets, "No Excel worksheets")
    return sheets

def inspect_workbook(path):
    inventory = []
    cells = []

    with zipfile.ZipFile(path) as z:
        say("[XLSX] Verifying archive and inspecting headers")
        require(
            z.testzip() is None,
            "Excel internal CRC verification failed"
        )

        strings = shared_strings(z)
        sheets = sheets_in_workbook(z)

        for sheet_name, location in sheets:
            dimension = "UNAVAILABLE"
            inspected = 0

            with z.open(location) as f:
                for _, element in ET.iterparse(
                    f, events=("end",)
                ):
                    kind = localname(element.tag)

                    if kind == "dimension":
                        dimension = element.attrib.get(
                            "ref", "UNAVAILABLE"
                        )

                    if kind != "row":
                        continue

                    rn = int(element.attrib.get(
                        "r", inspected + 1
                    ))

                    if rn <= 14:
                        for cell in element:
                            if localname(cell.tag) != "c":
                                continue

                            address = cell.attrib.get("r", "")
                            match = re.fullmatch(
                                r"([A-Z]+)(\d+)", address
                            )
                            if not match:
                                continue

                            typ = cell.attrib.get("t", "")
                            value = None

                            if typ == "inlineStr":
                                value = "".join(
                                    x.text or ""
                                    for x in cell.iter()
                                    if localname(x.tag) == "t"
                                )
                            else:
                                content = next(
                                    (
                                        x for x in cell
                                        if localname(x.tag) == "v"
                                    ),
                                    None
                                )
                                if (
                                    content is not None
                                    and content.text is not None
                                ):
                                    value = content.text
                                    if typ == "s":
                                        index = int(value)
                                        require(
                                            0 <= index < len(strings),
                                            "Invalid shared-string reference"
                                        )
                                        value = strings[index]

                            if value is not None and str(value).strip():
                                cells.append(dict(
                                    sheet=sheet_name,
                                    row=rn,
                                    column=column_number(match.group(1)),
                                    address=address,
                                    value=str(value)[:260]
                                ))

                    element.clear()
                    inspected += 1

                    if inspected >= 14:
                        break

            count = sum(
                x["sheet"] == sheet_name for x in cells
            )

            inventory.append(dict(
                sheet=sheet_name,
                worksheet_xml=location,
                declared_dimension=dimension,
                previewed_rows=inspected,
                nonempty_header_cells=count
            ))

            say(
                "[XLSX SHEET] {} | dimension={} | head cells={}"
                .format(sheet_name, dimension, count)
            )

            first_rows = sorted(set(
                x["row"] for x in cells
                if x["sheet"] == sheet_name
            ))[:8]

            for row in first_rows:
                items = [
                    x for x in cells
                    if x["sheet"] == sheet_name
                    and x["row"] == row
                ]
                preview = "; ".join(
                    "{}={}".format(
                        x["address"], x["value"][:65]
                    )
                    for x in items[:9]
                )
                say(
                    "[XLSX HEADER | {} | row {}] {}{}"
                    .format(
                        sheet_name,
                        row,
                        preview,
                        "; ..." if len(items) > 9 else ""
                    )
                )

    save_csv(
        "02_workbook_sheets.csv",
        inventory,
        [
            "sheet", "worksheet_xml", "declared_dimension",
            "previewed_rows", "nonempty_header_cells"
        ]
    )
    save_csv(
        "03_workbook_first14rows.csv",
        cells,
        ["sheet", "row", "column", "address", "value"]
    )

    candidates = [
        x for x in cells
        if re.search(
            r"(raw.?count|read.?count|gene.?id|fpkm|\btpm\b|sample|x776)",
            x["value"], re.IGNORECASE
        )
    ]
    save_csv(
        "04_raw_count_header_candidates.csv",
        candidates,
        ["sheet", "row", "column", "address", "value"]
    )

    say(
        "[XLSX HEADERS] candidate cells={} | all head cells={}"
        .format(len(candidates), len(cells))
    )
    return inventory

def main():
    say("=" * 88)
    say("PAPER 7 | V112 | GSE194200 EXTERNAL DATA INGEST")
    say(
        "REAL RNA-SEQ | GEO METADATA | NO TRAINING | NO GIT"
    )
    say("=" * 88)

    require(len(sys.argv) == 1, "No flags permitted")

    paths = {}
    for name, url in SOURCES.items():
        paths[name] = download(name, url)

    samples = parse_geo(
        paths["GSE194200_series_matrix.txt.gz"]
    )
    sheets = inspect_workbook(
        paths["GSE194200_Kalwat_TotalRNAseq_processeddata.xlsx"]
    )

    manifest = dict(
        version="v112",
        geo="GSE194200",
        run_utc=datetime.now(timezone.utc).isoformat(),
        sample_count=len(samples),
        experiments=3,
        conditions_per_experiment=13,
        source_urls=SOURCES,
        downloaded={
            name: dict(
                file=str(path),
                bytes=path.stat().st_size,
                sha256=sha(path)
            )
            for name, path in paths.items()
        },
        excel_sheets=[x["sheet"] for x in sheets],
        script_sha256=sha(__file__),
        python=sys.version,
        platform=platform.platform(),
        training_performed=False,
        no_git=True
    )

    (OUT / "05_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8"
    )

    protocol = """# Frozen first validation protocol

Goal: predict the experimentally documented exposure
duration among 1, 2, 6 and 24 hours.

This is not the unknown transcriptomic onset time.

Use raw gene counts, not FPKM or TPM as raw counts.

Match all 39 count columns to GEO GSM accessions,
treatments, durations and independent experiments.

Primary outer evaluation:
leave one entire experiment out.
Three outer folds: train on two experiments, test
on the third.

All feature selection and model tuning must occur
within the training experiments only.

Use simple baselines before a new method:
nearest-centroid and regularized prediction.

Evaluate SW016789 and thapsigargin separately.
Use DMSO as a time-matched negative control.

Report predictions for every held-out sample,
including failure cases.

Do not claim 95 percent distribution-free coverage
from only three independent experimental batches.

A second independent study and comparison with
published methods are required for broader claims.

V112 performs no model training and makes no
claim of new method validation.
"""

    (OUT / "06_frozen_validation_protocol.md").write_text(
        protocol, encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__), OUT / "script_snapshot_v112.py"
    )

    say(
        "[GATE] DATA AUDIT COMPLETE; RAW COUNT "
        "COLUMN MAPPING NOT YET CONFIRMED"
    )
    say(
        "[NEXT] Verify 39 raw-count columns against "
        "GEO metadata before held-out model evaluation"
    )

def finish(success):
    say(
        "[RESULT] " + ("SUCCESS" if success else "FAILED")
    )
    say("[REPORT] " + str(OUT / "console_summary.txt"))

    archive = OUT / "portable_report_v112.zip"
    say("[ZIP] " + str(archive))

    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "07_sha256_inventory.csv", archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    save_csv(
        "07_sha256_inventory.csv",
        items,
        ["filename", "bytes", "sha256"]
    )

    say("[ZIP] Packaging source files, code and reports")

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED,
        compresslevel=2
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for p in sorted(DATA.iterdir()):
            if p.is_file() and valid(p):
                z.write(p, "source_data/" + p.name)

    say(
        "[ZIP COMPLETE] original GEO files, "
        "code, metadata, checksums and logs"
    )

if __name__ == "__main__":
    success = True

    try:
        main()
    except Exception:
        success = False
        say("[ERROR]\n" + traceback.format_exc())

    try:
        finish(success)
    except Exception:
        success = False
        print(
            "[ARCHIVE ERROR]\n" + traceback.format_exc(),
            flush=True
        )

    if not success:
        sys.exit(1)
