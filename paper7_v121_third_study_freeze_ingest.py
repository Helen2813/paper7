#!/usr/bin/env python3
"""V121: freeze V120 architecture before reading GSE214965."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import csv, gzip, json, platform, shutil, sys, time
import traceback, urllib.request, zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_v121_third_study_lock" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
DATA = ROOT / "data" / "GSE214965"

OUT.mkdir(parents=True, exist_ok=False)
DATA.mkdir(parents=True, exist_ok=True)
LOG = []

STEM = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE214nnn/GSE214965/suppl/"
)
FILES = (
    "GSE214965_sample_metadata_LN229_DRUG_Seq.txt.gz",
    "GSE214965_rawcounts_LN229_DRUG_Seq.txt.gz",
    "GSE214965_normalizedcounts_LN229_DRUG_Seq.txt.gz",
)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def sha(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def csvout(name, rows, columns):
    with (OUT/name).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=columns,
            extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)

    say("[OUTPUT] {} | {} rows".format(name, len(rows)))

def freeze():
    base = ROOT/"paper7_v120_twostage_audit"
    required = (
        "09_sha256_inventory.csv",
        "01_protocol.json",
        "04_hybrid_summary.csv",
        "script_snapshot_v120.py"
    )

    runs = sorted(
        (
            r for r in base.glob("run_*")
            if r.is_dir()
            and all((r/x).is_file() for x in required)
        ),
        reverse=True
    )
    need(runs, "Verified V120 archive not found")
    run = runs[0]

    inv = pd.read_csv(
        run/"09_sha256_inventory.csv",
        encoding="utf-8-sig"
    )
    need(
        {"filename","bytes","sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "V120 inventory malformed"
    )

    for x in inv.itertuples(index=False):
        name = str(x.filename)
        p = run/name
        need(
            p.name == name
            and p.is_file()
            and p.stat().st_size == int(x.bytes)
            and sha(p) == str(x.sha256),
            "V120 archived file modified: "+name
        )

    protocol = json.loads(
        (run/"01_protocol.json").read_text("utf-8")
    )
    summary = pd.read_csv(
        run/"04_hybrid_summary.csv",
        encoding="utf-8-sig"
    )

    need(
        protocol.get("version") == "v120"
        and protocol.get(
            "architecture_selected_after_V119_outcomes"
        ) is True
        and len(summary) == 1
        and int(
            summary.iloc[0].hybrid_correct_and_accepted
        ) == 79
        and int(summary.iloc[0].DMSO_rejected) == 29,
        "Frozen V120 results mismatch"
    )

    original = ROOT/"paper7_v120_twostage_audit.py"
    need(
        original.is_file()
        and sha(original) == sha(
            run/"script_snapshot_v120.py"
        ),
        "V120 source code changed"
    )

    for source,target in (
        (
            "01_protocol.json",
            "input_v120_development_protocol.json"
        ),
        (
            "04_hybrid_summary.csv",
            "input_v120_hybrid_summary.csv"
        ),
        (
            "script_snapshot_v120.py",
            "input_v120_script_snapshot.py"
        )
    ):
        shutil.copy2(run/source, OUT/target)

    lock = dict(
        version="v121",
        role="V120_HYBRID_LOCKED_BEFORE_GSE214965",
        development_datasets=[
            "GSE194200", "GSE149428"
        ],
        external_dataset="GSE214965",
        target="experimental exposure duration",
        not_biological_onset=True,
        candidate_hours=[6,22],
        five_class_validation=False,
        gene_count=200,
        variance_floor=0.25,
        ridge_per_gene=0.25,
        selection_rule=(
            "Training-only temporal variance "
            "of drug-minus-DMSO contrasts divided "
            "by replicate variance plus 0.25"
        ),
        raw_count_filter=(
            "At least 10 counts in at least "
            "3 training samples"
        ),
        normalization="log2(1 + CPM) from raw counts",
        activity_gate=(
            "Training-only pooled DMSO reference, "
            "zero score-margin threshold"
        ),
        duration_model=(
            "Training-only time-matched DMSO "
            "with nonnegative ridge amplitude"
        ),
        known_treatment_identity=True,
        DMSO_evaluation=(
            "Per planned drug and unique DMSO "
            "any-drug alert, reported separately"
        ),
        no_test_DMSO_self_reference=True,
        comparator=(
            "Vehicle-only method with "
            "identical reference access"
        ),
        primary_split=(
            "Sequencing batches replicates 1-2 "
            "versus 3-4, only if metadata verifies"
        ),
        batch_structure_not_yet_verified=True,
        no_reselection_after_outcomes=True,
        failure_policy=(
            "NOT_EVALUABLE if controls or "
            "batch metadata incomplete"
        ),
        independent_95pct_coverage_unproven=True,
        methodological_novelty_unproven=True,
        V120_protocol_sha256=sha(
            run/"01_protocol.json"
        ),
        V120_script_sha256=sha(original),
        V121_script_sha256=sha(__file__)
    )

    target = OUT/"01_third_study_locked_protocol.json"
    target.write_text(
        json.dumps(lock, indent=2)+"\n",
        encoding="utf-8"
    )

    say(
        "[V120 VERIFIED] {} archived files "
        "| 79/90 active correct-and-accepted "
        "| 29/30 DMSO rejected".format(len(inv))
    )
    say(
        "[THIRD STUDY LOCKED] GSE214965 "
        "| 6 vs 22h | 200 genes | ridge=.25 | floor=.25"
    )
    say("[LOCK SHA256] "+sha(target))
    return target

def gzip_valid(path):
    if not path.is_file() or path.stat().st_size < 50:
        return False, 0

    total = 0
    try:
        with gzip.open(path, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                total += len(data)
        return total > 0, total
    except (OSError, EOFError, ValueError):
        return False, 0

def download(name):
    destination = DATA/name
    good, original_bytes = gzip_valid(destination)

    if good:
        say(
            "[CACHED] {} | SHA256={}".format(
                name, sha(destination)
            )
        )
        return destination, "cache", original_bytes

    urls = [
        STEM+name,
        (
            "https://www.ncbi.nlm.nih.gov/geo/download/"
            "?acc=GSE214965&format=file&file="+name
        )
    ]

    partial = DATA/(name+".partial")
    errors = []

    for url in urls:
        for attempt in (1,2):
            try:
                say(
                    "[DOWNLOAD] {} | attempt={}/2".format(
                        name, attempt
                    )
                )

                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent":
                            "Mozilla/5.0 Paper7Research/1.0",
                        "Accept":
                            "application/octet-stream,*/*"
                    }
                )

                with urllib.request.urlopen(
                    request, timeout=120
                ) as response, partial.open("wb") as out:

                    head = response.read(64)

                    say(
                        "[HTTP] status={} | type={} | first={}"
                        .format(
                            response.status,
                            response.headers.get(
                                "Content-Type",""
                            ),
                            repr(head[:24])
                        )
                    )

                    need(
                        head.startswith(b"\x1f\x8b"),
                        "Server response is not gzip"
                    )
                    out.write(head)

                    total = len(head)
                    next_report = 5*1048576

                    while True:
                        block = response.read(1048576)
                        if not block:
                            break
                        out.write(block)
                        total += len(block)

                        if total >= next_report:
                            say(
                                "[PROGRESS] {:.1f} MiB".format(
                                    total/1048576
                                )
                            )
                            next_report += 5*1048576

                good, original_bytes = gzip_valid(partial)
                need(
                    good,
                    "Decompression or gzip CRC check failed"
                )

                partial.replace(destination)

                say(
                    "[SOURCE VERIFIED] {} | {} bytes "
                    "| SHA256={}".format(
                        name,
                        destination.stat().st_size,
                        sha(destination)
                    )
                )

                return destination, url, original_bytes

            except Exception as exc:
                errors.append(
                    "{}: {}".format(
                        type(exc).__name__,
                        str(exc)[:160]
                    )
                )
                say("[RETRY] "+errors[-1])

                if partial.exists():
                    partial.unlink()
                if attempt == 1:
                    time.sleep(1)

    raise RuntimeError(
        "All download sources failed for "+name+
        " | "+" | ".join(errors)
    )

def preview(path):
    rows = []
    widths = set()
    nrows = 0

    with gzip.open(
        path, "rt",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        first = f.readline()
        need(first.strip(), "Empty table")

        sep = (
            "\t" if "\t" in first
            else "," if "," in first
            else None
        )
        need(sep is not None, "Unknown table delimiter")

        f.seek(0)
        reader = csv.reader(f, delimiter=sep)

        for i,row in enumerate(reader):
            nrows += 1
            widths.add(len(row))

            if i < 5:
                rows.append(dict(
                    file=path.name,
                    row=i+1,
                    width=len(row),
                    first_10=" | ".join(
                        str(x)[:80] for x in row[:10]
                    ),
                    last_5=" | ".join(
                        str(x)[:60] for x in row[-5:]
                    )
                ))

            if i and i%15000 == 0:
                say(
                    "[SCAN] {} | {} data rows"
                    .format(path.name,i)
                )

    need(
        nrows >= 2
        and len(widths) == 1
        and max(widths) > 1,
        "Inconsistent or incomplete tabular file"
    )

    width = next(iter(widths))

    say(
        "[TABLE] {} | lines={} | columns={} | delimiter={}"
        .format(
            path.name,nrows,width,repr(sep)
        )
    )

    for row in rows[:3]:
        say(
            "[HEADER | {} | row {}] {}"
            .format(
                path.name,
                row["row"],
                row["first_10"][:240]
            )
        )

    return rows,dict(
        filename=path.name,
        lines=nrows,
        columns=width,
        delimiter=repr(sep)
    )

def main():
    say("="*88)
    say(
        "PAPER 7 | V121 | FREEZE TWO-STAGE METHOD "
        "AND INGEST GSE214965"
    )
    say(
        "THIRD STUDY | RAW COUNTS | "
        "NO FITTING | NO PREDICTIONS | NO GIT"
    )
    say("="*88)

    need(len(sys.argv)==1,"Run without flags")

    locked = freeze()

    source_rows = []
    header_rows = []
    shapes = []

    for name in FILES:
        path,url,original_bytes = download(name)
        headers,info = preview(path)

        header_rows.extend(headers)
        shapes.append(info)

        source_rows.append(dict(
            filename=name,
            source_url=url,
            bytes=path.stat().st_size,
            uncompressed_bytes=original_bytes,
            sha256=sha(path)
        ))

    csvout(
        "02_third_study_file_inventory.csv",
        source_rows,
        [
            "filename","source_url","bytes",
            "uncompressed_bytes","sha256"
        ]
    )
    csvout(
        "03_first_five_rows_per_file.csv",
        header_rows,
        [
            "file","row","width",
            "first_10","last_5"
        ]
    )
    csvout(
        "04_table_shapes.csv",
        shapes,
        [
            "filename","lines","columns","delimiter"
        ]
    )

    (OUT/"05_manifest.json").write_text(
        json.dumps(dict(
            version="v121",
            locked_protocol_sha256=sha(locked),
            study="GSE214965",
            source_files=source_rows,
            model_training=False,
            model_evaluation=False,
            sample_mapping_status="NOT_YET_VERIFIED",
            script_sha256=sha(__file__),
            python=sys.version,
            platform=platform.platform(),
            no_git=True
        ),indent=2)+"\n",
        encoding="utf-8"
    )

    shutil.copy2(
        __file__,OUT/"script_snapshot_v121.py"
    )

    say(
        "[GATE] Source files and table structure "
        "verified; sample mapping and batch "
        "splits NOT YET VERIFIED"
    )
    say(
        "[COMPLETE] Third-study source "
        "audit and frozen protocol archived"
    )

def finish(ok):
    say("[RESULT] "+("SUCCESS" if ok else "FAILED"))

    archive = OUT/"portable_report_v121.zip"
    say("[REPORT] "+str(OUT/"console_summary.txt"))
    say("[ZIP] "+str(archive))

    (OUT/"console_summary.txt").write_text(
        "\n".join(LOG)+"\n",
        encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "06_sha256_inventory.csv",archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    csvout(
        "06_sha256_inventory.csv",
        items,
        ["filename","bytes","sha256"]
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED,
        compresslevel=5
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p,p.name)

        for name in FILES:
            p = DATA/name
            valid,_ = gzip_valid(p)
            if valid:
                z.write(
                    p,"source_data/"+p.name,
                    compress_type=zipfile.ZIP_STORED
                )

    say(
        "[ZIP COMPLETE] Frozen protocol, "
        "original files, code, SHA256 and log"
    )

if __name__ == "__main__":
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say("[ERROR]\n"+traceback.format_exc())

    try:
        finish(ok)
    except Exception:
        ok = False
        print(
            "[ARCHIVE ERROR]\n"+traceback.format_exc(),
            flush=True
        )

    if not ok:
        sys.exit(1)
