from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import csv, io, json, re, shutil, sys, time
import traceback, urllib.request, zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "GSE149428" / "elife"
OUT = ROOT / "paper7_v116c_elife_external_audit" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
FILES = {
    "MCF7": "elife-52707-data1-v2.csv.zip",
    "LNCaP": "elife-52707-data3-v2.csv.zip"
}
CDN = "https://cdn.elifesciences.org/articles/52707/"
LOCK_SHA = "a3043faf9edffd4d1b38d8d0b6b0d04ad24bc6bd3f8620840dec52b4a5e3dfff"
TIMES = (0, 3, 6, 9, 12, 24)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def digest(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def csvout(name, cols, rows):
    with (OUT / name).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=cols, extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)
    say("[OUTPUT] {} | rows={}".format(name, len(rows)))

def freeze():
    base = ROOT / "paper7_v116_external_lock"
    runs = sorted(
        (
            p for p in base.glob("run_*")
            if (p / "01_frozen_external_protocol.json").is_file()
            and (p / "input_v115_script_snapshot.py").is_file()
        ),
        reverse=True
    )
    need(runs, "Original V116 locked protocol not found")

    p = runs[0] / "01_frozen_external_protocol.json"
    need(
        digest(p) == LOCK_SHA,
        "Original V116 lock SHA256 changed"
    )

    settings = json.loads(p.read_text("utf-8"))
    need(
        settings.get("status") == "FROZEN_BEFORE_EXTERNAL_EXPRESSION_AUDIT"
        and settings.get("gene_count") == 200
        and settings.get("variance_floor") == 0.25
        and settings.get("ridge_penalty_per_gene") == 0.25,
        "Frozen settings changed"
    )

    shutil.copy2(p, OUT / "01_original_frozen_protocol.json")
    shutil.copy2(
        runs[0] / "input_v115_script_snapshot.py",
        OUT / "02_frozen_v115_script.py"
    )
    say("[LOCK VERIFIED] original V116 SHA256=" + LOCK_SHA)

def zipvalid(p):
    if not p.is_file() or p.stat().st_size < 100000:
        return False
    try:
        with zipfile.ZipFile(p) as z:
            return bool(z.namelist()) and z.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False

def download(cell, name):
    dest = DATA / name
    if zipvalid(dest):
        say(
            "[CACHE VERIFIED] {} | {:.2f} MiB | SHA256={}".format(
                name, dest.stat().st_size / 1048576,
                digest(dest)
            )
        )
        return dest

    url = CDN + name
    part = DATA / (name + ".partial")

    for trial in (1, 2):
        try:
            say(
                "[DOWNLOAD eLife] {} | attempt={}/2 | {}"
                .format(cell, trial, url)
            )

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 Paper7Research/1.0",
                    "Accept": "application/zip,application/octet-stream,*/*"
                }
            )

            with urllib.request.urlopen(
                req, timeout=120
            ) as src, part.open("wb") as dst:
                say(
                    "[HTTP] status={} | type={} | final={}".format(
                        src.status,
                        src.headers.get("Content-Type", ""),
                        src.url
                    )
                )
                lead = src.read(64)
                say("[HTTP FIRST BYTES] " + repr(lead[:48]))

                need(
                    lead.startswith(b"PK\x03\x04"),
                    "Response is not a ZIP; no fabricated extension"
                )

                dst.write(lead)
                total = len(lead)
                mark = 5 * 1048576

                while True:
                    buf = src.read(1048576)
                    if not buf:
                        break
                    dst.write(buf)
                    total += len(buf)

                    if total >= mark:
                        say(
                            "[PROGRESS] {} | {:.1f} MiB"
                            .format(name, total / 1048576)
                        )
                        mark += 5 * 1048576

            need(
                zipvalid(part),
                "ZIP CRC/integrity check failed: " + name
            )
            part.replace(dest)

            say(
                "[SOURCE VERIFIED] {} | {:.2f} MiB | SHA256={}"
                .format(
                    name,
                    dest.stat().st_size / 1048576,
                    digest(dest)
                )
            )
            return dest

        except Exception as e:
            say(
                "[DOWNLOAD ERROR] {}: {}".format(
                    type(e).__name__, str(e)[:250]
                )
            )
            if part.exists():
                part.unlink()
            if trial < 2:
                time.sleep(2)

    raise RuntimeError("eLife download unavailable: " + name)

def label(s):
    normalized = re.sub(
        "[^A-Z0-9]", "", str(s).upper()
    )
    m = re.search(
        r"(DMSO|T|M|W)(0|3|6|9|12|24)([ABC])$",
        normalized
    )
    if not m:
        return None
    return (m.group(1), int(m.group(2)), m.group(3))

def inspect(cell, path):
    files, headers, candidates = [], [], []
    csv.field_size_limit(2**26)

    with zipfile.ZipFile(path) as z:
        need(z.testzip() is None, "Corrupt ZIP: " + path.name)
        members = z.infolist()
        say("[ZIP MEMBERS | {}] {}".format(cell, len(members)))

        for info in members:
            if info.is_dir():
                continue

            files.append(dict(
                cell_line=cell,
                source_archive=path.name,
                member=info.filename,
                uncompressed_bytes=info.file_size,
                compressed_bytes=info.compress_size
            ))

            say(
                "[MEMBER | {}] {} | {:.1f} MiB uncompressed"
                .format(
                    cell, info.filename,
                    info.file_size / 1048576
                )
            )

            if not info.filename.lower().endswith(
                (".csv", ".tsv", ".txt")
            ):
                continue

            delimiter = (
                "\t" if info.filename.lower().endswith(".tsv")
                else ","
            )

            with z.open(info) as raw:
                reader = csv.reader(
                    io.TextIOWrapper(
                        raw,
                        encoding="utf-8-sig",
                        errors="replace",
                        newline=""
                    ),
                    delimiter=delimiter
                )

                for rowno in range(1, 6):
                    try:
                        row = next(reader)
                    except StopIteration:
                        break

                    matched = []
                    for ix, value in enumerate(row):
                        signature = label(value)
                        if signature is not None:
                            matched.append(
                                (ix, value, signature)
                            )

                    headers.append(dict(
                        cell_line=cell,
                        member=info.filename,
                        row=rowno,
                        width=len(row),
                        matched_sample_codes=len(matched),
                        first_values=" | ".join(
                            str(v)[:60] for v in row[:12]
                        ),
                        last_values=" | ".join(
                            str(v)[:40] for v in row[-6:]
                        )
                    ))

                    say(
                        "[HEADER | {} | row {}] width={} "
                        "| sample codes={} | first: {}".format(
                            cell, rowno, len(row),
                            len(matched),
                            " ; ".join(
                                str(x)[:35] for x in row[:10]
                            )
                        )
                    )

                    signatures = [a[2] for a in matched]
                    unique = set(signatures)
                    expected = {
                        (drug, hour, rep)
                        for drug in ("DMSO", "T", "M", "W")
                        for hour in TIMES
                        for rep in "ABC"
                    }

                    for col, name, signature in matched:
                        candidates.append(dict(
                            cell_line=cell,
                            member=info.filename,
                            row=rowno,
                            column_index=col,
                            header=name,
                            condition=signature[0],
                            hours=signature[1],
                            replicate=signature[2]
                        ))

                    if (
                        len(unique & expected) == len(expected)
                        and len(signatures) == len(unique)
                    ):
                        say(
                            "[PANEL VERIFIED | {}] 72/72 "
                            "DMSO/T/M/W timepoints by column label "
                            "(header row {})".format(
                                cell, rowno
                            )
                        )

        need(files, "No archive members in " + path.name)

    return files, headers, candidates

def main():
    say("=" * 80)
    say(
        "PAPER 7 | V116C | eLIFE PUBLISHER MIRROR SOURCE INGEST"
    )
    say(
        "V115 LOCKED | NO EXPRESSION MODEL FITTING | NO GIT"
    )
    say("=" * 80)

    need(
        len(sys.argv) == 1,
        "Run with python filename only"
    )

    freeze()
    allfiles, headers, matches, paths = [], [], [], []

    for cell, name in FILES.items():
        p = download(cell, name)
        paths.append(p)
        a, b, c = inspect(cell, p)
        allfiles += a
        headers += b
        matches += c

    csvout(
        "03_archive_members.csv",
        [
            "cell_line", "source_archive", "member",
            "uncompressed_bytes", "compressed_bytes"
        ],
        allfiles
    )

    csvout(
        "04_expression_header_audit.csv",
        [
            "cell_line", "member", "row", "width",
            "matched_sample_codes", "first_values",
            "last_values"
        ],
        headers
    )

    csvout(
        "05_candidate_sample_columns.csv",
        [
            "cell_line", "member", "row", "column_index",
            "header", "condition", "hours", "replicate"
        ],
        matches
    )

    panels = []

    for cell in FILES:
        rows = [
            r for r in headers
            if r["cell_line"] == cell
        ]
        best = max(
            (r["matched_sample_codes"] for r in rows),
            default=0
        )

        panels.append(dict(
            cell_line=cell,
            max_header_code_matches=best,
            needs_mapping_confirmation=best < 72
        ))

        status = (
            "CANDIDATE_COMPLETE"
            if best >= 72 else "NEEDS_REVIEW"
        )
        say(
            "[MAPPING SCREEN | {}] max matching "
            "header cells={}; status={}".format(
                cell, best, status
            )
        )

    csvout(
        "06_mapping_screen.csv",
        [
            "cell_line", "max_header_code_matches",
            "needs_mapping_confirmation"
        ],
        panels
    )

    manifest = dict(
        version="v116c",
        original_frozen_V116_SHA256=LOCK_SHA,
        data_provenance=(
            "eLife 2020 Diaz et al "
            "10.7554/eLife.52707 version 2"
        ),
        files={
            p.name: dict(
                url=CDN + p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            )
            for p in paths
        },
        external_predictions=0,
        models_retrained=False,
        source_script_sha256=digest(__file__),
        sample_mapping_status=(
            "UNVERIFIED_UNTIL_GEO_CROSSCHECK"
        )
    )

    (OUT / "07_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8"
    )
    shutil.copy2(
        __file__, OUT / "script_snapshot_v116c.py"
    )

    say(
        "[GATE] Publisher archives verified; "
        "GEO GSM alignment and expression scale "
        "still to verify"
    )
    say("[RESULT] SUCCESS")

def finish(ok):
    if not ok:
        say("[RESULT] FAILED")

    say("[REPORT] " + str(OUT / "console_summary.txt"))
    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n",
        encoding="utf-8"
    )

    archive = OUT / "portable_report_v116c.zip"
    inventory = []

    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "08_sha256_inventory.csv", archive.name
        ):
            inventory.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    csvout(
        "08_sha256_inventory.csv",
        ["filename", "bytes", "sha256"],
        inventory
    )

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED,
        compresslevel=3
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for p in sorted(DATA.iterdir()):
            if (
                p.is_file()
                and p.name in FILES.values()
                and zipvalid(p)
            ):
                z.write(
                    p,
                    "eLife_source/" + p.name,
                    compress_type=zipfile.ZIP_STORED
                )

    say("[ZIP] " + str(archive))

if __name__ == "__main__":
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say("[ERROR]\n" + traceback.format_exc())

    try:
        finish(ok)
    except Exception:
        ok = False
        print(
            "[ARCHIVE ERROR]\n" + traceback.format_exc(),
            flush=True
        )

    if not ok:
        sys.exit(1)
