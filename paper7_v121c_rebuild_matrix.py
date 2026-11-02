#!/usr/bin/env python3
"""Paper 7 V121c: verify original DRUG-seq counts with unlabeled gene-ID field."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from collections import Counter
import csv, gzip, json, sys, platform, shutil, traceback, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "GSE214965"
OUT = ROOT / "paper7_v121c_verified_matrix" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
META = "GSE214965_sample_metadata_LN229_DRUG_Seq.txt.gz"
RAW = "GSE214965_rawcounts_LN229_DRUG_Seq.txt.gz"
EXPECTED = {
    META: "89a69aa0a3d8fc25147bd1009aaebb5621ac67af73e5ef33f7204349a2ace28d",
    RAW: "82446902882766ce0ae8aff69ba42edf8951b7f974125fb52a7c9b56de0e5db6"
}
LOCK = "14864a8291e089e97897aba40fb3d68c57d9faa48fa24cbd488d3d74c6b530fb"

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()

def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | rows={}".format(name, len(frame)))
    return frame

def sources():
    for name, expected_hash in EXPECTED.items():
        path = SRC / name
        need(
            path.is_file() and sha(path) == expected_hash,
            "Missing or modified source: " + name
        )
        say("[SOURCE VERIFIED] " + name + " SHA256=" + expected_hash)

    parent = ROOT / "paper7_v121_third_study_lock"
    candidates = sorted(
        (
            r / "01_third_study_locked_protocol.json"
            for r in parent.glob("run_*")
            if (r / "01_third_study_locked_protocol.json").is_file()
        ),
        reverse=True
    )
    need(candidates, "V121 freeze not found")

    lock = next(
        (p for p in candidates if sha(p) == LOCK),
        None
    )
    need(lock is not None, "V121 frozen protocol digest changed")

    original = json.loads(lock.read_text("utf-8"))
    need(
        original.get("version") == "v121"
        and original.get("external_dataset") == "GSE214965"
        and original.get("gene_count") == 200
        and original.get("ridge_per_gene") == 0.25
        and original.get("variance_floor") == 0.25,
        "Frozen architecture mismatch"
    )

    shutil.copy2(lock, OUT / "00_original_frozen_protocol.json")
    say("[MODEL FREEZE VERIFIED] SHA256=" + LOCK)

def load_metadata():
    m = pd.read_csv(
        SRC / META, compression="gzip",
        sep="\t", dtype=str, keep_default_na=False
    )
    required = (
        "sample_column_name", "well_barcode",
        "library", "drug", "timepoint",
        "condition", "count_sum", "replicate"
    )
    need(
        len(m) == 160 and set(required).issubset(m.columns),
        "Metadata sample count or schema mismatch"
    )
    need(
        m.sample_column_name.is_unique
        and m.sample_column_name.str.len().gt(0).all(),
        "Duplicate or empty sample names"
    )
    say("[METADATA VERIFIED] 160 samples | 11 columns")
    return m

def read_raw(metadata):
    genes = []
    numeric = []

    with gzip.open(
        SRC / RAW, "rt", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        need(
            len(header) == 160 and len(set(header)) == 160,
            "Raw-count header is not 160 distinct sample labels"
        )
        need(
            set(header) == set(metadata.sample_column_name),
            "Raw-count sample labels do not match metadata"
        )
        say(
            "[RAW HEADER] 160 sample names; "
            "gene identifier column is unlabeled"
        )

        for rowno, row in enumerate(reader, start=2):
            need(
                len(row) == 161,
                "Row {}: expected gene ID + 160 counts; found {} fields"
                .format(rowno, len(row))
            )
            gene = row[0].strip()
            need(gene != "", "Empty gene identifier")

            try:
                values = np.asarray(row[1:], dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    "Invalid count at row {}: {}".format(rowno, exc)
                )

            need(
                np.isfinite(values).all()
                and (values >= 0).all()
                and np.all(values == np.floor(values)),
                "Noninteger, negative or missing counts at row {}".format(rowno)
            )
            genes.append(gene)
            numeric.append(values)

            if len(genes) % 10000 == 0:
                say("[RAW PARSED] {} gene rows".format(len(genes)))

    need(
        len(genes) == 22760,
        "Expected 22760 gene rows; found {}".format(len(genes))
    )

    values = np.stack(numeric, axis=0)
    need(values.shape == (22760, 160), "Invalid matrix shape")
    need(
        values.max() < np.iinfo(np.int32).max,
        "Counts exceed int32 range"
    )
    counts = values.astype(np.int32)
    need(
        (counts.sum(axis=0) > 0).all(),
        "One or more zero-count samples"
    )

    repeated = sum(
        n - 1 for n in Counter(genes).values() if n > 1
    )

    say(
        "[MATRIX VERIFIED] 22760 genes x 160 samples "
        "| noninteger=0 | duplicate gene IDs={}"
        .format(repeated)
    )
    return header, genes, counts

def summarize(metadata, header, genes, counts):
    ordered = (
        metadata.set_index("sample_column_name", drop=False)
        .loc[header].reset_index(drop=True)
    )
    need(
        ordered.sample_column_name.tolist() == header,
        "Sample ordering mismatch"
    )

    libraries = counts.sum(axis=0, dtype=np.int64)
    ordered["matrix_sum"] = libraries
    ordered["nonzero_genes"] = (counts > 0).sum(axis=0)

    declared = pd.to_numeric(
        ordered.count_sum, errors="coerce"
    )
    ordered["metadata_count_sum_numeric"] = declared
    ordered["matrix_to_metadata_sum_ratio"] = np.where(
        np.isfinite(declared) & (declared > 0),
        libraries / declared,
        np.nan
    )

    save("01_verified_sample_mapping_and_QC.csv", ordered)
    save(
        "02_gene_ids.csv",
        pd.DataFrame({
            "gene_row": np.arange(len(genes)),
            "gene_id": genes
        })
    )

    design = (
        ordered.groupby(
            ["library", "drug", "timepoint"], dropna=False
        )
        .agg(
            n_samples=("sample_column_name", "size"),
            n_replicates=("replicate", "nunique"),
            min_assigned_reads=("matrix_sum", "min"),
            max_assigned_reads=("matrix_sum", "max")
        )
        .reset_index()
    )
    save("03_library_drug_time_design.csv", design)

    by_time = (
        ordered.groupby(["library", "timepoint"], dropna=False)
        .agg(
            n_samples=("sample_column_name", "size"),
            n_drugs=("drug", "nunique"),
            n_replicate_labels=("replicate", "nunique")
        )
        .reset_index()
    )
    save("04_library_time_summary.csv", by_time)

    dmso = ordered[
        ordered.drug.str.strip().str.upper().eq("DMSO")
    ]
    save("05_DMSO_samples.csv", dmso)

    say(
        "[DESIGN] libraries={} | times={} | distinct drugs={}"
        .format(
            sorted(ordered.library.unique().tolist()),
            sorted(ordered.timepoint.unique().tolist()),
            ordered.drug.nunique()
        )
    )
    say(
        "[REPLICATES] labels={}".format(
            sorted(ordered.replicate.unique().tolist())
        )
    )
    say(
        "[DESIGN] library-drug-time groups={} "
        "| samples/group={}"
        .format(
            len(design),
            design.n_samples.value_counts().sort_index().to_dict()
        )
    )
    say(
        "[DMSO] total={} | by library/time={}"
        .format(
            len(dmso),
            dmso.groupby(["library", "timepoint"]).size().to_dict()
        )
    )
    say(
        "[COUNTS QC] min={:,} median={:,.0f} max={:,} per sample"
        .format(
            int(libraries.min()),
            float(np.median(libraries)),
            int(libraries.max())
        )
    )

    present = np.isfinite(declared)
    if present.any():
        differences = np.abs(
            libraries[present]
            - declared[present].to_numpy(dtype=float)
        )
        say(
            "[METADATA COUNT AUDIT] exact sums={}/{} "
            "| median absolute difference={:.1f}"
            .format(
                int((differences == 0).sum()),
                int(present.sum()),
                float(np.median(differences))
            )
        )

    norm = np.log2(
        1 + counts.astype(np.float64)
        * 1e6 / libraries[None, :]
    )

    np.savez_compressed(
        OUT / "06_verified_counts_log2CPM.npz",
        counts=counts,
        log2CPM=norm.astype(np.float32),
        gene_ids=np.array(genes, dtype=str),
        sample_columns=np.array(header, dtype=str)
    )

    say(
        "[NORMALIZATION] log2(1+CPM) saved; "
        "original raw counts unchanged"
    )

    return dict(
        n_gene_rows=len(genes),
        n_samples=len(header),
        n_library_drug_time_groups=len(design),
        n_DMSO_samples=len(dmso),
        n_libraries=ordered.library.nunique(),
        min_library_count=int(libraries.min()),
        duplicate_gene_ids=len(genes) - len(set(genes))
    )

def main():
    say("=" * 85)
    say("PAPER 7 | V121C | GSE214965 RAW COUNTS: GENE-ID HEADER REPAIR")
    say(
        "OFFLINE | NO DROPPED GENES | "
        "NO FITTING | NO GIT"
    )
    say("=" * 85)

    need(len(sys.argv) == 1, "Run filename only")

    sources()
    metadata = load_metadata()
    header, genes, counts = read_raw(metadata)
    info = summarize(metadata, header, genes, counts)

    (OUT / "07_provenance.json").write_text(
        json.dumps(dict(
            version="v121c",
            original_sources=EXPECTED,
            frozen_V121_SHA256=LOCK,
            raw_format=(
                "160 sample header names; "
                "161 fields per gene row"
            ),
            repair=(
                "Assign first field to gene_id; "
                "do not modify source or drop rows"
            ),
            training_performed=False,
            validation_status="NOT_RUN",
            biological_onset_truth=False,
            design=info,
            script_sha256=sha(__file__),
            python=sys.version,
            pandas=pd.__version__,
            numpy=np.__version__,
            platform=platform.platform()
        ), indent=2) + "\n",
        encoding="utf-8"
    )

    (OUT / "08_methods_limitations.md").write_text(
        """# Paper 7 V121c

Original count file: 160 sample labels in the header,
followed by 22760 gene rows with one unnamed gene ID
and 160 raw integer counts.

All 160 labels must match the metadata exactly.
No gene rows are dropped or imputed.

The original source files are unchanged.

log2(1+CPM) is a transparent preprocessing candidate,
not an equivalent of DESeq2 VST normalization.

Libraries and replicate labels are audited, but
biological independence is not assumed.

The V120 architecture remains frozen.
No classification or validation is performed here.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        __file__, OUT / "script_snapshot_v121c.py"
    )

    say(
        "[GATE] Matrix and sample mapping verified; "
        "independent evaluation NOT YET RUN"
    )
    say("[COMPLETE] V121c matrix and design audit archived")

def finish(ok):
    say("[RESULT] " + ("SUCCESS" if ok else "FAILED"))

    archive = OUT / "portable_report_v121c.zip"
    say("[REPORT] " + str(OUT / "console_summary.txt"))
    say("[ZIP] " + str(archive))

    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8"
    )

    inventory = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "09_sha256_inventory.csv", archive.name
        ):
            inventory.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(inventory).to_csv(
        OUT / "09_sha256_inventory.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED,
        compresslevel=5
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for name in (META, RAW):
            z.write(
                SRC / name,
                "original_source/" + name,
                compress_type=zipfile.ZIP_STORED
            )

    say(
        "[ZIP COMPLETE] Original data, verified matrix, "
        "metadata, code, logs and SHA256"
    )

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
