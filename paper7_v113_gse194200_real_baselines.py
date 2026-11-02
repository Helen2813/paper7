#!/usr/bin/env python3
"""Paper 7 V113: real count extraction and held-out baselines."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, math, platform, re, shutil, sys, traceback, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "GSE194200"
OUT = ROOT / "paper7_gse194200_real_baseline_v113" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
XLSX_NAME = "GSE194200_Kalwat_TotalRNAseq_processeddata.xlsx"
SERIES_NAME = "GSE194200_series_matrix.txt.gz"
XLSX_SHA = "7d220b4ef8e02deda74c1f51295e9636634796a086580323ee315e58d8f5ea7b"
SERIES_SHA = "7aef2271b0d193a059cf95ab0356d32ea3b643495fdf97d7f01267d5f24f6543"
ACTIVE = ("SW016789", "Thapsigargin")
CLASSES = (1, 2, 6, 24)
RAW_START = 87
N_SAMPLES = 39
TOP_GENES = 200

def say(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)

def need(test, message):
    if not bool(test):
        raise RuntimeError(message)

def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def save(name, data):
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | rows={}".format(name, len(df)))
    return df

def elemname(s):
    return s.rsplit("}", 1)[-1]

def col_number(s):
    value = 0
    for char in s:
        value = value * 26 + ord(char) - ord("A") + 1
    return value

def col_letter(n):
    s = ""
    while n:
        n, k = divmod(n - 1, 26)
        s = chr(k + 65) + s
    return s

def canon(s):
    return re.sub("[^a-z0-9]", "", str(s).casefold())

def find_v112():
    xlsx = DATA / XLSX_NAME
    matrix = DATA / SERIES_NAME

    need(
        xlsx.is_file() and matrix.is_file(),
        "V112 files missing from data/GSE194200"
    )
    need(sha(xlsx) == XLSX_SHA, "XLSX SHA256 mismatch")
    need(sha(matrix) == SERIES_SHA, "GEO matrix SHA256 mismatch")

    folder = ROOT / "paper7_gse194200_ingest_v112"
    runs = sorted(
        (
            r for r in folder.glob("run_*")
            if (r / "01_GEO_sample_metadata.csv").is_file()
            and (r / "05_source_manifest.json").is_file()
        ),
        reverse=True
    )
    need(runs, "V112 sample metadata and manifest not found")
    r = runs[0]

    man = json.loads(
        (r / "05_source_manifest.json").read_text("utf-8")
    )
    need(
        man.get("version") == "v112"
        and man.get("sample_count") == 39,
        "V112 manifest mismatch"
    )

    for name, value in (
        (XLSX_NAME, XLSX_SHA),
        (SERIES_NAME, SERIES_SHA)
    ):
        need(
            man["downloaded"][name]["sha256"] == value,
            "V112 source differs: " + name
        )

    meta = pd.read_csv(
        r / "01_GEO_sample_metadata.csv",
        encoding="utf-8-sig",
        dtype={
            "gsm": str,
            "title": str,
            "description": str
        }
    )

    need(
        len(meta) == 39
        and meta.gsm.is_unique
        and meta.title.is_unique,
        "Incomplete GEO metadata"
    )

    for exp in (1, 2, 3):
        sub = meta[meta.experiment == exp]
        expected = {("Baseline", 0)} | {
            (t, h)
            for t in ("DMSO",) + ACTIVE
            for h in CLASSES
        }
        need(
            len(sub) == 13
            and set(zip(sub.treatment, sub.hours)) == expected,
            "Experimental design mismatch"
        )

    say(
        "[V112 VERIFIED] both original SHA256 hashes "
        "and 39 GEO sample labels"
    )
    return xlsx, r, meta

def shared_strings(z):
    result = []
    path = "xl/sharedStrings.xml"

    if path not in z.namelist():
        return result

    with z.open(path) as f:
        for _, element in ET.iterparse(f, events=("end",)):
            if elemname(element.tag) == "si":
                result.append("".join(
                    n.text or ""
                    for n in element.iter()
                    if elemname(n.tag) == "t"
                ))
                element.clear()

    say("[XLSX STRINGS] {}".format(len(result)))
    return result

def counts_sheet(z):
    book = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(
        z.read("xl/_rels/workbook.xml.rels")
    )

    targets = {
        r.attrib["Id"]: r.attrib["Target"]
        for r in rels
        if "Id" in r.attrib
    }

    for element in book.iter():
        if (
            elemname(element.tag) != "sheet"
            or element.attrib.get("name") != "Counts"
        ):
            continue

        rid = next(
            (
                v for k, v in element.attrib.items()
                if elemname(k) == "id"
            ),
            None
        )
        need(rid in targets, "Counts relationship missing")

        target = targets[rid]
        name = (
            target.lstrip("/")
            if target.startswith("/")
            else "xl/" + target.lstrip("/")
        )
        name = re.sub(r"(^|/)\./", r"\1", name)

        need(
            name in z.namelist(),
            "Counts worksheet XML missing"
        )
        return name

    raise RuntimeError("Counts worksheet not found")

def cell_value(c, strings):
    typ = c.attrib.get("t", "")

    if typ == "inlineStr":
        return "".join(
            x.text or ""
            for x in c.iter()
            if elemname(x.tag) == "t"
        )

    element = next(
        (
            x for x in c
            if elemname(x.tag) == "v"
        ),
        None
    )

    if element is None or element.text is None:
        return None

    value = element.text

    if typ == "s":
        ix = int(value)
        need(
            0 <= ix < len(strings),
            "Invalid shared-string index"
        )
        return strings[ix]

    if typ == "e":
        raise RuntimeError("Excel error: " + value)

    return value

def extract(xlsx, metadata):
    head = {}
    counts = []
    annotation = []
    missing = 0
    fractional = 0

    with zipfile.ZipFile(xlsx) as z:
        strings = shared_strings(z)
        part = counts_sheet(z)

        say("[XLSX SOURCE] {}".format(part))

        with z.open(part) as fp:
            for _, row in ET.iterparse(
                fp, events=("end",)
            ):
                if elemname(row.tag) != "row":
                    continue

                r = int(row.attrib.get("r", -1))
                cells = {}

                for c in row:
                    if elemname(c.tag) != "c":
                        continue

                    match = re.fullmatch(
                        r"([A-Z]+)(\d+)",
                        c.attrib.get("r", "")
                    )
                    if match:
                        cells[col_number(
                            match.group(1)
                        )] = cell_value(c, strings)

                if r in (1, 2, 3):
                    head[r] = cells

                elif r >= 4:
                    values = [
                        cells.get(i)
                        for i in range(
                            RAW_START,
                            RAW_START + N_SAMPLES
                        )
                    ]

                    if any(
                        v is None or str(v).strip() == ""
                        for v in values
                    ):
                        missing += 1
                        if missing <= 3:
                            say(
                                "[INVALID RAW ROW] Excel row {}"
                                .format(r)
                            )
                        row.clear()
                        continue

                    try:
                        nums = np.asarray(
                            values, dtype=np.float64
                        )
                    except (ValueError, TypeError) as e:
                        raise RuntimeError(
                            "Non-numeric counts at row {}: {}"
                            .format(r, e)
                        )

                    need(
                        np.isfinite(nums).all()
                        and (nums >= 0).all(),
                        "Invalid counts at Excel row {}"
                        .format(r)
                    )

                    fractional += int(np.count_nonzero(
                        np.abs(nums - np.rint(nums)) > 1e-6
                    ))

                    counts.append(nums)
                    annotation.append(dict(
                        workbook_row=r,
                        gene_symbol=str(
                            cells.get(1) or ""
                        ),
                        entrez_id=str(
                            cells.get(7) or ""
                        ),
                        gene_name=str(
                            cells.get(8) or ""
                        )
                    ))

                    if len(counts) % 5000 == 0:
                        say(
                            "[XLSX COUNTS] {} genes read"
                            .format(len(counts))
                        )

                row.clear()

    need(
        missing == 0,
        "{} gene rows have blank raw counts"
        .format(missing)
    )
    need(
        len(counts) == 24421,
        "Expected 24421 genes, found {}"
        .format(len(counts))
    )
    need(
        all(i in head for i in (1, 2, 3)),
        "Missing Excel header rows"
    )

    need(
        canon(head[1].get(RAW_START, "")) == "rawcounts",
        "Raw counts do not begin at CI"
    )
    need(
        canon(head[1].get(9, "")) == "log2fpkm",
        "FPKM header mismatch"
    )
    need(
        canon(head[1].get(48, "")) == "tpm",
        "TPM header mismatch"
    )

    mappings = []
    used = set()

    for c in range(
        RAW_START,
        RAW_START + N_SAMPLES
    ):
        title = head[2].get(c)
        machine = head[3].get(c)

        need(
            title and machine,
            "Missing raw-count sample header"
        )

        found = metadata[
            metadata.title.map(canon) == canon(title)
        ]

        need(
            len(found) == 1,
            "Nonunique GEO title match: {}"
            .format(title)
        )

        rec = found.iloc[0]

        need(
            canon(machine) == canon(rec.description),
            "Machine ID mismatch: {} vs {}"
            .format(machine, rec.description)
        )
        need(
            rec.gsm not in used,
            "Duplicate GSM mapping"
        )

        used.add(rec.gsm)

        mappings.append(dict(
            raw_column_index=c,
            excel_column=col_letter(c),
            gsm=rec.gsm,
            title=rec.title,
            machine_id=str(machine),
            experiment=int(rec.experiment),
            treatment=str(rec.treatment),
            hours=int(rec.hours)
        ))

    need(
        len(used) == 39,
        "Not all GEO samples matched"
    )

    matrix = np.vstack(counts)

    need(
        matrix.shape == (24421, 39),
        "Matrix shape mismatch"
    )
    need(
        (matrix.sum(axis=0) > 0).all(),
        "Zero library size"
    )

    say(
        "[MAPPING VERIFIED] 39/39 XLSX sample "
        "title AND machine identifier match GEO"
    )
    say(
        "[RAW COUNTS] {} genes x {} samples; "
        "fractional cells={}"
        .format(*matrix.shape, fractional)
    )

    return (
        matrix,
        pd.DataFrame(mappings),
        pd.DataFrame(annotation),
        fractional
    )

def normalize(counts, samples, fractional):
    libraries = counts.sum(axis=0)
    cpm = counts * (
        1000000.0 / libraries[np.newaxis, :]
    )
    X = np.log2(1.0 + cpm)

    qc = samples[
        ["gsm", "experiment", "treatment", "hours"]
    ].copy()

    qc["library_gene_counts"] = libraries
    qc["detected_genes_count_ge10"] = (
        counts >= 10
    ).sum(axis=0)
    qc["detected_genes_nonzero"] = (
        counts > 0
    ).sum(axis=0)
    qc["median_log2CPM"] = np.median(
        X, axis=0
    )

    save("03_sample_QC.csv", qc)

    np.savez_compressed(
        OUT / "04_gene_counts_and_logCPM.npz",
        counts=counts.astype(np.float64),
        log2CPM=X.astype(np.float32)
    )

    say(
        "[QC] library assigned counts "
        "min={:.0f}, median={:.0f}, max={:.0f}"
        .format(
            libraries.min(),
            np.median(libraries),
            libraries.max()
        )
    )

    if fractional:
        say(
            "[QC WARNING] Fractional raw counts present; "
            "do not interpret as integer reads"
        )

    return X

def fit_predict(X, raw, meta, heldout):
    exp = meta.experiment.to_numpy(int)
    drug = meta.treatment.astype(str).to_numpy()
    hours = meta.hours.to_numpy(int)

    train = np.where(
        (exp != heldout) &
        np.isin(drug, ACTIVE)
    )[0]

    test = np.where(
        (exp == heldout) &
        np.isin(drug, ACTIVE)
    )[0]

    controls = np.where(
        (exp == heldout) &
        (drug == "DMSO")
    )[0]

    need(
        len(train) == 16
        and len(test) == 8
        and len(controls) == 4,
        "Fold split mismatch"
    )
    need(
        len(set(exp[train])) == 2
        and len(set(exp[test])) == 1,
        "Experimental-group leakage"
    )

    # Only training samples influence gene selection.
    present = (
        (raw[:, train] >= 10).sum(axis=1) >= 3
    )
    variance = np.var(
        X[:, train], axis=1
    )

    eligible = np.flatnonzero(
        present &
        np.isfinite(variance) &
        (variance > 1e-10)
    )

    need(
        len(eligible) >= TOP_GENES,
        "Insufficient training-only eligible genes"
    )

    order = np.argsort(
        -variance[eligible],
        kind="stable"
    )
    selected = eligible[
        order[:TOP_GENES]
    ]

    mu = X[selected][:, train].mean(axis=1)
    sd = X[selected][:, train].std(axis=1)
    scale = np.maximum(sd, 0.5)

    Z = (
        (X[selected].T - mu) / scale
    )

    centroids = {}

    for k in CLASSES:
        take = train[hours[train] == k]

        need(
            len(take) == 4,
            "Pooled training duration incomplete"
        )

        centroids[("pooled", k)] = (
            Z[take].mean(axis=0)
        )

        for treatment in ACTIVE:
            subset = take[
                drug[take] == treatment
            ]

            need(
                len(subset) == 2,
                "Treatment-specific duration incomplete"
            )

            centroids[(treatment, k)] = (
                Z[subset].mean(axis=0)
            )

    results = []

    for i in list(test) + list(controls):
        for method in (
            "constant_2h",
            "pooled_centroid",
            "drug_aware_centroid"
        ):
            if (
                method == "drug_aware_centroid"
                and drug[i] == "DMSO"
            ):
                continue

            if method == "constant_2h":
                prediction = 2
                distance = float("nan")
            else:
                group = (
                    "pooled"
                    if method == "pooled_centroid"
                    else drug[i]
                )

                distances = {
                    k: float(np.sum(
                        (
                            Z[i] -
                            centroids[(group, k)]
                        ) ** 2
                    ))
                    for k in CLASSES
                }

                prediction = min(
                    CLASSES,
                    key=lambda k: (distances[k], k)
                )
                distance = distances[prediction]

            results.append(dict(
                gsm=meta.iloc[i].gsm,
                heldout_experiment=heldout,
                treatment=drug[i],
                true_exposure_hours=int(hours[i]),
                method=method,
                predicted_hours=int(prediction),
                correct=bool(
                    prediction == hours[i]
                ),
                absolute_error_hours=abs(
                    prediction - int(hours[i])
                ),
                log2_duration_absolute_error=abs(
                    math.log2(
                        prediction / int(hours[i])
                    )
                ),
                selected_genes=(
                    TOP_GENES
                    if method != "constant_2h"
                    else 0
                ),
                min_centroid_squared_distance=distance,
                is_active=bool(
                    drug[i] in ACTIVE
                )
            ))

    return results, selected, len(eligible)

def baseline(X, counts, samples, genes):
    rows = []
    features = []

    for heldout in (1, 2, 3):
        predictions, selected, eligible = fit_predict(
            X, counts, samples, heldout
        )

        rows.extend(predictions)

        features.extend(
            dict(
                heldout_experiment=heldout,
                feature_rank=rank + 1,
                workbook_row=int(
                    genes.iloc[ix].workbook_row
                ),
                gene_symbol=str(
                    genes.iloc[ix].gene_symbol
                ),
                entrez_id=str(
                    genes.iloc[ix].entrez_id
                )
            )
            for rank, ix in enumerate(selected)
        )

        active = [
            r for r in predictions
            if r["is_active"]
        ]

        for name in (
            "constant_2h",
            "pooled_centroid",
            "drug_aware_centroid"
        ):
            sub = [
                r for r in active
                if r["method"] == name
            ]

            say(
                "[FOLD {}] {:21s} {:d}/{:d} correct "
                "| MAE={:.3f} hours"
                .format(
                    heldout,
                    name,
                    sum(r["correct"] for r in sub),
                    len(sub),
                    np.mean([
                        r["absolute_error_hours"]
                        for r in sub
                    ])
                )
            )

        say(
            "[FOLD {}] eligible genes={} | "
            "selected top variance={}"
            .format(
                heldout,
                eligible,
                len(selected)
            )
        )

    predictions = save(
        "05_heldout_sample_predictions.csv",
        rows
    )
    save(
        "06_train_only_gene_selections.csv",
        features
    )

    active = predictions[
        predictions.is_active == True
    ]

    need(
        len(active) == 72,
        "Incomplete heldout predictions"
    )

    metrics = []

    for (method, treatment), g in active.groupby(
        ["method", "treatment"]
    ):
        metrics.append(dict(
            method=method,
            treatment=treatment,
            n=len(g),
            accuracy=float(g.correct.mean()),
            correct=int(g.correct.sum()),
            MAE_hours=float(
                g.absolute_error_hours.mean()
            ),
            mean_abs_log2_duration_error=float(
                g.log2_duration_absolute_error.mean()
            )
        ))

    for method, g in active.groupby("method"):
        metrics.append(dict(
            method=method,
            treatment="ALL_ACTIVE",
            n=len(g),
            accuracy=float(g.correct.mean()),
            correct=int(g.correct.sum()),
            MAE_hours=float(
                g.absolute_error_hours.mean()
            ),
            mean_abs_log2_duration_error=float(
                g.log2_duration_absolute_error.mean()
            )
        ))

        say(
            "[OVERALL] {} | accuracy={}/24 ({:.1f}%) "
            "| MAE={:.3f}h"
            .format(
                method,
                int(g.correct.sum()),
                100 * g.correct.mean(),
                g.absolute_error_hours.mean()
            )
        )

    save(
        "07_heldout_metrics.csv",
        metrics
    )

    confusion = (
        active.groupby([
            "method",
            "true_exposure_hours",
            "predicted_hours"
        ])
        .size()
        .reset_index(name="n")
    )

    save(
        "08_confusion_counts.csv",
        confusion
    )

    negative = predictions[
        (predictions.treatment == "DMSO") &
        (predictions.method == "pooled_centroid")
    ]

    need(
        len(negative) == 12,
        "DMSO control predictions incomplete"
    )

    say(
        "[DMSO PROBE] pooled classifier correctly "
        "predicts DMSO duration in {}/12 "
        "(not a no-delay truth label)"
        .format(int(negative.correct.sum()))
    )

def main():
    say("=" * 89)
    say(
        "PAPER 7 | V113 | REAL RAW COUNTS + "
        "LEAVE-ONE-EXPERIMENT-OUT BASELINES"
    )
    say(
        "39 REAL SAMPLES | NO SYNTHETIC SHIFTS "
        "| NO GIT | TRAIN-ONLY FEATURE SELECTION"
    )
    say("=" * 89)

    need(
        len(sys.argv) == 1,
        "Run Python with filename only"
    )

    xlsx, v112, metadata = find_v112()
    counts, sample_map, gene_info, fractional = (
        extract(xlsx, metadata)
    )

    save(
        "01_verified_sample_mapping.csv",
        sample_map
    )
    save(
        "02_gene_annotations.csv",
        gene_info
    )

    X = normalize(
        counts, sample_map, fractional
    )

    baseline(
        X, counts, sample_map, gene_info
    )

    protocol = dict(
        version="v113",
        source_geo="GSE194200",
        samples=39,
        time_classes=CLASSES,
        active_treatments=ACTIVE,
        raw_block="CI:DU",
        raw_xlsx_sha256=XLSX_SHA,
        geo_matrix_sha256=SERIES_SHA,
        folds="leave one entire experiment out",
        training_samples_per_fold=16,
        test_active_samples_per_fold=8,
        feature_filter=(
            ">=10 counts in >=3 training "
            "active samples"
        ),
        feature_selection=(
            "top 200 train-only gene variances "
            "on log2(1+CPM)"
        ),
        scaling="training mean/std, std floor=0.5",
        baseline_methods=[
            "always 2h",
            "pooled duration centroid",
            "drug-aware duration centroid"
        ],
        no_inner_tuning=True,
        true_target=(
            "known exposure duration, "
            "NOT true transcription onset"
        ),
        negatives=(
            "DMSO time probe; "
            "NOT a known no-delay label"
        ),
        no_95pct_coverage_claim=True,
        new_method_validated=False,
        script_sha256=sha(__file__),
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
        original_v112_run=str(v112),
        no_git=True,
        no_network=True
    )

    (OUT / "09_protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8"
    )

    (OUT / "10_interpretation.md").write_text(
        """# GSE194200 V113

Prediction target:
elapsed experimental drug exposure duration,
1, 2, 6 or 24 hours.

This is NOT known biological transcription onset.

Three independent experiment groups.
Each outer fold withholds an entire group.

Training-only gene filtering, selection and scaling.

Drug-aware baseline knows the treatment identity.
Pooled baseline does not require treatment identity.

DMSO observations probe generic time-related signal.
They are not known negative biological delay labels.

No inner hyperparameter tuning.
No method novelty claimed.
No distribution-free 95% coverage claim.

The next step is to evaluate whether a separately
specified method improves on these frozen baselines
under the same group holdouts.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT / "script_snapshot_v113.py"
    )
    shutil.copy2(
        v112 / "01_GEO_sample_metadata.csv",
        OUT / "input_v112_geo_metadata.csv"
    )

    say(
        "[GATE] Real heldout duration baselines only; "
        "biological onset NOT VALIDATED"
    )
    say(
        "[COMPLETE] Raw counts and first "
        "real-data benchmark archived"
    )

def finish(success):
    say(
        "[RESULT] " + (
            "SUCCESS" if success else "FAILED"
        )
    )
    say(
        "[REPORT] " +
        str(OUT / "console_summary.txt")
    )

    archive = OUT / "portable_report_v113.zip"
    say("[ZIP] " + str(archive))

    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n",
        encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "11_sha256_inventory.csv",
            archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT / "11_sha256_inventory.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=3
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for p in (
            DATA / XLSX_NAME,
            DATA / SERIES_NAME
        ):
            if p.exists():
                z.write(
                    p,
                    "source_data/" + p.name
                )

    say(
        "[ZIP COMPLETE] source data, "
        "processed counts, code and predictions"
    )

if __name__ == "__main__":
    success = True

    try:
        main()
    except Exception:
        success = False
        say(
            "[ERROR]\n" +
            traceback.format_exc()
        )

    try:
        finish(success)
    except Exception:
        success = False
        print(
            "[ARCHIVE ERROR]\n" +
            traceback.format_exc(),
            flush=True
        )

    if not success:
        sys.exit(1)
