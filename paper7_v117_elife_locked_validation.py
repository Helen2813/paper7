#!/usr/bin/env python3
"""Paper 7 V117: external pilot of the frozen V115 model class."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, re, shutil, sys, traceback, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "GSE149428" / "elife"
OUT = ROOT / "paper7_v117_external_locked_pilot" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)
LOG = []

FILES = {
    "MCF7": (
        "elife-52707-data1-v2.csv.zip",
        "SourceDataFile1mcf7_log_cpm.csv",
        "3b3b5e99030b6a1cfe7fe0354c6ffc9a3596b428c429bb81662402305a2fc84f"
    ),
    "LNCaP": (
        "elife-52707-data3-v2.csv.zip",
        "SourceDataFile3lncap_expression.csv",
        "f9430447d9f2a5ba80a171ad34f4e97257ba28a5501b2b3c2a589d8ff683642c"
    )
}

CONDITIONS = {
    "MCF7": ("DMSO","T","M","W","TM","TW","MW"),
    "LNCaP": ("DMSO","T","M","W","TM","TW","MW","TMW")
}
DRUGS = ("T","M","W")
REPS = ("A","B","C")
ALL_TIMES = (0,3,6,9,12,24)
TIMES = (3,6,9,12,24)
METHODS = (
    "vehicle_reference_only",
    "paired_effect_fixed",
    "paired_effect_ridge_amplitude"
)
PAIRS = ((3,6),(6,9),(9,12),(12,24))

TOP = 200
VAR_FLOOR = 0.25
RIDGE = 0.25
LOCK_SHA = (
    "a3043faf9edffd4d1b38d8d0b6b0d04ad24bc6bd3f8620840dec52b4a5e3dfff"
)

PATTERN = re.compile(
    r"^(DMSO|TMW|TM|TW|MW|T|M|W)_(0|3|6|9|12|24)(A|B|C)$",
    re.I
)

def say(x):
    x = str(x)
    print(x, flush=True)
    LOG.append(x)

def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def digest(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def save(name, obj):
    x = obj if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
    x.to_csv(
        OUT/name,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g"
    )
    say("[OUTPUT] {} | {} rows".format(name,len(x)))
    return x

def locked_inputs():
    v116 = ROOT/"paper7_v116_external_lock"
    runs = sorted(
        (
            r for r in v116.glob("run_*")
            if r.is_dir()
            and (r/"01_frozen_external_protocol.json").is_file()
        ),
        reverse=True
    )
    need(runs,"Original V116 model freeze missing")

    lock = runs[0]/"01_frozen_external_protocol.json"
    need(
        digest(lock)==LOCK_SHA,
        "Frozen external protocol SHA256 mismatch"
    )

    rules = json.loads(lock.read_text("utf-8"))
    need(
        rules.get("gene_count")==TOP
        and rules.get("variance_floor")==VAR_FLOOR
        and rules.get("ridge_penalty_per_gene")==RIDGE
        and rules.get("external_study")=="GSE149428",
        "Frozen model settings differ"
    )

    shutil.copy2(
        lock,OUT/"input_v116_frozen_protocol.json"
    )

    need(
        all((DATA/n).is_file() for n,_,_ in FILES.values()),
        "Publisher archives absent; run V116c first"
    )

    say(
        "[LOCK VERIFIED] 200 genes | "
        "variance floor=0.25 | ridge=0.25"
    )
    return lock

def load_cell(cell):
    archive,member,expected_sha = FILES[cell]
    path = DATA/archive

    need(
        digest(path)==expected_sha,
        "Source SHA256 changed: "+archive
    )

    with zipfile.ZipFile(path) as z:
        need(
            z.testzip() is None,
            "Publisher ZIP CRC error: "+archive
        )
        need(
            z.namelist()==[member],
            "Unexpected source member: "+archive
        )
        with z.open(member) as f:
            df = pd.read_csv(
                f,index_col=0,
                keep_default_na=False,
                low_memory=False
            )

    df.columns = [
        str(x).strip() for x in df.columns
    ]
    genes = df.index.astype(str).tolist()

    need(
        len(genes)>=10000
        and not df.columns.duplicated().any(),
        "Expression file malformed: "+cell
    )

    pairs = {}
    for name in df.columns:
        m = PATTERN.fullmatch(name)
        need(
            m is not None,
            "Unexpected condition label: "+name
        )

        drug,hour,rep = m.groups()
        key = (
            drug.upper(),int(hour),rep.upper()
        )

        need(
            key not in pairs,
            "Duplicate cell/drug/time/rep: "+name
        )
        pairs[key] = name

    expected = {
        (d,t,r)
        for d in CONDITIONS[cell]
        for t in ALL_TIMES
        for r in REPS
    }

    need(
        set(pairs)==expected,
        "Missing/excess conditions for {}: "
        "missing={} excess={}".format(
            cell,
            sorted(expected-set(pairs)),
            sorted(set(pairs)-expected)
        )
    )

    need(
        len(df.columns)==(
            126 if cell=="MCF7" else 144
        ),
        "Unexpected expression sample count"
    )

    labels = [
        (d,t,r)
        for d in ("DMSO",)+DRUGS
        for t in ALL_TIMES
        for r in REPS
    ]

    selected_names = [
        pairs[k] for k in labels
    ]

    X = df.loc[
        :,selected_names
    ].to_numpy(dtype=np.float64).T

    need(
        X.shape==(72,len(genes))
        and np.isfinite(X).all(),
        "Invalid numeric log2CPM matrix: "+cell
    )

    need(
        (X<0).any(),
        "Unexpected logCPM scale: "+cell
    )

    idx = {
        k:i for i,k in enumerate(labels)
    }

    samples = [
        dict(
            cell_line=cell,
            condition=k[0],
            hours=k[1],
            replicate=k[2],
            publisher_column=pairs[k],
            used_for_prediction=(k[1] in TIMES)
        )
        for k in labels
    ]

    say(
        "[STRICT MAPPING | {}] {} header labels; "
        "72/72 exact DMSO/T/M/W; "
        "excluded combinations={} | genes={}"
        .format(
            cell,len(pairs),
            len(pairs)-72,len(genes)
        )
    )
    say(
        "[EXPRESSION | {}] author-adjusted logCPM "
        "min={:.3f} median={:.3f} max={:.3f}; "
        "duplicate gene labels={}"
        .format(
            cell,float(X.min()),
            float(np.median(X)),float(X.max()),
            len(genes)-len(set(genes))
        )
    )

    return X,idx,genes,samples

def trained_template(X,idx,source,hold):
    train_reps = tuple(
        r for r in REPS if r!=hold
    )

    v = np.stack([
        np.column_stack([
            X[idx[(source,t,r)]]
            -X[idx[("DMSO",t,r)]]
            for t in TIMES
        ])
        for r in train_reps
    ],axis=1)

    mean = v.mean(axis=1)
    between = (
        0.5*(v[:,0,:]-v[:,1,:])**2
    ).mean(axis=1)
    signal = np.var(mean,axis=1)

    with np.errstate(
        invalid="ignore",divide="ignore"
    ):
        ranks = signal/(between+VAR_FLOOR)

    eligible = np.flatnonzero(
        np.isfinite(ranks)
        &(signal>1e-12)
    )

    need(
        len(eligible)>=TOP,
        "Too few varying training genes"
    )

    selected = eligible[
        np.argsort(
            -ranks[eligible],kind="stable"
        )[:TOP]
    ]

    scales = np.sqrt(
        between[selected]+VAR_FLOOR
    )
    templates = mean[selected]

    return (
        selected,scales,templates,
        ranks,len(eligible)
    )

def scores_for(
    X,idx,selected,scales,
    templates,held,sample_idx
):
    obs = X[sample_idx,selected]
    scores = {
        m:{} for m in METHODS
    }
    amplitudes = {}
    lam = RIDGE*len(selected)

    for j,candidate in enumerate(TIMES):
        veh = X[
            idx[("DMSO",candidate,held)],
            selected
        ]

        residual = (obs-veh)/scales
        effect = templates[:,j]/scales

        scores[
            "vehicle_reference_only"
        ][candidate] = float(
            np.mean(residual**2)
        )

        scores[
            "paired_effect_fixed"
        ][candidate] = float(
            np.mean((residual-effect)**2)
        )

        amp = max(
            0.,
            float(
                residual@effect+lam
            )/float(effect@effect+lam)
        )

        scores[
            "paired_effect_ridge_amplitude"
        ][candidate] = float(
            (
                np.sum(
                    (residual-amp*effect)**2
                )+lam*(amp-1.)**2
            )/len(selected)
        )

        amplitudes[candidate] = amp

    return scores,amplitudes

def evaluate_cell(cell,X,idx,genes):
    decisions = []
    candidates = []
    features = []
    pairs = []

    for held in REPS:
        for source in DRUGS:
            (
                chosen,scale,templates,
                ranks,n_eligible
            ) = trained_template(
                X,idx,source,held
            )

            for j,g in enumerate(chosen):
                features.append(dict(
                    cell_line=cell,
                    holdout_replicate=held,
                    source_drug=source,
                    rank=j+1,
                    gene_row=int(g),
                    gene_name=genes[g],
                    training_only_rank_score=float(
                        ranks[g]
                    )
                ))

            for target in DRUGS:
                for true_hour in TIMES:
                    sample = idx[
                        (target,true_hour,held)
                    ]

                    scores,amps = scores_for(
                        X,idx,chosen,scale,
                        templates,held,sample
                    )

                    for method in METHODS:
                        ordered = sorted(
                            TIMES,
                            key=lambda t:(
                                scores[method][t],t
                            )
                        )
                        predicted = int(ordered[0])

                        decisions.append(dict(
                            cell_line=cell,
                            holdout_replicate=held,
                            source_drug=source,
                            target_drug=target,
                            transfer=(
                                "same_drug"
                                if source==target
                                else "cross_drug"
                            ),
                            test_sample=(
                                "{}_{}{}_{}".format(
                                    target,true_hour,
                                    held,cell
                                )
                            ),
                            true_duration=true_hour,
                            method=method,
                            predicted_duration=predicted,
                            correct=bool(
                                predicted==true_hour
                            ),
                            abs_error_hours=abs(
                                predicted-true_hour
                            ),
                            top2_contains_truth=bool(
                                true_hour in ordered[:2]
                            ),
                            best_score=float(
                                scores[method][ordered[0]]
                            ),
                            second_best_gap=float(
                                scores[method][ordered[1]]
                                -scores[method][ordered[0]]
                            )
                        ))

                        for t in TIMES:
                            candidates.append(dict(
                                cell_line=cell,
                                replicate=held,
                                source_drug=source,
                                target_drug=target,
                                true_duration=true_hour,
                                method=method,
                                candidate_hour=t,
                                score=scores[method][t],
                                amplitude=(
                                    amps[t]
                                    if method==
                                    "paired_effect_ridge_amplitude"
                                    else np.nan
                                )
                            ))

                        for a,b in PAIRS:
                            if true_hour not in (a,b):
                                continue

                            p = min(
                                (a,b),
                                key=lambda t:(
                                    scores[method][t],t
                                )
                            )
                            pairs.append(dict(
                                cell_line=cell,
                                replicate=held,
                                source_drug=source,
                                target_drug=target,
                                transfer=(
                                    "same_drug"
                                    if source==target
                                    else "cross_drug"
                                ),
                                pair="{}v{}".format(a,b),
                                method=method,
                                true_hour=true_hour,
                                predicted_hour=p,
                                correct=bool(
                                    p==true_hour
                                )
                            ))

            say(
                "[HOLDOUT | {} | {} | source={}] "
                "train=2 replicate labels; "
                "test=third; eligible genes={}"
                .format(
                    cell,held,source,n_eligible
                )
            )

    need(
        len(decisions)==405
        and len(candidates)==2025
        and len(features)==1800
        and len(pairs)==648,
        "Incomplete external comparison: "+cell
    )

    return decisions,candidates,features,pairs

def summarize(decisions,pairs):
    df = save(
        "03_external_predictions.csv",
        decisions
    )
    p = save(
        "04_adjacent_pair_predictions.csv",
        pairs
    )

    report = []

    for (
        cell,transfer,method
    ),sub in df.groupby([
        "cell_line","transfer","method"
    ]):
        n = len(sub)
        need(
            n==(
                45 if transfer=="same_drug"
                else 90
            ),
            "Unexpected result count"
        )

        report.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            n=n,
            correct=int(sub.correct.sum()),
            accuracy=float(sub.correct.mean()),
            MAE_hours=float(
                sub.abs_error_hours.mean()
            ),
            top2_coverage_descriptive=float(
                sub.top2_contains_truth.mean()
            )
        ))

        say(
            "[EXTERNAL | {} | {} | {}] "
            "correct={}/{} | MAE={:.3f}h"
            .format(
                cell,transfer,method,
                int(sub.correct.sum()),n,
                sub.abs_error_hours.mean()
            )
        )

    save(
        "05_external_method_metrics.csv",
        report
    )

    for (
        transfer,method
    ),sub in df.groupby([
        "transfer","method"
    ]):
        say(
            "[POOLED | {} | {}] correct={}/{} "
            "| MAE={:.3f}h "
            "(dependent predictions)"
            .format(
                transfer,method,
                int(sub.correct.sum()),
                len(sub),
                sub.abs_error_hours.mean()
            )
        )

    pair_rows = []
    for (
        cell,transfer,method,pair
    ),g in p.groupby([
        "cell_line","transfer","method","pair"
    ]):
        pair_rows.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            pair=pair,
            n=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean())
        ))

    save(
        "06_pair_resolution.csv",
        pair_rows
    )

    by_drug = []
    for (
        cell,transfer,method,drug
    ),g in df.groupby([
        "cell_line","transfer",
        "method","target_drug"
    ]):
        by_drug.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            target_drug=drug,
            n=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.abs_error_hours.mean()
            )
        ))

    save(
        "07_per_drug_performance.csv",
        by_drug
    )

    discord = []
    for (
        cell,transfer
    ),sub in df.groupby([
        "cell_line","transfer"
    ]):
        key = [
            "cell_line","holdout_replicate",
            "source_drug","target_drug",
            "true_duration"
        ]

        ridge = sub[
            sub.method==
            "paired_effect_ridge_amplitude"
        ].set_index(key)

        veh = sub[
            sub.method=="vehicle_reference_only"
        ].set_index(key)

        need(
            ridge.index.is_unique
            and set(ridge.index)==set(veh.index),
            "Paired sample mismatch"
        )

        a = ridge.correct.astype(bool)
        b = veh.reindex(
            ridge.index
        ).correct.astype(bool)

        discord.append(dict(
            cell_line=cell,
            transfer=transfer,
            n=len(a),
            ridge_only=int(
                (a & ~b).sum()
            ),
            vehicle_only=int(
                (~a & b).sum()
            ),
            both=int(
                (a & b).sum()
            ),
            neither=int(
                (~a & ~b).sum()
            )
        ))

    save(
        "08_paired_discordances.csv",
        discord
    )

def main():
    say("="*95)
    say(
        "PAPER 7 | V117 | GSE149428 EXTERNAL PILOT: "
        "PRE-FROZEN PAIRED-VEHICLE ARCHITECTURE"
    )
    say(
        "AUTHOR-NORMALIZED log2CPM | "
        "STRICT LABEL MAPPING | NO GIT"
    )
    say("="*95)

    need(
        len(sys.argv)==1,
        "Run without command-line flags"
    )

    lock = locked_inputs()

    protocol = dict(
        version="v117",
        source_study="GSE149428",
        original_lock_sha256=digest(lock),
        algorithm=(
            "V115 ridge amplitude plus "
            "matched vehicle baseline"
        ),
        source_archives={
            cell:dict(
                file=entry[0],
                sha256=entry[2]
            )
            for cell,entry in FILES.items()
        },
        inference_targets=list(TIMES),
        training_source_drugs=list(DRUGS),
        gene_count=TOP,
        regularizer=RIDGE,
        variance_floor=VAR_FLOOR,
        data_adaptation=(
            "Author-processed plate-adjusted log2CPM. "
            "Raw count eligibility from V115 "
            "cannot be reproduced. Use finite "
            "author-filtered rows with training-only "
            "nonzero temporal contrast variance."
        ),
        split=(
            "Leave one A/B/C replicate label out "
            "within each cell line"
        ),
        replication_warning=(
            "A/B/C are not demonstrated to be "
            "independent biological batches"
        ),
        holdout_reference=(
            "Five time-labelled DMSO samples "
            "from same heldout replicate"
        ),
        use_only_monotherapy=True,
        no_time_zero_prediction=True,
        time_zero_note=(
            "0h label denotes less than "
            "30 minutes after treatment"
        ),
        test_GSM_mapping=(
            "Not independently verified; "
            "publisher study labels matched exactly"
        ),
        no_test_label_tuning=True,
        logCPM_preprocessing_warning=(
            "Published TMM and plate adjustment "
            "may use all samples"
        ),
        biological_onset_truth=False,
        no_git=True,
        code_sha256=digest(__file__)
    )

    (
        OUT/"01_frozen_external_adaptation.json"
    ).write_text(
        json.dumps(protocol,indent=2)+"\n",
        encoding="utf-8"
    )

    all_samples = []
    all_predictions = []
    all_candidates = []
    all_features = []
    all_pairs = []

    for cell in FILES:
        X,idx,genes,samples = load_cell(cell)
        all_samples.extend(samples)

        d,c,f,p = evaluate_cell(
            cell,X,idx,genes
        )

        all_predictions.extend(d)
        all_candidates.extend(c)
        all_features.extend(f)
        all_pairs.extend(p)

    save(
        "02_verified_sample_labels.csv",
        all_samples
    )

    summarize(
        all_predictions,all_pairs
    )

    save(
        "09_training_only_gene_selections.csv",
        all_features
    )
    save(
        "10_all_candidate_scores.csv",
        all_candidates
    )

    (
        OUT/"11_methods_and_limitations.md"
    ).write_text(
        """# Paper 7 V117 — external pilot

Target:
Experimentally documented exposure duration,
NOT the onset of transcriptional response.

Source:
Diaz et al., eLife 2020, GSE149428.

Uses author-normalized log2CPM.
Conditions: DMSO, T, M, W.
Drug combinations excluded from primary test.

Candidate exposure durations:
3, 6, 9, 12 and 24 hours.

Published time zero represents less than
30 minutes of exposure, not perfect baseline.

Leave-one-replicate-letter-out is not
proven to be an independent biological
batch validation.

Requires five time-labelled DMSO controls
from the same heldout replicate group.

The control-only comparator receives
identical reference samples.

The original V115 count threshold cannot
be applied because publisher matrices
contain processed log2CPM, not raw counts.

Author normalization and plate correction
may involve samples from both training
and heldout folds.

This is an explicitly documented input
adaptation, not unchanged mouse model
coefficients or a strict raw-count replay.

No biological response onset truth is known.

No formal 95% coverage or significance
claims are made.

Repeated predictions sharing the same
test sample are dependent.

No DMSO classification gate is evaluated:
self-matching a vehicle query against the
same DMSO reference would be circular.

After examining V117 results, subsequent
changes require another unused dataset
for an independent validation claim.

Method novelty remains unestablished.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        __file__,
        OUT/"script_snapshot_v117.py"
    )

    (
        OUT/"12_manifest.json"
    ).write_text(
        json.dumps(dict(
            version="v117",
            original_frozen_sha256=LOCK_SHA,
            source_sha256={
                k:v[2] for k,v in FILES.items()
            },
            n_sample_labels=len(all_samples),
            n_predictions=len(all_predictions),
            n_external_test_classes=5,
            source_code_sha256=digest(__file__),
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            external_pilot=True,
            strict_no_preprocessing_leakage=False,
            independent_biological_batches_unverified=True,
            no_git=True
        ),indent=2)+"\n",
        encoding="utf-8"
    )

    say(
        "[BOUNDARY] External pilot; "
        "NO biological onset truth; "
        "A/B/C independence not established"
    )
    say(
        "[COMPLETE] Frozen architecture "
        "external pilot archived"
    )

def finish(ok):
    say(
        "[RESULT] "+(
            "SUCCESS" if ok else "FAILED"
        )
    )

    say(
        "[REPORT] "+str(
            OUT/"console_summary.txt"
        )
    )

    archive = OUT/"portable_report_v117.zip"
    say("[ZIP] "+str(archive))

    (
        OUT/"console_summary.txt"
    ).write_text(
        "\n".join(LOG)+"\n",
        encoding="utf-8"
    )

    records = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            archive.name,
            "13_sha256_inventory.csv"
        ):
            records.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(records).to_csv(
        OUT/"13_sha256_inventory.csv",
        index=False,encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED,
        compresslevel=3
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

        for cell,(name,_,_) in FILES.items():
            p = DATA/name
            if (
                p.is_file()
                and digest(p)==FILES[cell][2]
            ):
                z.write(
                    p,"publisher_source/"+name,
                    compress_type=zipfile.ZIP_STORED
                )

    say(
        "[ZIP COMPLETE] original source archives, "
        "code, predictions, log and SHA256"
    )

if __name__=="__main__":
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say(
            "[ERROR]\n"+traceback.format_exc()
        )

    try:
        finish(ok)
    except Exception:
        ok = False
        print(
            "[ARCHIVE ERROR]\n"+
            traceback.format_exc(),
            flush=True
        )

    if not ok:
        sys.exit(1)
