#!/usr/bin/env python3
"""V118: evidence audit of frozen V117 external results.
No model fitting, retuning, downloads or Git.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, sys, shutil, traceback, zipfile, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_v118_external_evidence_audit" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
METHODS = (
    "vehicle_reference_only",
    "paired_effect_fixed",
    "paired_effect_ridge_amplitude"
)
KEY = [
    "cell_line", "holdout_replicate",
    "target_drug", "true_duration"
]
FULLKEY = KEY + ["source_drug"]

def say(message):
    message = str(message)
    print(message, flush=True)
    LOG.append(message)

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def save(name, data):
    df = (
        data if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data)
    )
    df.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | {} rows".format(name, len(df)))
    return df

def load():
    base = ROOT / "paper7_v117_external_locked_pilot"
    required = (
        "13_sha256_inventory.csv",
        "12_manifest.json",
        "03_external_predictions.csv",
        "04_adjacent_pair_predictions.csv",
        "05_external_method_metrics.csv",
        "08_paired_discordances.csv",
        "02_verified_sample_labels.csv",
        "01_frozen_external_adaptation.json",
        "script_snapshot_v117.py"
    )

    runs = sorted(
        (
            r for r in base.glob("run_*")
            if r.is_dir()
            and all((r / n).is_file() for n in required)
        ),
        reverse=True
    )
    require(runs, "Complete V117 run not found")
    run = runs[0]

    inv = pd.read_csv(
        run / "13_sha256_inventory.csv",
        encoding="utf-8-sig"
    )
    require(
        {"filename","bytes","sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "Invalid V117 checksum inventory"
    )

    for r in inv.itertuples(index=False):
        name = str(r.filename)
        p = run / name
        require(
            Path(name).name == name
            and p.is_file()
            and p.stat().st_size == int(r.bytes)
            and digest(p) == str(r.sha256),
            "Archived file changed: " + name
        )

    manifest = json.loads(
        (run / "12_manifest.json").read_text("utf-8")
    )
    require(
        manifest.get("version") == "v117"
        and manifest.get("n_predictions") == 810
        and manifest.get("n_sample_labels") == 144,
        "V117 manifest mismatch"
    )

    pred = pd.read_csv(
        run / "03_external_predictions.csv",
        encoding="utf-8-sig"
    )
    pairs = pd.read_csv(
        run / "04_adjacent_pair_predictions.csv",
        encoding="utf-8-sig"
    )
    metric = pd.read_csv(
        run / "05_external_method_metrics.csv",
        encoding="utf-8-sig"
    )
    discord = pd.read_csv(
        run / "08_paired_discordances.csv",
        encoding="utf-8-sig"
    )
    labels = pd.read_csv(
        run / "02_verified_sample_labels.csv",
        encoding="utf-8-sig"
    )

    require(
        (
            len(pred), len(pairs), len(metric),
            len(discord), len(labels)
        ) == (810,1296,12,4,144),
        "Unexpected V117 table sizes"
    )

    for col in ("correct","top2_contains_truth"):
        pred[col] = (
            pred[col].astype(str)
            .str.strip().str.lower().eq("true")
        )

    pairs["correct"] = (
        pairs.correct.astype(str)
        .str.strip().str.lower().eq("true")
    )

    require(
        set(pred.method) == set(METHODS)
        and set(pred.cell_line) == {"MCF7","LNCaP"},
        "Unexpected methods or cell lines"
    )
    require(
        set(pred.true_duration) == {3,6,9,12,24}
        and set(pred.holdout_replicate) == {"A","B","C"},
        "Unexpected time or replicate labels"
    )
    require(
        not pred.duplicated(
            FULLKEY + ["method"]
        ).any(),
        "Duplicate predictions"
    )
    require(
        not labels.duplicated([
            "cell_line","condition","hours","replicate"
        ]).any(),
        "Duplicate sample labels"
    )

    say(
        "[V117 VERIFIED] {} checksum-verified artifacts "
        "| {} predictions | {} pair decisions".format(
            len(inv),len(pred),len(pairs)
        )
    )
    return run,pred,pairs,metric,discord,labels

def audit(run,pred,pairs,metric,discord,labels):
    queries = pred[KEY].drop_duplicates()

    require(
        len(queries) == 90,
        "Expected 90 unique active samples"
    )

    sizes = (
        pred.groupby(
            KEY + ["method","transfer"]
        ).size().reset_index(name="n_predictions")
    )

    require(
        (
            sizes[sizes.transfer=="same_drug"]
            .n_predictions == 1
        ).all(),
        "Unexpected same-drug duplicates"
    )
    require(
        (
            sizes[sizes.transfer=="cross_drug"]
            .n_predictions == 2
        ).all(),
        "Cross-drug must use two source templates"
    )
    require(
        len(sizes) == 540,
        "Unexpected grouped prediction counts"
    )

    say(
        "[UNIT AUDIT] 90 unique active samples | "
        "90 same-drug predictions/method | "
        "180 cross-drug predictions/method "
        "from two source drugs per sample"
    )

    selected_labels = labels[
        labels.condition.isin(("T","M","W"))
        & labels.hours.isin((3,6,9,12,24))
    ]
    require(
        len(selected_labels) == 90,
        "Sample-label count mismatch"
    )

    metrics = []
    for (transfer,method),g in pred.groupby([
        "transfer","method"
    ]):
        expected = (
            90 if transfer=="same_drug"
            else 180
        )
        require(
            len(g) == expected,
            "Transfer denominator mismatch"
        )

        archived = metric[
            (metric.transfer==transfer)
            & (metric.method==method)
        ]
        require(
            len(archived)==2
            and int(archived.correct.sum())
                ==int(g.correct.sum()),
            "V117 metric replay mismatch"
        )

        metrics.append(dict(
            transfer=transfer,
            method=method,
            n_predictions=len(g),
            unique_query_samples=len(
                g[KEY].drop_duplicates()
            ),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.abs_error_hours.mean()
            ),
            top2_descriptive=float(
                g.top2_contains_truth.mean()
            )
        ))

    save(
        "02_corrected_denominators_and_metrics.csv",
        metrics
    )

    comparisons = []

    for transfer in ("same_drug","cross_drug"):
        g = pred[pred.transfer==transfer]

        if transfer=="same_drug":
            agg = g[
                KEY+["method","correct"]
            ].copy()
            agg["fraction_correct"] = (
                agg.correct.astype(float)
            )
        else:
            agg = (
                g.groupby(
                    KEY+["method"],as_index=False
                ).correct.agg(
                    fraction_correct="mean"
                )
            )

        wide = agg.pivot(
            index=KEY,
            columns="method",
            values="fraction_correct"
        )

        require(
            len(wide)==90
            and set(wide.columns)==set(METHODS)
            and wide.notna().all().all(),
            "Unique-query comparison incomplete"
        )

        comparisons_to_make = (
            ("paired_effect_ridge_amplitude",
             "vehicle_reference_only"),
            ("paired_effect_fixed",
             "vehicle_reference_only"),
            ("paired_effect_fixed",
             "paired_effect_ridge_amplitude")
        )

        for left,right in comparisons_to_make:
            for keys,row in wide.iterrows():
                delta = float(
                    row[left]-row[right]
                )
                comparisons.append(dict(
                    transfer=transfer,
                    cell_line=keys[0],
                    replicate=keys[1],
                    target_drug=keys[2],
                    true_duration=keys[3],
                    method_A=left,
                    method_B=right,
                    score_A=float(row[left]),
                    score_B=float(row[right]),
                    fractional_gain=delta,
                    better=delta>1e-12,
                    worse=delta< -1e-12,
                    equal=abs(delta)<=1e-12
                ))

    paired = save(
        "03_unique_sample_paired_differences.csv",
        comparisons
    )

    for transfer in ("same_drug","cross_drug"):
        for left,right in comparisons_to_make:
            g = paired[
                (paired.transfer==transfer)
                & (paired.method_A==left)
                & (paired.method_B==right)
            ]
            say(
                "[PAIRED UNIQUE | {} | {} vs {}] "
                "queries={} | mean gain={:+.4f} "
                "| better={} worse={} ties={}"
                .format(
                    transfer,left,right,len(g),
                    g.fractional_gain.mean(),
                    int(g.better.sum()),
                    int(g.worse.sum()),
                    int(g.equal.sum())
                )
            )

    for r in discord.itertuples(index=False):
        g = pred[
            (pred.cell_line==r.cell_line)
            & (pred.transfer==r.transfer)
        ]

        pivot = g.pivot(
            index=FULLKEY,
            columns="method",
            values="correct"
        )

        a = pivot[
            "paired_effect_ridge_amplitude"
        ]
        b = pivot[
            "vehicle_reference_only"
        ]

        require(
            len(pivot)==int(r.n)
            and int((a&~b).sum())==int(r.ridge_only)
            and int((~a&b).sum())==int(r.vehicle_only)
            and int((a&b).sum())==int(r.both)
            and int((~a&~b).sum())==int(r.neither),
            "Archived paired discordance mismatch"
        )

    say(
        "[DISCORDANCE REPLAY] "
        "4/4 archived paired comparisons"
    )

    clusters = []

    for (
        cell,rep,transfer,method
    ),g in pred.groupby([
        "cell_line","holdout_replicate",
        "transfer","method"
    ]):
        expected = (
            15 if transfer=="same_drug"
            else 30
        )
        require(
            len(g)==expected,
            "Incorrect replicate-stratum size"
        )

        clusters.append(dict(
            cell_line=cell,
            replicate=rep,
            transfer=transfer,
            method=method,
            n_predictions=len(g),
            n_unique_samples=len(
                g[KEY].drop_duplicates()
            ),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.abs_error_hours.mean()
            )
        ))

    cluster = save(
        "04_cell_replicate_strata.csv",
        clusters
    )

    for transfer in ("same_drug","cross_drug"):
        a = cluster[
            (cluster.transfer==transfer)
            & (
                cluster.method==
                "paired_effect_ridge_amplitude"
            )
        ].set_index([
            "cell_line","replicate"
        ]).accuracy

        b = cluster[
            (cluster.transfer==transfer)
            & (
                cluster.method==
                "vehicle_reference_only"
            )
        ].set_index([
            "cell_line","replicate"
        ]).accuracy

        delta = a-b

        say(
            "[STRATUM SENSITIVITY | {}] "
            "ridge gain positive in {}/6 "
            "cell-by-replicate strata; "
            "min={:+.3f}; max={:+.3f}"
            .format(
                transfer,
                int((delta>0).sum()),
                delta.min(),delta.max()
            )
        )

    per_time = []
    for (
        cell,transfer,method,hour
    ),g in pred.groupby([
        "cell_line","transfer",
        "method","true_duration"
    ]):
        per_time.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            true_duration=hour,
            n_predictions=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.abs_error_hours.mean()
            )
        ))

    save(
        "05_per_duration_results.csv",
        per_time
    )

    per_drug = []
    for (
        cell,transfer,method,drug
    ),g in pred.groupby([
        "cell_line","transfer",
        "method","target_drug"
    ]):
        per_drug.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            target_drug=drug,
            n_predictions=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean()),
            MAE_hours=float(
                g.abs_error_hours.mean()
            )
        ))

    save(
        "06_per_drug_results.csv",
        per_drug
    )

    adjacent = []
    for (
        cell,transfer,method,pair
    ),g in pairs.groupby([
        "cell_line","transfer",
        "method","pair"
    ]):
        adjacent.append(dict(
            cell_line=cell,
            transfer=transfer,
            method=method,
            pair=pair,
            n_predictions=len(g),
            correct=int(g.correct.sum()),
            accuracy=float(g.correct.mean())
        ))

    save(
        "07_adjacent_pair_resolution.csv",
        adjacent
    )

    for transfer in ("same_drug","cross_drug"):
        for method in METHODS:
            g = pairs[
                (pairs.transfer==transfer)
                & (pairs.method==method)
            ]
            note = " ".join(
                "{}:{}/{}".format(
                    name,int(sub.correct.sum()),len(sub)
                )
                for name,sub in g.groupby("pair")
            )
            say(
                "[PAIR RESOLUTION | {} | {}] {}"
                .format(transfer,method,note)
            )

    say(
        "[NEGATIVE GATE] No independently "
        "evaluated DMSO rejection gate "
        "on GSE149428"
    )
    say(
        "[LEAKAGE BOUNDARY] Author-normalized "
        "plate-adjusted expression precedes "
        "the heldout splits"
    )
    say(
        "[METHOD BOUNDARY] Frozen architecture, "
        "but gene selection and templates "
        "refitted on each study"
    )

    for src,dest in (
        (
            "03_external_predictions.csv",
            "input_v117_predictions.csv"
        ),
        (
            "04_adjacent_pair_predictions.csv",
            "input_v117_pairs.csv"
        ),
        (
            "08_paired_discordances.csv",
            "input_v117_discordance.csv"
        ),
        (
            "12_manifest.json",
            "input_v117_manifest.json"
        ),
        (
            "02_verified_sample_labels.csv",
            "input_v117_sample_labels.csv"
        )
    ):
        shutil.copy2(run/src,OUT/dest)

def main():
    say("="*88)
    say(
        "PAPER 7 | V118 | EXTERNAL EVIDENCE AUDIT: "
        "UNIQUE SAMPLES / DRUGS / TIME"
    )
    say(
        "NO REFITTING | NO MORE DATA | "
        "NO POSTHOC METHOD SELECTION | NO GIT"
    )
    say("="*88)

    require(
        len(sys.argv)==1,
        "Use python script.py with no flags"
    )

    run,pred,pairs,metric,discord,labels = load()
    audit(
        run,pred,pairs,metric,discord,labels
    )

    (
        OUT/"01_protocol.json"
    ).write_text(
        json.dumps(dict(
            version="v118",
            purpose=(
                "Post-result evidence audit; "
                "not new validation"
            ),
            v117_manifest_sha256=digest(
                run/"12_manifest.json"
            ),
            script_sha256=digest(__file__),
            methods=METHODS,
            no_new_models=True,
            no_new_data=True,
            inference="descriptive only",
            no_git=True
        ),indent=2)+"\n",
        encoding="utf-8"
    )

    notes = """# V118 — External evidence audit

GSE149428 is a separate published study,
but V117 trains gene selections and templates
on its training subsets.

There are 90 unique active query samples:
two cell lines, three drugs, five times,
and three replicate labels.

For cross-drug testing, the same query sample
receives predictions from two different source
drugs. The 180 decisions are dependent.

A/B/C are experimental replicates,
not verified independent biological batches.

Author preprocessing included normalization
and adjustment for plate covariates before
our holdout splits.

The fixed-response and ridge variants were
specified before V117. Choosing a new primary
variant after examining V117 would be posthoc.

No independent DMSO rejection gate has been
evaluated in this external study. Testing
DMSO against itself as its own matched
control would be circular.

This is exposure-duration inference,
not independently observed molecular onset.

No 95% coverage, confirmatory p-values,
or methodological novelty claim is supported.
"""

    (
        OUT/"08_methods_and_cautions.md"
    ).write_text(
        notes,encoding="utf-8"
    )

    shutil.copy2(
        __file__,
        OUT/"script_snapshot_v118.py"
    )

    (
        OUT/"09_manifest.json"
    ).write_text(
        json.dumps(dict(
            version="v118",
            script_sha256=digest(__file__),
            v117_manifest_sha256=digest(
                run/"12_manifest.json"
            ),
            n_unique_active_samples=90,
            n_cross_predictions_per_method=180,
            independent_batches_verified=False,
            author_processing_precedes_split=True,
            new_training=False,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform()
        ),indent=2)+"\n",
        encoding="utf-8"
    )

    say(
        "[RESULT BOUNDARY] Stronger same-drug result; "
        "cross-drug advantage less established; "
        "biological onset and novelty unproven"
    )
    say("[COMPLETE] Evidence audit archived")

def finish(ok):
    say(
        "[RESULT] "+(
            "SUCCESS" if ok else "FAILED"
        )
    )
    say("[REPORT] "+str(
        OUT/"console_summary.txt"
    ))

    archive = OUT/"portable_report_v118.zip"
    say("[ZIP] "+str(archive))

    (
        OUT/"console_summary.txt"
    ).write_text(
        "\n".join(LOG)+"\n",
        encoding="utf-8"
    )

    entries = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "10_sha256_inventory.csv",archive.name
        ):
            entries.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(entries).to_csv(
        OUT/"10_sha256_inventory.csv",
        index=False,encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED,
        compresslevel=5
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

if __name__=="__main__":
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
