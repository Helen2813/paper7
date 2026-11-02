#!/usr/bin/env python3
"""V119: noncircular DMSO gate on GSE149428.
No heldout DMSO used as its own reference.
Exploratory, not independent validation.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, re, shutil, sys, traceback, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "GSE149428" / "elife"
OUT = ROOT / "paper7_v119_noncircular_vehicle_gate" / (
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
DRUGS = ("T", "M", "W")
TIMES = (3, 6, 9, 12, 24)
REPS = ("A", "B", "C")
MODES = (
    "training_time_matched",
    "training_pooled_vehicle"
)
TOP = 200
FLOOR = 0.25
RIDGE = 0.25

LOCK_SHA = (
    "a3043faf9edffd4d1b38d8d0b6b0d04ad24bc6bd3f8620840dec52b4a5e3dfff"
)
PATTERN = re.compile(
    r"^(DMSO|TMW|TM|TW|MW|T|M|W)_(0|3|6|9|12|24)(A|B|C)$",
    re.I
)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def save(name, obj):
    d = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    d.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | {} rows".format(name, len(d)))
    return d

def load_prior():
    base = ROOT / "paper7_v116_external_lock"
    locks = sorted(
        (
            r / "01_frozen_external_protocol.json"
            for r in base.glob("run_*")
            if (r / "01_frozen_external_protocol.json").is_file()
        ),
        reverse=True
    )
    need(
        locks and sha(locks[0]) == LOCK_SHA,
        "Original V116 lock changed"
    )
    lock = json.loads(locks[0].read_text("utf-8"))
    need(
        lock["gene_count"] == TOP
        and lock["variance_floor"] == FLOOR
        and lock["ridge_penalty_per_gene"] == RIDGE,
        "V115 settings changed"
    )

    base = ROOT / "paper7_v117_external_locked_pilot"
    files = (
        "13_sha256_inventory.csv",
        "12_manifest.json",
        "03_external_predictions.csv",
        "09_training_only_gene_selections.csv"
    )
    runs = sorted(
        (
            r for r in base.glob("run_*")
            if r.is_dir()
            and all((r / n).is_file() for n in files)
        ),
        reverse=True
    )
    need(runs, "V117 archive not found")
    r = runs[0]

    inv = pd.read_csv(
        r / "13_sha256_inventory.csv",
        encoding="utf-8-sig"
    )
    need(
        {"filename", "bytes", "sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "V117 inventory malformed"
    )
    for rec in inv.itertuples(index=False):
        name = str(rec.filename)
        p = r / name
        need(
            Path(name).name == name
            and p.is_file()
            and p.stat().st_size == int(rec.bytes)
            and sha(p) == str(rec.sha256),
            "V117 artifact changed: " + name
        )

    man = json.loads(
        (r / "12_manifest.json").read_text("utf-8")
    )
    need(
        man.get("version") == "v117"
        and man.get("n_predictions") == 810,
        "V117 manifest mismatch"
    )

    old = pd.read_csv(
        r / "03_external_predictions.csv",
        encoding="utf-8-sig"
    )
    features = pd.read_csv(
        r / "09_training_only_gene_selections.csv",
        encoding="utf-8-sig"
    )
    need(
        len(old) == 810 and len(features) == 3600,
        "V117 evidence incomplete"
    )
    old["correct"] = (
        old.correct.astype(str).str.lower().eq("true")
    )

    for method, expected in (
        ("paired_effect_fixed", 78),
        ("paired_effect_ridge_amplitude", 68),
        ("vehicle_reference_only", 30)
    ):
        g = old[
            (old.transfer == "same_drug")
            & (old.method == method)
        ]
        need(
            len(g) == 90
            and int(g.correct.sum()) == expected,
            "V117 metric differs: " + method
        )

    shutil.copy2(
        locks[0], OUT / "input_original_v116_lock.json"
    )
    say(
        "[V117 VERIFIED] {} files | fixed=78/90 "
        "| ridge=68/90 | vehicle=30/90"
        .format(len(inv))
    )
    return r, old, features

def load_cell(cell):
    archive, member, expected = FILES[cell]
    path = DATA / archive
    need(
        path.is_file() and sha(path) == expected,
        "Publisher SHA256 mismatch: " + cell
    )

    with zipfile.ZipFile(path) as z:
        need(
            z.testzip() is None and z.namelist() == [member],
            "Publisher ZIP invalid"
        )
        with z.open(member) as f:
            df = pd.read_csv(
                f, index_col=0,
                keep_default_na=False,
                low_memory=False
            )

    cols = [str(n).strip() for n in df.columns]
    need(
        len(cols) == (126 if cell == "MCF7" else 144)
        and len(set(cols)) == len(cols),
        "Source column mismatch"
    )

    labels = {}
    for name in cols:
        m = PATTERN.fullmatch(name)
        need(m is not None, "Unknown sample label: " + name)
        drug, hour, rep = m.groups()
        key = (drug.upper(), int(hour), rep.upper())
        need(key not in labels, "Duplicate sample")
        labels[key] = name

    keys = [
        (drug, hour, rep)
        for drug in ("DMSO",) + DRUGS
        for hour in (0,) + TIMES
        for rep in REPS
    ]
    need(
        all(k in labels for k in keys),
        "Incomplete DMSO/monotherapy grid"
    )

    X = df.loc[
        :, [labels[k] for k in keys]
    ].to_numpy(dtype=np.float64).T

    genes = df.index.astype(str).tolist()
    need(
        X.shape == (72, len(genes))
        and np.isfinite(X).all()
        and len(genes) >= 10000
        and len(set(genes)) == len(genes),
        "Expression values invalid"
    )

    index = {k: i for i, k in enumerate(keys)}
    say(
        "[CELL VERIFIED | {}] genes={} | "
        "72 labelled DMSO/monotherapy samples"
        .format(cell, len(genes))
    )
    return X, genes, index

def train(X, ix, cell, held, source, old_features):
    reps = tuple(r for r in REPS if r != held)

    v = np.stack([
        np.column_stack([
            X[ix[(source, t, r)]]
            - X[ix[("DMSO", t, r)]]
            for t in TIMES
        ])
        for r in reps
    ], axis=1)

    mean = v.mean(axis=1)
    between = (
        0.5 * (v[:, 0, :] - v[:, 1, :])**2
    ).mean(axis=1)
    signal = np.var(mean, axis=1)
    ranks = signal / (between + FLOOR)

    eligible = np.flatnonzero(
        np.isfinite(ranks) & (signal > 1e-12)
    )
    need(len(eligible) >= TOP, "Too few eligible genes")

    selected = eligible[
        np.argsort(
            -ranks[eligible],
            kind="stable"
        )[:TOP]
    ]

    previous = old_features[
        (old_features.cell_line == cell)
        & (old_features.holdout_replicate == held)
        & (old_features.source_drug == source)
    ].sort_values("rank")

    need(
        len(previous) == TOP
        and np.array_equal(
            previous.gene_row.to_numpy(int),
            selected
        ),
        "V117 feature-selection replay failed"
    )

    scale = np.sqrt(between[selected] + FLOOR)
    effect = mean[selected]

    vehicle = np.column_stack([
        np.mean([
            X[ix[("DMSO", t, r)], selected]
            for r in reps
        ], axis=0)
        for t in TIMES
    ])
    pooled = vehicle.mean(axis=1)

    return selected, scale, effect, vehicle, pooled

def score_query(
    X, sample, selected, scale,
    effect, vehicle, pooled, mode
):
    obs = X[sample, selected]
    null_scores = []
    active_scores = []
    amplitudes = []
    lam = RIDGE * len(selected)

    for j, hour in enumerate(TIMES):
        reference = (
            vehicle[:, j]
            if mode == "training_time_matched"
            else pooled
        )

        residual = (obs - reference) / scale
        response = effect[:, j] / scale

        null_score = float(
            np.mean(residual**2)
        )
        amplitude = max(
            0.,
            float(residual @ response + lam)
            / float(response @ response + lam)
        )
        active_score = float(
            (
                np.sum(
                    (residual - amplitude * response)**2
                )
                + lam * (amplitude - 1.)**2
            ) / len(selected)
        )

        null_scores.append(null_score)
        active_scores.append(active_score)
        amplitudes.append(amplitude)

    best_null = min(null_scores)
    best_active = min(active_scores)

    return dict(
        accept=best_active < best_null,
        guess=int(TIMES[int(np.argmin(active_scores))]),
        margin=best_null - best_active,
        best_null=best_null,
        best_active=best_active,
        null_scores=null_scores,
        active_scores=active_scores,
        amplitudes=amplitudes
    )

def evaluate(old, old_features):
    decisions = []
    candidate_scores = []
    selections = []

    historical = old[
        (old.transfer == "same_drug")
        & (
            old.method ==
            "paired_effect_ridge_amplitude"
        )
    ]
    need(len(historical) == 90, "V117 historical sample set missing")

    for cell in FILES:
        X, genes, ix = load_cell(cell)

        for held in REPS:
            for source in DRUGS:
                (
                    selected, scale, effect,
                    vehicle, pooled
                ) = train(
                    X, ix, cell, held,
                    source, old_features
                )

                for rank, g in enumerate(selected):
                    selections.append(dict(
                        cell_line=cell,
                        replicate=held,
                        source_drug=source,
                        rank=rank + 1,
                        gene_row=int(g),
                        gene_name=genes[g]
                    ))

                for target in ("DMSO",) + DRUGS:
                    for true_hour in TIMES:
                        sample = ix[
                            (target, true_hour, held)
                        ]

                        for mode in MODES:
                            result = score_query(
                                X, sample, selected,
                                scale, effect, vehicle,
                                pooled, mode
                            )

                            actual_active = target != "DMSO"
                            accepted = bool(result["accept"])

                            decisions.append(dict(
                                cell_line=cell,
                                replicate=held,
                                source_drug=source,
                                target_drug=target,
                                true_hour=true_hour,
                                reference_mode=mode,
                                same_drug=target == source,
                                is_DMSO=target == "DMSO",
                                accepted_active=accepted,
                                predicted_hour=result["guess"],
                                correct_time=(
                                    actual_active
                                    and result["guess"] == true_hour
                                ),
                                correct_and_accepted=(
                                    actual_active
                                    and accepted
                                    and result["guess"] == true_hour
                                ),
                                margin_vehicle_minus_active=(
                                    result["margin"]
                                ),
                                best_vehicle_score=(
                                    result["best_null"]
                                ),
                                best_active_score=(
                                    result["best_active"]
                                )
                            ))

                            for j, hour in enumerate(TIMES):
                                candidate_scores.append(dict(
                                    cell_line=cell,
                                    replicate=held,
                                    source_drug=source,
                                    target_drug=target,
                                    true_hour=true_hour,
                                    reference_mode=mode,
                                    candidate_hour=hour,
                                    vehicle_score=(
                                        result["null_scores"][j]
                                    ),
                                    active_score=(
                                        result["active_scores"][j]
                                    ),
                                    amplitude=(
                                        result["amplitudes"][j]
                                    )
                                ))

        say(
            "[CELL DONE | {}] held-out A/B/C; "
            "DMSO self-reference prohibited"
            .format(cell)
        )

    decisions = save(
        "02_noncircular_per_source_decisions.csv",
        decisions
    )
    save(
        "03_noncircular_candidate_scores.csv",
        candidate_scores
    )
    save(
        "04_gene_selection_replay.csv",
        selections
    )

    need(
        len(decisions) == 720
        and len(candidate_scores) == 3600
        and len(selections) == 3600,
        "V119 result count mismatch"
    )

    return decisions, historical

def summarize(p, historical):
    metrics = []
    unique_negatives = []
    references = []

    for mode in MODES:
        part = p[p.reference_mode == mode]

        same = part[
            (part.target_drug != "DMSO")
            & part.same_drug
        ]
        cross = part[
            (part.target_drug != "DMSO")
            & (~part.same_drug)
        ]
        dmso = part[
            part.target_drug == "DMSO"
        ]

        need(
            len(same) == 90
            and len(cross) == 180
            and len(dmso) == 90,
            "Source-drug counts mismatch"
        )

        agg = dmso.groupby(
            ["cell_line", "replicate", "true_hour"],
            as_index=False
        ).agg(
            any_source_active=(
                "accepted_active", "any"
            ),
            n_sources_active=(
                "accepted_active", "sum"
            )
        )
        need(len(agg) == 30, "Unique DMSO count mismatch")

        for row in agg.itertuples(index=False):
            unique_negatives.append(dict(
                reference_mode=mode,
                **row._asdict()
            ))

        for group, frame in (
            ("same_drug", same),
            ("cross_drug", cross),
            ("DMSO_source_calls", dmso)
        ):
            metrics.append(dict(
                reference_mode=mode,
                group=group,
                n_calls=len(frame),
                n_unique_queries=len(
                    frame[[
                        "cell_line", "replicate",
                        "target_drug", "true_hour"
                    ]].drop_duplicates()
                ),
                accepted_active=int(
                    frame.accepted_active.sum()
                ),
                time_correct_regardless_of_gate=int(
                    frame.correct_time.sum()
                ),
                time_correct_and_accepted=int(
                    frame.correct_and_accepted.sum()
                )
            ))

        say(
            "[V119 DMSO | {}] rejected per source={}/90 "
            "| ANY-source rejected={}/30".format(
                mode,
                int((~dmso.accepted_active).sum()),
                int((~agg.any_source_active).sum())
            )
        )

        say(
            "[V119 ACTIVE | {}] known-drug accepted={}/90 "
            "| time right before gate={}/90 "
            "| correct AND accepted={}/90".format(
                mode,
                int(same.accepted_active.sum()),
                int(same.correct_time.sum()),
                int(same.correct_and_accepted.sum())
            )
        )

        say(
            "[V119 TRANSFER | {}] cross-drug accepted={}/180 "
            "| correct AND accepted={}/180 "
            "(dependent predictions)".format(
                mode,
                int(cross.accepted_active.sum()),
                int(cross.correct_and_accepted.sum())
            )
        )

        for cell in FILES:
            a = same[same.cell_line == cell]
            d = agg[agg.cell_line == cell]

            say(
                "[V119 CELL | {} | {}] "
                "active accepted={}/45 | "
                "correct AND accepted={}/45 | "
                "DMSO unique rejected={}/15".format(
                    cell, mode,
                    int(a.accepted_active.sum()),
                    int(a.correct_and_accepted.sum()),
                    int((~d.any_source_active).sum())
                )
            )

    save("05_aggregate_gate_metrics.csv", metrics)
    save("06_unique_dmso_any_source.csv", unique_negatives)

    old = historical[[
        "cell_line", "holdout_replicate",
        "source_drug", "target_drug",
        "true_duration", "correct"
    ]].rename(columns={
        "holdout_replicate": "replicate",
        "true_duration": "true_hour",
        "correct": "v117_correct_using_heldout_DMSO"
    })

    for mode in MODES:
        same = p[
            (p.reference_mode == mode)
            & p.same_drug
        ]
        joined = same.merge(
            old,
            on=[
                "cell_line", "replicate",
                "source_drug", "target_drug",
                "true_hour"
            ],
            validate="one_to_one"
        )
        need(
            len(joined) == 90,
            "V117/V119 pairing mismatch"
        )

        prior_n = int(
            joined.v117_correct_using_heldout_DMSO.sum()
        )
        current_n = int(joined.correct_time.sum())

        references.append(dict(
            reference_mode=mode,
            v117_correct_with_heldout_DMSO=prior_n,
            v119_correct_training_only_DMSO=current_n,
            v119_correct_and_accepted=int(
                joined.correct_and_accepted.sum()
            ),
            n_unique_active=90
        ))

        say(
            "[REFERENCE ABLATION | {}] "
            "V117 heldout DMSO={}/90 | "
            "V119 training-only DMSO={}/90 "
            "(without activity gate)".format(
                mode, prior_n, current_n
            )
        )

    save(
        "07_reference_utility_vs_V117.csv",
        references
    )

def main():
    say("=" * 90)
    say(
        "PAPER 7 | V119 | NONCIRCULAR DMSO GATE "
        "AND TRAINING-ONLY VEHICLE REFERENCE"
    )
    say(
        "EXPLORATORY POST-V118 | NO RETUNING "
        "| NO HELDOUT DMSO SELF-REFERENCE | NO GIT"
    )
    say("=" * 90)

    need(len(sys.argv) == 1, "Run without flags")

    prior, old, features = load_prior()

    protocol = dict(
        version="v119",
        source="Diaz et al. GSE149428 processed logCPM",
        reference_modes=MODES,
        source_drugs=DRUGS,
        hours=TIMES,
        gene_count=TOP,
        ridge=RIDGE,
        variance_floor=FLOOR,
        training_replicates=2,
        heldout_replicates=1,
        gate_rule=(
            "minimum active ridge score "
            "less than minimum vehicle score"
        ),
        gate_threshold=0,
        unknown_drug_DMSO_rule=(
            "ANY of three source programs calls active"
        ),
        heldout_DMSO_used_as_reference=False,
        original_lock_SHA256=LOCK_SHA,
        prior_manifest_SHA256=sha(
            prior / "12_manifest.json"
        ),
        script_SHA256=sha(__file__),
        independent_validation=False,
        results_from_already_examined_study=True,
        no_hyperparameter_tuning=True,
        biological_onset_truth=False,
        preprocessing_leakage_not_ruled_out=True,
        no_network=True,
        no_git=True
    )

    (OUT / "01_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8"
    )

    decisions, historical = evaluate(old, features)
    summarize(decisions, historical)

    (OUT / "08_methods_and_limitations.md").write_text(
        """# V119: Noncircular vehicle gate

This is exploratory analysis on a previously
examined external study. It is not an additional
independent validation.

For BOTH active and DMSO query samples, the
vehicle reference comes only from the two
training replicate labels.

No DMSO query is compared to itself.

Two reference regimes are evaluated:
time-specific training-vehicle averages and
a time-pooled training-vehicle average.

The active model fits a ridge-regularized,
nonnegative response amplitude.

The gate accepts activity if the minimum active
score is lower than the minimum vehicle score.
There is no calibrated false-positive guarantee.

Known-drug active calls use the correct drug
identity. For DMSO with unknown drug identity,
the stricter rule requires rejection by ALL
three source-drug models.

A/B/C biological batch independence is not
established. Published preprocessing was
performed before train/test splitting.

Target is exposure duration, not experimentally
measured transcriptional onset.

No new independent validation or method
novelty claim is supported.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        __file__,
        OUT / "script_snapshot_v119.py"
    )

    for source, target in (
        ("12_manifest.json", "input_v117_manifest.json"),
        ("03_external_predictions.csv",
         "input_v117_predictions.csv"),
        ("09_training_only_gene_selections.csv",
         "input_v117_features.csv")
    ):
        shutil.copy2(prior / source, OUT / target)

    (OUT / "09_manifest.json").write_text(
        json.dumps(dict(
            version="v119",
            script_SHA256=sha(__file__),
            v117_manifest_SHA256=sha(
                prior / "12_manifest.json"
            ),
            decisions=len(decisions),
            source_files={
                cell: {
                    "filename": item[0],
                    "sha256": item[2]
                }
                for cell, item in FILES.items()
            },
            independent_validation=False,
            novelty_unproven=True,
            query_DMSO_not_used_as_reference=True,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__
        ), indent=2) + "\n",
        encoding="utf-8"
    )

    say(
        "[GATE] Descriptive activity detection only; "
        "no validated 95% specificity or biological onset"
    )
    say("[COMPLETE] Noncircular DMSO audit archived")

def finish(ok):
    say("[RESULT] " + ("SUCCESS" if ok else "FAILED"))

    archive = OUT / "portable_report_v119.zip"
    say("[REPORT] " + str(OUT / "console_summary.txt"))
    say("[ZIP] " + str(archive))

    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n",
        encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "10_sha256_inventory.csv", archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT / "10_sha256_inventory.csv",
        index=False, encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED,
        compresslevel=4
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for cell, (name, member, expected) in FILES.items():
            p = DATA / name
            need(
                p.is_file() and sha(p) == expected,
                "Publisher ZIP changed: " + cell
            )
            z.write(
                p, "publisher_source/" + name,
                compress_type=zipfile.ZIP_STORED
            )

    say(
        "[ZIP COMPLETE] Source archives, code, "
        "results, hashes and log"
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
