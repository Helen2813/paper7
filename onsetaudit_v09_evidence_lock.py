#!/usr/bin/env python3
"""OnsetAudit v0.9 — frozen evidence and external eligibility audit.

OFFLINE | NO FLAGS | NO REFITTING | NO THRESHOLD TUNING
All source data, prior models and prior reports remain unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import subprocess
import sys
import traceback
import zipfile

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "onsetaudit/_generated/reports"

TIMES = (1, 2, 6, 24)
GROUPS = (
    "experiment_1",
    "experiment_2",
    "experiment_3",
)
LABELS = ("DMSO", "SW016789", "Thapsigargin")

FALSE_CONTROLS = {
    "GSM5830673",
    "GSM5830674",
}

LOG = []


def say(*args):
    line = " ".join(map(str, args))
    print(line, flush=True)
    LOG.append(line)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    check(bool(rows), "Cannot save empty CSV")
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        w.writeheader()
        w.writerows(rows)


def as_bool(x):
    if x in ("True", "true", "Yes", "1"):
        return True
    if x in ("False", "false", "No", "0"):
        return False
    raise RuntimeError("Invalid boolean: " + str(x))


def latest_report():
    candidates = sorted(
        BASE.glob(
            "v08_shared_competition_*/audit_summary.json"
        )
    )
    check(candidates, "No v0.8 report found")
    return candidates[-1].parent


def verify_report(folder):
    manifest_path = folder / "sha256_manifest.json"

    check(
        manifest_path.is_file(),
        "Missing v0.8 SHA256 manifest",
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    check(
        isinstance(manifest, dict) and manifest,
        "Invalid manifest",
    )

    for name, expected in manifest.items():
        check(
            Path(name).name == name,
            "Invalid manifest filename",
        )
        p = folder / name
        check(p.is_file(), "Missing: " + name)
        check(
            sha(p) == expected,
            "SHA256 mismatch: " + name,
        )

    archive = folder / "onsetaudit_v08_audit.zip"
    check(archive.is_file(), "Missing v0.8 ZIP")

    with zipfile.ZipFile(archive) as z:
        check(
            z.testzip() is None,
            "Invalid ZIP integrity",
        )
        names = set(z.namelist())

        for name, expected in manifest.items():
            check(name in names, "ZIP missing: " + name)
            check(
                hashlib.sha256(
                    z.read(name)
                ).hexdigest() == expected,
                "ZIP mismatch: " + name,
            )

    summary = json.loads(
        (folder / "audit_summary.json").read_text(
            encoding="utf-8"
        )
    )

    check(
        summary.get("audit_version") == "v0.8.0",
        "Unexpected report version",
    )

    frozen = {
        "frozen_active_correct_and_accepted": 22,
        "frozen_dmso_rejected_distinct": 10,
        "cross_program_accepted": 24,
        "cross_program_duration_correct": 17,
    }

    for key, expected in frozen.items():
        check(
            summary.get(key) == expected,
            "Frozen result changed: " + key,
        )

    raw = read_csv(
        folder / "joint_36_heldout_predictions.csv"
    )

    check(len(raw) == 36, "Expected 36 predictions")

    seen_samples = set()
    design = Counter()
    rows = []

    for r in raw:
        sample = r["sample"]
        group = r["heldout_group"]
        actual = r["true_treatment"]
        predicted = r["predicted_treatment"]

        time = float(r["true_duration_h"])
        predicted_time = float(
            r["predicted_duration_h"]
        )

        check(
            sample not in seen_samples,
            "Duplicate sample: " + sample,
        )
        seen_samples.add(sample)

        check(
            group in GROUPS
            and actual in LABELS
            and predicted in LABELS,
            "Unexpected group or treatment",
        )

        check(
            time in TIMES and predicted_time in TIMES,
            "Unexpected timepoint",
        )

        class_ok = actual == predicted
        time_ok = time == predicted_time
        joint_ok = class_ok and time_ok

        check(
            as_bool(r["treatment_correct"]) == class_ok,
            "Classification flag mismatch",
        )
        check(
            as_bool(r["time_correct"]) == time_ok,
            "Time flag mismatch",
        )
        check(
            as_bool(
                r["joint_treatment_time_correct"]
            ) == joint_ok,
            "Joint flag mismatch",
        )

        gap = float(
            r["loss_gap_best_vs_second_class"]
        )
        check(gap >= -1e-8, "Invalid loss gap")

        design[(group, actual, time)] += 1

        rows.append({
            "sample": sample,
            "heldout_group": group,
            "true_treatment": actual,
            "predicted_treatment": predicted,
            "true_time_h": int(time),
            "predicted_time_h": int(predicted_time),
            "class_correct": class_ok,
            "time_correct": time_ok,
            "joint_correct": joint_ok,
            "loss_gap": gap,
        })

    for group in GROUPS:
        for label in LABELS:
            for t in TIMES:
                check(
                    design[(group, label, t)] == 1,
                    "Incomplete group/treatment/time design",
                )

    active = [
        r for r in rows
        if r["true_treatment"] != "DMSO"
    ]
    controls = [
        r for r in rows
        if r["true_treatment"] == "DMSO"
    ]

    metrics = {
        "n_all": 36,
        "n_active": 24,
        "n_dmso": 12,
        "active_detected": sum(
            r["predicted_treatment"] != "DMSO"
            for r in active
        ),
        "dmso_rejected": sum(
            r["predicted_treatment"] == "DMSO"
            for r in controls
        ),
        "active_drug_correct": sum(
            r["class_correct"] for r in active
        ),
        "active_time_correct": sum(
            r["time_correct"] for r in active
        ),
        "active_drug_and_time_correct": sum(
            r["joint_correct"] for r in active
        ),
        "three_class_correct": sum(
            r["class_correct"] for r in rows
        ),
        "all_class_and_time_correct": sum(
            r["joint_correct"] for r in rows
        ),
    }

    expected = {
        "active_detected": 24,
        "dmso_rejected": 10,
        "active_drug_correct": 21,
        "active_time_correct": 22,
        "active_drug_and_time_correct": 19,
        "three_class_correct": 31,
    }

    for key, value in expected.items():
        check(
            metrics[key] == value,
            "Metric changed: " + key,
        )

    result_mapping = {
        "active_detected": "active_detected",
        "dmso_correctly_rejected": "dmso_rejected",
        "active_drug_identity_correct":
            "active_drug_correct",
        "active_time_correct_regardless_drug":
            "active_time_correct",
        "active_drug_and_time_correct":
            "active_drug_and_time_correct",
        "all_3_class_correct":
            "three_class_correct",
        "all_joint_class_time_correct":
            "all_class_and_time_correct",
    }

    for source_key, metric_key in result_mapping.items():
        check(
            summary["results"][source_key]
            == metrics[metric_key],
            "JSON/CSV disagreement: " + source_key,
        )

    false_ids = {
        r["sample"] for r in controls
        if r["predicted_treatment"] != "DMSO"
    }

    check(
        false_ids == FALSE_CONTROLS,
        "False-control identities changed",
    )

    folds = read_csv(
        folder / "training_only_feature_union.csv"
    )

    check(len(folds) == 3, "Missing fold information")

    for f in folds:
        n = int(f["n_shared_union_genes"])
        overlap = int(f["selected_gene_overlap"])

        check(
            n + overlap == 400,
            "Invalid feature union",
        )
        check(
            len(f["shared_union_sha256"]) == 64,
            "Invalid gene hash",
        )

    say("[VERIFY] v0.8 ZIP / SHA256 / CSV / JSON: OK")
    say("[FROZEN] v0.4 22/24 active, 10/12 DMSO")
    say("[V07] Cross-program 24/24 accepted, 17/24 time")
    say(
        "[V08]",
        "24/24 detected;",
        "10/12 DMSO rejected;",
        "21/24 identity;",
        "19/24 identity+time",
    )
    say("[FALSE DMSO]", ", ".join(sorted(false_ids)))

    return summary, rows, metrics


def external_candidates():
    # Literature-based preliminary design triage only.
    # No external expression or sample metadata is loaded.
    return [
        {
            "accession": "GSE238017",
            "study": "Endocrine-cell ER stress",
            "design": "DMSO 24h; Thapsigargin 6h/24h",
            "critical_issue":
                "MIN6 samples reused from GSE194200; "
                "no full matching vehicle time series",
            "status": "NOT_INDEPENDENT_OR_FULLY_EVALUABLE",
            "source":
                "https://www.ncbi.nlm.nih.gov/geo/query/"
                "acc.cgi?acc=GSE238017",
        },
        {
            "accession": "GSE243039",
            "study": "Human endometrial EPC decidualization",
            "design":
                "Normal/endometriosis donors; "
                "EPC at days 0/2/4/6/8",
            "critical_issue":
                "Different treatment, organism and time scale; "
                "no matched DMSO hourly series",
            "status": "NOT_EVALUABLE_FOR_FROZEN_MODEL",
            "source":
                "https://www.ncbi.nlm.nih.gov/geo/query/"
                "acc.cgi?acc=GSE243039",
        },
        {
            "accession": "GSE254139",
            "study": "Human LUHMES neuronal stress",
            "design":
                "30min treatment followed by washout; "
                "samples at 3/6/24h after removal",
            "critical_issue":
                "Post-washout time is not ongoing "
                "exposure duration",
            "status": "DIFFERENT_TARGET_NOT_DIRECT_TRANSFER",
            "source":
                "https://www.ncbi.nlm.nih.gov/geo/query/"
                "acc.cgi?acc=GSE254139",
        },
        {
            "accession": "GSE129757",
            "study": "Human LN308 ER stress",
            "design":
                "Untreated baseline; drug at 2h/6h",
            "critical_issue":
                "Different organism and cell context; "
                "missing matching DMSO time series",
            "status": "NOT_EVALUABLE_FOR_FROZEN_MODEL",
            "source":
                "https://www.ncbi.nlm.nih.gov/geo/query/"
                "acc.cgi?acc=GSE129757",
        },
    ]


def html_table(rows):
    keys = list(rows[0])

    head = "".join(
        "<th>" + html.escape(k) + "</th>"
        for k in keys
    )

    body = ""

    for r in rows:
        body += "<tr>"
        for k in keys:
            body += (
                "<td>"
                + html.escape(str(r[k]))
                + "</td>"
            )
        body += "</tr>"

    return (
        "<div class='scroll'><table><tr>"
        + head + "</tr>"
        + body + "</table></div>"
    )


def main():
    BASE.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    out = BASE / ("v09_evidence_lock_" + stamp)
    out.mkdir(parents=True, exist_ok=False)

    try:
        say("=" * 70)
        say("PAPER 7 | ONSETAUDIT V0.9 | EVIDENCE LOCK")
        say(
            "OFFLINE | NO FLAGS | FROZEN RESULTS | "
            "NO NEW MODEL"
        )
        say("=" * 70)

        previous = latest_report()

        summary, rows, metrics = verify_report(previous)

        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        check(branch == "main", "Expected main")

        say("[GIT]", branch, head)

        groups = []

        for group in GROUPS:
            rr = [
                r for r in rows
                if r["heldout_group"] == group
            ]
            aa = [
                r for r in rr
                if r["true_treatment"] != "DMSO"
            ]
            cc = [
                r for r in rr
                if r["true_treatment"] == "DMSO"
            ]

            g = {
                "group": group,
                "total_samples": len(rr),
                "three_class_correct": sum(
                    r["class_correct"] for r in rr
                ),
                "active_joint_correct": sum(
                    r["joint_correct"] for r in aa
                ),
                "dmso_rejected": sum(
                    r["predicted_treatment"] == "DMSO"
                    for r in cc
                ),
            }

            groups.append(g)

            say(
                "[GROUP]", group,
                "class", str(g["three_class_correct"]) + "/12",
                "joint", str(g["active_joint_correct"]) + "/8",
            )

        candidates = external_candidates()

        for candidate in candidates:
            say(
                "[EXTERNAL GATE]",
                candidate["accession"],
                candidate["status"],
            )

        write_csv(
            out / "verified_36_predictions.csv",
            rows,
        )
        write_csv(
            out / "group_level_results.csv",
            groups,
        )
        write_csv(
            out / "external_candidate_gates.csv",
            candidates,
        )

        protocol = """# OnsetAudit — external validation protocol

## Frozen benchmark

GSE194200 remains a retrospective exploratory benchmark.

v0.4: 22/24 active correct duration and accepted;
10/12 DMSO rejected.

v0.8: 24/24 active detected; 10/12 DMSO rejected;
21/24 correct drug identities;
19/24 correct drug identity and duration.

The observed outcome is elapsed exposure duration,
NOT independently measured molecular onset.

## Eligibility for independent transfer

1. Independent samples with verified non-overlap.
2. Explicit and comparable intervention.
3. Known elapsed exposure time and units.
4. No confusion between exposure and post-washout time.
5. Negative controls matched to assessed timepoints.
6. Frozen features, objectives and parameters.
7. No use of target labels to tune the model.
8. Assayability and gene mapping verified before evaluation.
9. Proper donor/batch grouping.
10. Failures reported as NOT EVALUABLE when appropriate.

## Exclusions and limitations

GSE238017 partially reuses GSE194200 MIN6 samples.

GSE243039 measures EPC decidualization over days,
not the same perturbation and hour-scale model.

The other candidates require separate protocol checks.

This registry is based on published study designs.
No external matrices or sample-level metadata were
downloaded or verified in this run.

No new performance or significance claim is authorized.
"""
        (out / "external_validation_protocol.md").write_text(
            protocol,
            encoding="utf-8",
        )

        locked = {
            "version": "v0.9.0",
            "status":
                "FROZEN_EVIDENCE_LOCK_NO_NEW_MODEL",
            "source_dataset": "GSE194200",
            "git_head": head,
            "source_v08": str(previous),
            "source_v08_summary_sha256": sha(
                previous / "audit_summary.json"
            ),
            "source_v08_zip_sha256": sha(
                previous / "onsetaudit_v08_audit.zip"
            ),
            "metadata_sha256":
                summary.get("metadata_sha256"),
            "counts_sha256":
                summary.get("counts_sha256"),
            "frozen_v04": {
                "active_correct_accepted": 22,
                "vehicle_rejected": 10,
            },
            "exploratory_v08": metrics,
            "external_candidates": candidates,
            "external_data_downloaded": False,
            "external_performance_estimated": False,
            "model_refitted": False,
            "threshold_changed": False,
            "interpretation":
                "Retrospective elapsed-duration audit, "
                "not independently validated onset detection",
        }

        write_json(
            out / "evidence_lock.json",
            locked,
        )

        dashboard = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OnsetAudit v0.9 Evidence Lock</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  max-width: 1150px;
  margin: 36px auto;
  padding: 0 20px;
  background: #f5f7fa;
  color: #192e46;
}}
h1 {{ font-size: 27px; }}
h2 {{ margin-top: 30px; }}
.box {{
  background: white;
  border: 1px solid #dce4ee;
  padding: 19px;
  border-radius: 12px;
  margin: 16px 0;
}}
.warning {{
  background: #fff8e8;
  border-left: 4px solid #b88a34;
  padding: 16px;
  line-height: 1.6;
}}
.scroll {{ overflow-x: auto; }}
table {{
  border-collapse: collapse;
  font-size: 12px;
  width: 100%;
}}
td, th {{
  border-bottom: 1px solid #e3e9f1;
  padding: 9px;
  text-align: left;
  vertical-align: top;
}}
th {{ background: #eaf0f7; }}
</style>
</head>
<body>
<h1>OnsetAudit v0.9 — Evidence Lock</h1>
<p>GSE194200 · retrospective analysis · no refitting</p>

<div class="box">
<strong>24/24</strong> active detected<br>
<strong>10/12</strong> DMSO rejected<br>
<strong>21/24</strong> drug identity correct<br>
<strong>19/24</strong> drug + time correct
</div>

<div class="warning">
These are retrospective v0.8 results.
They are not independent validation or measurements
of biological onset.
</div>

<h2>Group-level results</h2>
<div class="box">
{html_table(groups)}
</div>

<h2>External dataset eligibility</h2>
<div class="box">
{html_table(candidates)}
</div>

<h2>Provenance</h2>
<div class="box">
Git commit: {html.escape(head)}<br>
Source v0.8 SHA256:
{html.escape(locked["source_v08_summary_sha256"])}
</div>
</body>
</html>
"""

        html_path = out / "onsetaudit_v09_dashboard.html"
        html_path.write_text(
            dashboard,
            encoding="utf-8",
        )

        shutil.copyfile(
            Path(__file__),
            out / Path(__file__).name,
        )

        say("[READY] All frozen results verified")
        say("[MODEL] Unchanged")
        say("[EXTERNAL PERFORMANCE] Not evaluated")
        say("[OUTPUT]", out)

        (out / "console.log").write_text(
            "\n".join(LOG) + "\n",
            encoding="utf-8",
        )

        files = sorted(
            p for p in out.iterdir()
            if p.is_file()
        )

        write_json(
            out / "sha256_manifest.json",
            {p.name: sha(p) for p in files},
        )

        archive = out / "onsetaudit_v09_evidence_lock.zip"

        with zipfile.ZipFile(
            archive, "w", zipfile.ZIP_DEFLATED
        ) as z:
            for p in sorted(out.iterdir()):
                if p.is_file() and p != archive:
                    z.write(p, arcname=p.name)

        with zipfile.ZipFile(archive) as z:
            check(
                z.testzip() is None,
                "New ZIP integrity failure",
            )

        say("[ZIP]", archive)

        if sys.platform == "darwin":
            subprocess.run(
                ["open", str(html_path)],
                check=False,
            )

    except Exception:
        error = traceback.format_exc()
        print("[STOP]", error, file=sys.stderr)
        (out / "failure.log").write_text(
            "\n".join(LOG) + "\n" + error,
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
