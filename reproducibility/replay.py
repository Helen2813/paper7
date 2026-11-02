"""Replay the historical Paper 7 development analyses."""

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reproducibility" / "_generated"

STAGES = (
    ("V112", "paper7_v112_gse194200_import.py"),
    ("V113", "paper7_v113_gse194200_real_baselines.py"),
    ("V114", "paper7_v114_external_stress_audit.py"),
    ("V115", "paper7_v115_control_anchored_pilot.py"),
    ("V116_freeze_only", None),
    ("V116c", "paper7_v116c_elife_source_audit.py"),
    ("V117", "paper7_v117_elife_locked_validation.py"),
    ("V118", "paper7_v118_external_evidence_audit.py"),
    ("V119", "paper7_v119_noncircular_vehicle_gate.py"),
    ("V120", "paper7_v120_twostage_audit.py"),
)

FREEZE_HASH = (
    "a3043faf9edffd4d1b38d8d0b6b0d04ad24bc6bd3f8620840dec52b4a5e3dfff"
)

def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()

def freeze():
    script = ROOT / "paper7_v116_lock_external_discovery.py"
    spec = importlib.util.spec_from_file_location(
        "paper7_v116_original", script
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.freeze_spec()
    mod.verify_v115()

    lock = mod.OUT / "01_frozen_external_protocol.json"

    if sha(lock) != FREEZE_HASH:
        raise RuntimeError(
            "Original V116 freeze checksum differs: " + sha(lock)
        )

    print("[ORIGINAL V116 LOCK VERIFIED]", sha(lock))

def archived(folder, filename):
    base = ROOT / folder
    runs = sorted(
        (
            p for p in base.glob("run_*")
            if (p / filename).is_file()
            and (p / "console_summary.txt").is_file()
        ),
        reverse=True
    )

    if not runs:
        raise RuntimeError("Missing result: " + folder)

    run = runs[0]
    log = (run / "console_summary.txt").read_text("utf-8")

    if "[RESULT] SUCCESS" not in log:
        raise RuntimeError("Archived run failed: " + str(run))

    return run / filename

def verify():
    import pandas as pd

    checks = (
        ("V113", "paper7_gse194200_real_baseline_v113",
         "07_heldout_metrics.csv",
         {"method": "drug_aware_centroid",
          "treatment": "ALL_ACTIVE"},
         "correct", 12),

        ("V115 same", "paper7_gse194200_paired_vehicle_v115",
         "06_paired_method_metrics.csv",
         {"method": "paired_effect_ridge_amplitude",
          "transfer": "same_drug"},
         "correct", 21),

        ("V115 cross", "paper7_gse194200_paired_vehicle_v115",
         "06_paired_method_metrics.csv",
         {"method": "paired_effect_ridge_amplitude",
          "transfer": "cross_drug"},
         "correct", 18),

        ("V117 fixed", "paper7_v117_external_locked_pilot",
         "05_external_method_metrics.csv",
         {"method": "paired_effect_fixed",
          "transfer": "same_drug"},
         "correct", 78),

        ("V117 cross", "paper7_v117_external_locked_pilot",
         "05_external_method_metrics.csv",
         {"method": "paired_effect_ridge_amplitude",
          "transfer": "cross_drug"},
         "correct", 96),

        ("V119 timing", "paper7_v119_noncircular_vehicle_gate",
         "05_aggregate_gate_metrics.csv",
         {"reference_mode": "training_time_matched",
          "group": "same_drug"},
         "time_correct_regardless_of_gate", 84),

        ("V120 joint", "paper7_v120_twostage_audit",
         "04_hybrid_summary.csv",
         {}, "hybrid_correct_and_accepted", 79),

        ("V120 DMSO", "paper7_v120_twostage_audit",
         "04_hybrid_summary.csv",
         {}, "DMSO_rejected", 29),
    )

    proof = []

    for label, folder, filename, filters, field, expected in checks:
        path = archived(folder, filename)
        df = pd.read_csv(path, encoding="utf-8-sig")

        for key, value in filters.items():
            if key not in df:
                raise RuntimeError("Missing field " + key)
            df = df[df[key].astype(str) == str(value)]

        if df.empty or field not in df:
            raise RuntimeError("Missing metric: " + label)

        actual = int(pd.to_numeric(df[field]).sum())
        passed = actual == expected

        print(
            "[{}] {}: {} / {}".format(
                "PASS" if passed else "FAIL",
                label, actual, expected
            )
        )

        proof.append({
            "label": label,
            "actual": actual,
            "expected": expected,
            "source": str(path.relative_to(ROOT)),
            "sha256": sha(path)
        })

        if not passed:
            raise RuntimeError("Scientific replay mismatch: " + label)

    return proof

def main():
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Python 3.11 or 3.12 recommended")

    import numpy
    import pandas

    tag = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    folder = RUNS / ("replay_" + tag)
    folder.mkdir(parents=True, exist_ok=False)

    report = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "platform": platform.platform(),
        "steps": [],
        "status": "INCOMPLETE"
    }

    try:
        for name, relative in STAGES:
            script = (
                ROOT / relative if relative
                else ROOT / "paper7_v116_lock_external_discovery.py"
            )

            print("\n[STAGE]", name, flush=True)
            logfile = folder / (name + ".log")

            with logfile.open("w", encoding="utf-8") as log:
                if relative:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", str(script)],
                        cwd=ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    for line in proc.stdout:
                        log.write(line)
                        print(line, end="", flush=True)

                    code = proc.wait()
                else:
                    code = 0
                    try:
                        freeze()
                    except Exception as exc:
                        code = 1
                        log.write(str(exc) + "\n")

                report["steps"].append({
                    "name": name,
                    "exit": code,
                    "source_sha256": sha(script)
                })

            if code != 0:
                raise RuntimeError(
                    "Stopped at {}. See {}".format(name, logfile)
                )

        report["numerical_checks"] = verify()
        report["status"] = "REPLAY_PASSED_DEVELOPMENT_COUNTS"

    finally:
        manifest = folder / "manifest.json"
        manifest.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8"
        )
        print("[MANIFEST]", manifest)

    print("[COMPLETE] Development results reproduced")

if __name__ == "__main__":
    main()
