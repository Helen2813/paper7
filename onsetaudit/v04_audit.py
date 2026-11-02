"""OnsetAudit v0.4 descriptive negative-control diagnostics.

Post-hoc sensitivity summaries are NEVER used to select a gate threshold,
retrain the duration model, or make calibration/coverage claims.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any


FIXED_THRESHOLDS = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def build_v04_audit(
    predictions: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    vehicle_drug_predictions: list[dict[str, Any]],
    drugs: list[str],
    groups: list[str],
    times: list[float],
) -> dict[str, Any]:
    """Check baseline invariants, then summarize drug-specific held-out controls.

    Each vehicle is scored against every trained active-drug program while
    excluding its entire experimental group from template/feature learning.
    Unit counts are sample counts, not independent biological replicates.
    """
    if not predictions or not vehicles or not vehicle_drug_predictions:
        raise ValueError("NOT EVALUABLE: v0.4 requires active and DMSO margin rows")
    expected_drugs = set(drugs)
    by_sample = defaultdict(dict)
    for row in vehicle_drug_predictions:
        sid = str(row["sample"])
        drug = str(row["drug"])
        if drug not in expected_drugs or drug in by_sample[sid]:
            raise ValueError("Inconsistent/duplicate vehicle sample × drug row")
        val = float(row["gate_margin"])
        if not isfinite(val):
            raise ValueError("Non-finite held-out vehicle margin")
        if bool(row["accepted_as_active"]) != (val > 0.0):
            raise ValueError("Vehicle gate/score inconsistency")
        by_sample[sid][drug] = row

    if len(by_sample) != len(vehicles):
        raise ValueError("Unique vehicle sample count does not match margin table")

    for v in vehicles:
        sid = str(v["sample"])
        if set(by_sample[sid]) != expected_drugs:
            raise ValueError("Incomplete vehicle-drug scoring for " + sid)
        if len(by_sample[sid]) != int(v["source_programs_checked"]):
            raise ValueError("Source program count mismatch for " + sid)
        is_fp = any(x["accepted_as_active"] for x in by_sample[sid].values())
        if is_fp != bool(v["false_active_any_drug"]):
            raise ValueError("Frozen negative-control baseline mismatch for " + sid)
        if "max_gate_margin" in v:
            largest = max(float(x["gate_margin"]) for x in by_sample[sid].values())
            if abs(largest - float(v["max_gate_margin"])) > 1e-10:
                raise ValueError("Aggregated vehicle margin mismatch")

    for p in predictions:
        margin = float(p["gate_margin"])
        if not isfinite(margin):
            raise ValueError("Non-finite active sample margin")
        if bool(p["accepted_active"]) != (margin > 0.0):
            raise ValueError("Frozen active gate/score mismatch for " + str(p["sample"]))
        if bool(p["correct_and_accepted"]) != (bool(p["correct_time"]) and margin > 0.0):
            raise ValueError("Frozen correct-and-accepted mismatch")

    n_a, n_v = len(predictions), len(vehicles)
    rows_drug = []
    for drug in drugs:
        positives = [p for p in predictions if p["drug"] == drug]
        controls = [v for v in vehicle_drug_predictions if v["drug"] == drug]
        fp = [v for v in controls if v["accepted_as_active"]]
        rows_drug.append({
            "drug": drug,
            "active_n": len(positives),
            "active_accepted": sum(bool(p["accepted_active"]) for p in positives),
            "duration_correct": sum(bool(p["correct_time"]) for p in positives),
            "control_n": len(controls),
            "false_active_calls": len(fp),
            "false_active_fraction": len(fp) / len(controls) if controls else None,
            "false_positive_samples": ";".join(str(v["sample"]) for v in fp),
        })

    rows_group = []
    for group in groups:
        pos = [p for p in predictions if p["group"] == group]
        neg = [v for v in vehicles if v["group"] == group]
        fp = sum(bool(v["false_active_any_drug"]) for v in neg)
        rows_group.append({
            "holdout_group": group,
            "active_n": len(pos),
            "duration_correct": sum(bool(p["correct_time"]) for p in pos),
            "active_accepted": sum(bool(p["accepted_active"]) for p in pos),
            "vehicle_n": len(neg),
            "vehicle_false_positive_any_drug": fp,
            "vehicle_rejected_any_drug": len(neg) - fp,
        })

    rows_time = []
    for t in times:
        pos = [p for p in predictions if float(p["true_hours"]) == float(t)]
        neg = [v for v in vehicles if float(v["time"]) == float(t)]
        rows_time.append({
            "time_h": float(t),
            "active_n": len(pos),
            "duration_correct": sum(bool(p["correct_time"]) for p in pos),
            "active_accepted": sum(bool(p["accepted_active"]) for p in pos),
            "vehicle_n": len(neg),
            "vehicle_false_positive_any_drug": sum(bool(v["false_active_any_drug"]) for v in neg),
        })

    max_vehicle_margins = {
        sid: max(float(item["gate_margin"]) for item in items.values())
        for sid, items in by_sample.items()
    }
    rows_sweep = []
    for threshold in FIXED_THRESHOLDS:
        accepted = sum(float(p["gate_margin"]) > threshold for p in predictions)
        jointly_correct = sum(bool(p["correct_time"]) and float(p["gate_margin"]) > threshold for p in predictions)
        fp = sum(m > threshold for m in max_vehicle_margins.values())
        rows_sweep.append({
            "threshold": threshold,
            "is_frozen_v03_gate": threshold == 0.0,
            "active_accepted": accepted,
            "active_n": n_a,
            "correct_and_accepted": jointly_correct,
            "vehicle_false_positive_any_drug": fp,
            "vehicle_rejected_any_drug": n_v - fp,
            "vehicle_n": n_v,
            "interpretation": "FROZEN_BASELINE" if threshold == 0.0 else "RETROSPECTIVE_SENSITIVITY_ONLY",
        })
    original = next(x for x in rows_sweep if x["is_frozen_v03_gate"])
    if original["active_accepted"] != sum(bool(p["accepted_active"]) for p in predictions):
        raise ValueError("Threshold zero changed frozen active results")
    if original["correct_and_accepted"] != sum(bool(p["correct_and_accepted"]) for p in predictions):
        raise ValueError("Threshold zero changed frozen duration results")
    if original["vehicle_rejected_any_drug"] != sum(not v["false_active_any_drug"] for v in vehicles):
        raise ValueError("Threshold zero changed frozen DMSO results")

    fp_drug_calls = [x for x in vehicle_drug_predictions if x["accepted_as_active"]]
    return {
        "status": "DESCRIPTIVE_AUDIT_NOT_CALIBRATED",
        "frozen_gate_margin_threshold": 0.0,
        "frozen_baseline_reproduced": True,
        "n_active_samples": n_a,
        "n_vehicle_samples": n_v,
        "n_vehicle_drug_comparisons": len(vehicle_drug_predictions),
        "vehicle_false_positive_samples_any_drug": sum(bool(v["false_active_any_drug"]) for v in vehicles),
        "vehicle_false_positive_drug_calls": len(fp_drug_calls),
        "by_drug": rows_drug,
        "by_group": rows_group,
        "by_time": rows_time,
        "threshold_sweep": rows_sweep,
        "limitations": [
            "Threshold grid is a post-hoc diagnostic on the same holdouts; do NOT select a new operating threshold from it",
            "Threshold 0.0 reproduces the frozen v0.3 gate; the duration model and all feature choices are unchanged",
            "Vehicle specificity is an any-drug multiple-comparison endpoint; per-drug calls are not independent",
            "Three experimental groups are too few to justify stable population-level calibration or 95% specificity",
            "Prediction target is recorded exposure duration, NOT independently measured molecular onset",
            "Group/replicate labels alone do not demonstrate biological independence",
        ],
    }
