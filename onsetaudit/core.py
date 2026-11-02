"""OnsetAudit research core — experimental duration, NOT biological onset.

No network access. Treat all user input as untrusted. No learned model is shipped.
The exploratory training-only paired-vehicle algorithm follows V115/V120
formulae and is labelled an *adaptation* on datasets with different cohorts.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from v04_audit import build_v04_audit

VERSION = "0.4.0"
MAX_DECOMPRESSED = 320 * 1024 * 1024
MAX_GENES = 70000
MAX_SAMPLES = 2000
TOP_GENES = 200
VAR_FLOOR = 0.25
RIDGE = 0.25
CONTROLS = {"dmso", "vehicle", "control", "untreated", "mock", "pbs"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack(data: bytes, filename: str) -> bytes:
    """Decode gzip or one-tabular-member ZIP; bound inflation to 320 MiB."""
    if len(data) > MAX_DECOMPRESSED:
        raise ValueError("File exceeds the 320 MiB input limit")
    if data[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            raw = handle.read(MAX_DECOMPRESSED + 1)
    elif data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            candidates = [
                m for m in archive.infolist()
                if not m.is_dir() and not m.filename.startswith("__MACOSX")
                and Path(m.filename).suffix.lower() in {".csv", ".tsv", ".txt"}
            ]
            if len(candidates) != 1:
                raise ValueError("ZIP must contain exactly one CSV/TSV/TXT data table")
            if candidates[0].file_size > MAX_DECOMPRESSED:
                raise ValueError("Expanded ZIP member is too large")
            with archive.open(candidates[0]) as f:
                raw = f.read(MAX_DECOMPRESSED + 1)
    else:
        raw = data
    if len(raw) > MAX_DECOMPRESSED:
        raise ValueError("Decompressed data exceed 320 MiB")
    return raw


def table_from_bytes(data: bytes, filename: str, *, matrix: bool = False) -> pd.DataFrame:
    raw = unpack(data, filename)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Data must be UTF-8 encoded CSV or TSV") from exc
    preview = io.StringIO(text)
    first = next((x for x in preview if x.strip()), "")
    if not first:
        raise ValueError("Empty table")
    sep = "\t" if "\t" in first else "," if first.count(",") >= first.count(";") else ";"
    try:
        header = next(csv.reader([first], delimiter=sep))
        nextline = next((x for x in preview if x.strip()), None)
        if nextline is None:
            raise ValueError("Table must have at least one data row")
        nfirst = len(next(csv.reader([nextline], delimiter=sep)))
        if matrix and nfirst == len(header) + 1:
            # GEO DRUG-seq: header has 160 sample IDs, rows contain gene ID + 160 counts.
            if not all(header) or header[0].strip().lower() in {"gene", "gene_id"}:
                raise ValueError("Ambiguous unlabeled gene-column layout")
            names = ["gene_id"] + header
            frame = pd.read_csv(io.StringIO(text), sep=sep, header=None,
                                names=names, skiprows=1, dtype=str,
                                keep_default_na=False, low_memory=False)
        elif nfirst == len(header):
            frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str,
                                keep_default_na=False, low_memory=False)
            if matrix and frame.columns[0].strip() in {"", "Unnamed: 0"}:
                frame.rename(columns={frame.columns[0]: "gene_id"}, inplace=True)
        else:
            raise ValueError(f"Inconsistent table widths: header {len(header)}, data row {nfirst}")
    except (pd.errors.ParserError, csv.Error) as exc:
        raise ValueError(f"CSV/TSV parsing error: {exc}") from exc
    if frame.empty:
        raise ValueError("No data rows")
    frame.columns = [str(x).strip() for x in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column labels in input table")
    return frame


def pick(columns: list[str], options: tuple[str, ...], required=True) -> str | None:
    by_lower = {re.sub(r"[^a-z0-9]", "", c.casefold()): c for c in columns}
    for option in options:
        key = re.sub(r"[^a-z0-9]", "", option.casefold())
        if key in by_lower:
            return by_lower[key]
    if required:
        raise ValueError("Missing metadata column; expected one of: " + ", ".join(options))
    return None


def time_hours(x: str) -> float:
    s = str(x).strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours?)?", s, flags=re.I)
    if not m:
        raise ValueError("Time values must be nonnegative hours, e.g. 6h, 22h, 3, 12.5")
    return float(m.group(1))


def clean_metadata(data: bytes, filename: str) -> pd.DataFrame:
    original = table_from_bytes(data, filename)
    cols = list(original.columns)
    sample = pick(cols, ("sample_column_name", "sample_id", "sample", "gsm", "sample_name", "id"))
    drug = pick(cols, ("drug", "treatment", "condition", "compound"))
    time = pick(cols, ("timepoint", "time_hours", "duration_hours", "duration", "hours", "time"))
    group = pick(cols, ("library", "batch", "experiment", "donor", "culture", "group"), required=False)
    replicate = pick(cols, ("replicate", "replicate_id", "sample_replicate", "rep", "replicate_label"), required=False)
    result = pd.DataFrame({
        "sample": original[sample].astype(str).str.strip(),
        "drug": original[drug].astype(str).str.strip(),
        "time": original[time].astype(str).map(time_hours),
        "group": original[group].astype(str).str.strip() if group else "UNSPECIFIED",
        "replicate": original[replicate].astype(str).str.strip() if replicate else "UNSPECIFIED",
    })
    if len(result) > MAX_SAMPLES:
        raise ValueError(f"Too many samples (>{MAX_SAMPLES})")
    if (result[["sample", "drug", "group", "replicate"]] == "").any().any():
        raise ValueError("Blank sample, treatment, group or replicate values")
    if result["sample"].duplicated().any():
        dup = result.loc[result["sample"].duplicated(), "sample"].iloc[0]
        raise ValueError(f"Duplicate sample ID: {dup}")
    if result.duplicated(["group", "drug", "time", "replicate"]).any() and replicate:
        raise ValueError("Duplicate group/drug/time/replicate assignments")
    result["control"] = result["drug"].str.lower().isin(CONTROLS)
    result.attrs["original_columns"] = cols
    result.attrs["group_explicit"] = group is not None
    result.attrs["replicate_explicit"] = replicate is not None
    return result


@dataclass
class Dataset:
    meta: pd.DataFrame | None = None
    gene_ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None  # genes × samples, normalized log2(1+CPM) or input logCPM
    counts: np.ndarray | None = None
    matrix_kind: str | None = None
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_model: dict[str, Any] | None = None
    demo: bool = False

    def import_metadata(self, payload: bytes, filename: str) -> dict:
        meta = clean_metadata(payload, filename)
        self.meta = meta
        self.gene_ids = []
        self.matrix = None
        self.counts = None
        self.matrix_kind = None
        self.last_model = None
        self.demo = False
        self.files["metadata"] = {"name": Path(filename).name, "sha256": sha256_bytes(payload), "bytes": len(payload)}
        return self.summary()

    def import_matrix(self, payload: bytes, filename: str, kind: str = "counts") -> dict:
        if self.meta is None:
            raise ValueError("Import sample metadata first")
        if kind not in {"counts", "logcpm"}:
            raise ValueError("Matrix type must be counts or logcpm")
        frame = table_from_bytes(payload, filename, matrix=True)
        if len(frame) > MAX_GENES:
            raise ValueError("Too many genes")
        if frame.shape[1] < 3:
            raise ValueError("Expression matrix must contain a gene-ID and at least two samples")
        first = frame.columns[0]
        names = self.meta["sample"].tolist()
        vals = frame.columns[1:].tolist()
        missing = sorted(set(names) - set(vals))
        extras = sorted(set(vals) - set(names))
        if missing or extras:
            raise ValueError(
                f"Matrix↔metadata sample mismatch. Missing: {missing[:5]}; extra: {extras[:5]}"
            )
        genes = frame[first].astype(str).str.strip().tolist()
        if any(not x for x in genes) or len(set(genes)) != len(genes):
            raise ValueError("Missing or duplicate gene identifiers")
        try:
            arr = frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("Expression cells must all be numeric") from exc
        if arr.shape != (len(genes), len(names)) or not np.isfinite(arr).all():
            raise ValueError("Non-finite or inconsistent expression values")
        if kind == "counts":
            if np.any(arr < 0) or not np.all(np.abs(arr - np.rint(arr)) < 1e-7):
                raise ValueError("Raw counts must be nonnegative integers; select logCPM for processed data")
            library = arr.sum(axis=0)
            if np.any(library <= 0):
                raise ValueError("Zero-depth count library")
            normalized = np.log2(1.0 + arr * (1e6 / library)[None, :])
            self.counts = arr
            self.matrix = normalized
        else:
            self.matrix = arr
            self.counts = None
        self.gene_ids = genes
        self.matrix_kind = kind
        self.files["expression"] = {"name": Path(filename).name, "sha256": sha256_bytes(payload), "bytes": len(payload)}
        self.last_model = None
        return self.summary()

    def summary(self) -> dict:
        if self.meta is None:
            return {"loaded": False, "matrix_loaded": False, "demo": False}
        meta = self.meta
        group_cells = meta.groupby(["drug", "time"], dropna=False).size()
        values = {str(drug): {str(float(t)): int(n) for (d, t), n in group_cells.items() if d == drug}
                  for drug in sorted(meta.drug.unique())}
        control_times = set(meta.loc[meta.control, "time"])
        active_times = set(meta.loc[~meta.control, "time"])
        groups = sorted(meta.group.unique().tolist())
        warnings: list[str] = []
        if not meta.attrs.get("group_explicit"):
            warnings.append("Batch / culture / experiment column missing: independent held-out testing unavailable")
        if not meta.attrs.get("replicate_explicit"):
            warnings.append("Replicate labels absent: matched biological pairing cannot be verified")
        if not meta.control.any():
            warnings.append("No DMSO / vehicle / untreated controls detected")
        if active_times - control_times:
            warnings.append("No time-matched control at: " + ", ".join(str(t) for t in sorted(active_times-control_times)))
        if len(groups) < 2:
            warnings.append("Fewer than two batch groups; grouped holdout unavailable")
        counts_stats = None
        if self.counts is not None:
            depths = self.counts.sum(axis=0)
            counts_stats = {"min": int(depths.min()), "median": int(np.median(depths)), "max": int(depths.max())}
            if depths.min() < 5000:
                warnings.append("Some RNA-seq libraries contain fewer than 5,000 assigned counts; interpretation may be unreliable")
            if depths.max()/depths.min() > 50:
                warnings.append("Library depths differ by more than 50-fold; inspect depth effects before evaluating time")
        return {
            "loaded": True, "demo": self.demo, "matrix_loaded": self.matrix is not None,
            "samples": int(len(meta)), "genes": len(self.gene_ids),
            "treatments": sorted(meta.drug.unique().tolist()),
            "groups": groups, "replicate_labels": sorted(meta.replicate.unique().tolist()),
            "times": sorted(float(x) for x in meta.time.unique()),
            "controls": int(meta.control.sum()), "counts_depth": counts_stats,
            "coverage": values, "warnings": warnings, "files": self.files,
            "matrix_kind": self.matrix_kind,
            "identifiability": "SHAPE_FREE_ONSET_NOT_POINT_IDENTIFIED_FROM_DESIGN_ALONE",
            "model_available": bool(self.matrix is not None and meta.control.any() and len(groups) >= 2),
            "model_run": self.last_model is not None,
        }

    def genes(self, query: str = "") -> list[str]:
        query = str(query).casefold().strip()
        if not query:
            return self.gene_ids[:40]
        return [g for g in self.gene_ids if query in g.casefold()][:80]

    def curve(self, gene: str, selected_drugs: list[str] | None = None) -> dict:
        if self.meta is None or self.matrix is None:
            raise ValueError("Load metadata and a matching expression matrix first")
        try:
            g = self.gene_ids.index(gene)
        except ValueError as exc:
            raise ValueError("Gene not found in expression matrix") from exc
        meta = self.meta
        if selected_drugs:
            drugs = selected_drugs
        else:
            all_drugs = sorted(meta.drug.unique())
            drugs = [d for d in all_drugs if d.casefold() in CONTROLS][:1] + [d for d in all_drugs if d.casefold() not in CONTROLS][:4]
        series = []
        for drug in drugs:
            points = []
            for hour in sorted(meta.loc[meta.drug == drug, "time"].unique()):
                ids = meta.index[(meta.drug == drug) & (meta.time == hour)].to_numpy()
                y = self.matrix[g, ids]
                points.append({"t": float(hour), "mean": float(np.mean(y)),
                               "sd": float(np.std(y, ddof=1)) if len(y)>1 else None,
                               "n": int(len(y))})
            if points:
                series.append({"drug": drug, "control": drug.casefold() in CONTROLS, "points": points})
        return {"gene": gene, "matrix_kind": self.matrix_kind, "series": series,
                "axis": "log2(1+CPM)" if self.matrix_kind=="counts" else "input logCPM",
                "note": "Mean ± SD across available samples; not an estimate of onset time"}

    def design_audit(self, tau_a: float, tau_b: float) -> dict:
        if self.meta is None:
            raise ValueError("Import metadata first")
        if not (0 <= tau_a < tau_b):
            raise ValueError("Require 0 ≤ earlier shift < later shift")
        times = sorted(float(x) for x in self.meta.time.unique())
        same = [t for t in times if t <= tau_a]
        between = [t for t in times if tau_a < t <= tau_b]
        later = [t for t in times if t > tau_b]
        return {
            "earlier_onset_h": tau_a, "later_onset_h": tau_b,
            "observed_times_h": times, "pre_both_onsets_h": same,
            "between_onsets_h": between, "after_both_onsets_h": later,
            "measured_in_discriminating_window": len(between) > 0,
            "status": "STRUCTURALLY_AMBIGUOUS_WITH_ARBITRARY_POST_ONSET_SHAPES",
            "note": ("A measured time between candidate onsets can constrain a flat-pre-onset model, "
                     "but does not prove a biological onset without a response-shape and noise model."),
        }

    def demonstration(self) -> dict:
        """A clearly labelled synthetic QC and model demo, never biological evidence."""
        rng = np.random.default_rng(7103)
        groups = ("batch_A", "batch_B", "batch_C")
        drugs = ("DMSO", "COMPOUND_A", "COMPOUND_B")
        times = (1, 2, 6, 24)
        records = []
        vals = []
        genes = [f"SIM_GENE_{i:03d}" for i in range(450)]
        t_axis = np.arange(450)
        baseline = rng.normal(5, 0.9, size=450)
        for grp in groups:
            for rep in ("r1", "r2"):
                for drug in drugs:
                    for t in times:
                        name = f"{grp}_{rep}_{drug}_{t}h"
                        records.append({"sample": name, "drug": drug, "time": float(t),
                                        "group": grp, "replicate": rep, "control": drug == "DMSO"})
                        shape = (np.sin(t_axis/19.0) + 1.0) / 2.0
                        response = 0.0 if drug == "DMSO" else (np.log2(t+1)-1) * shape * (1.2 if drug == "COMPOUND_A" else -0.9)
                        vals.append(baseline + response + rng.normal(0, .14, size=450))
        self.meta = pd.DataFrame(records)
        self.meta.attrs["group_explicit"] = True
        self.meta.attrs["replicate_explicit"] = True
        self.gene_ids = genes
        self.matrix = np.array(vals).T
        self.counts = None
        self.matrix_kind = "logcpm"
        self.files = {"metadata": {"name": "SYNTHETIC_DEMO", "sha256": "SYNTHETIC", "bytes": 0},
                      "expression": {"name": "SYNTHETIC_DEMO", "sha256": "SYNTHETIC", "bytes": 0}}
        self.last_model = None
        self.demo = True
        return self.summary()

    def evaluate(self, drugs: list[str] | None = None) -> dict:
        """Training-only controls; group holdout; known-drug duration model.

        This is a research development implementation, not clinically validated,
        not a claim of methodological novelty, and not transcription onset truth.
        """
        if self.meta is None or self.matrix is None:
            raise ValueError("Import metadata and expression data first")
        meta = self.meta
        X = self.matrix
        counts = self.counts
        controls = meta.loc[meta.control, "drug"].unique().tolist()
        if len(controls) != 1:
            raise ValueError("Exactly one consistently named vehicle/control treatment is required for this pilot")
        ctrl = controls[0]
        selected = (drugs or sorted(meta.loc[~meta.control, "drug"].unique().tolist()))
        if not selected or any(x not in set(meta.drug) or x == ctrl for x in selected):
            raise ValueError("Choose at least one valid active treatment")
        groups = sorted(meta.group.unique().tolist())
        times = sorted(meta.loc[~meta.control, "time"].unique().tolist())
        if len(groups) < 2 or len(times) < 2:
            raise ValueError("At least 2 heldout groups and 2 candidate durations are required")
        if set(times) - set(meta.loc[meta.control, "time"]):
            raise ValueError("Missing vehicle controls for candidate times")
        lookup: dict[tuple, int] = {}
        for i, row in meta.iterrows():
            key = (row['group'], row['replicate'], row['drug'], row['time'])
            if key in lookup:
                raise ValueError("Nonunique batch/replicate/drug/time unit")
            lookup[key] = i
        if len(self.gene_ids) < TOP_GENES:
            raise ValueError("At least 200 gene rows are required by frozen V115 architecture")
        predictions = []
        negative_source_calls: dict[int, list[dict[str, Any]]] = {}
        folds = []
        for held in groups:
            for drug in selected:
                train_groups = [g for g in groups if g != held]
                units = []
                for grp in train_groups:
                    reps = sorted(meta.loc[meta.group == grp, 'replicate'].unique())
                    for rep in reps:
                        if all((grp, rep, drug, t) in lookup and (grp, rep, ctrl, t) in lookup for t in times):
                            units.append((grp, rep))
                if len(units) < 2:
                    raise ValueError(f"NOT EVALUABLE: {drug} heldout {held} has only {len(units)} fully paired training replicates; need ≥2")
                contrasts = np.stack([
                    np.column_stack([X[:, lookup[(grp, rep, drug, t)]] - X[:, lookup[(grp, rep, ctrl, t)]] for t in times])
                    for grp, rep in units
                ], axis=1)
                response = contrasts.mean(axis=1)
                between = np.var(contrasts, axis=1, ddof=1).mean(axis=1)
                temporal_signal = np.var(response, axis=1)
                rank_score = temporal_signal / (between + VAR_FLOOR)
                eligible = np.isfinite(rank_score) & (temporal_signal > 1e-12)
                if counts is not None:
                    train_samples = [lookup[(grp, rep, d, t)] for grp, rep in units for d in (ctrl, drug) for t in times]
                    eligible &= ((counts[:, train_samples] >= 10).sum(axis=1) >= 3)
                eligible_ids = np.flatnonzero(eligible)
                if len(eligible_ids) < TOP_GENES:
                    raise ValueError(f"NOT EVALUABLE: {drug}/{held} only {len(eligible_ids)} selectable genes (<200)")
                selected_genes = eligible_ids[np.argsort(-rank_score[eligible_ids], kind='stable')[:TOP_GENES]]
                scales = np.sqrt(between[selected_genes] + VAR_FLOOR)
                template = response[selected_genes, :]
                time_vehicle = np.column_stack([
                    np.mean([X[selected_genes, lookup[(grp, rep, ctrl, t)]] for grp, rep in units], axis=0)
                    for t in times
                ])
                pooled_vehicle = np.mean(time_vehicle, axis=1)
                lam = RIDGE * TOP_GENES
                folds.append({"holdout_group": held, "drug": drug, "training_replicate_pairs": len(units),
                              "training_groups": len(train_groups), "selected_genes": TOP_GENES,
                              "selection_score_sha256": hashlib.sha256(np.asarray(selected_genes, dtype=np.int64).tobytes()).hexdigest()})

                def scores(obs: np.ndarray):
                    gate_null, gate_active = [], []
                    time_score, veh_score = [], []
                    for j in range(len(times)):
                        eff = template[:, j] / scales
                        base = (obs - pooled_vehicle) / scales
                        residual = (obs - time_vehicle[:, j]) / scales
                        def ridge_loss(r):
                            amp = max(0., float(r @ eff + lam) / float(eff @ eff + lam))
                            return float(((r - amp * eff) @ (r - amp * eff) + lam * (amp - 1.)**2) / len(r))
                        gate_null.append(float(np.mean(base**2)))
                        gate_active.append(ridge_loss(base))
                        time_score.append(ridge_loss(residual))
                        veh_score.append(float(np.mean(residual**2)))
                    return (min(gate_active) < min(gate_null),
                            float(min(gate_null)-min(gate_active)),
                            time_score, veh_score)
                test_active = meta.index[(meta.group == held) & (meta.drug == drug)].tolist()
                test_controls = meta.index[(meta.group == held) & (meta.drug == ctrl)].tolist()
                if not test_active or not test_controls:
                    raise ValueError(f"NOT EVALUABLE: no test active/control specimens in holdout {held}")
                for i in test_active:
                    row = meta.iloc[i]
                    gate, margin, ridge, vehicle = scores(X[selected_genes, i])
                    estimate = times[int(np.argmin(ridge))]
                    base_estimate = times[int(np.argmin(vehicle))]
                    predictions.append({
                        "sample": row['sample'], "group": held, "drug": drug,
                        "true_hours": float(row['time']), "predicted_hours": float(estimate),
                        "vehicle_only_predicted_hours": float(base_estimate),
                        "accepted_active": bool(gate),
                        "correct_time": bool(estimate == row['time']),
                        "correct_and_accepted": bool(gate and estimate == row['time']),
                        "gate_margin": margin, "n_training_replicate_pairs": len(units),
                    })
                for i in test_controls:
                    neg_gate, neg_margin, _, _ = scores(X[selected_genes, i])
                    negative_source_calls.setdefault(i, []).append({
                        "drug": str(drug), "accepted_as_active": bool(neg_gate),
                        "gate_margin": float(neg_margin),
                    })
        negative = []
        vehicle_drug_predictions = []
        for i, flags in negative_source_calls.items():
            row = meta.iloc[i]
            for flag in flags:
                vehicle_drug_predictions.append({
                    "sample": str(row['sample']), "group": str(row['group']),
                    "time": float(row['time']), **flag,
                })
            negative.append({"sample": row['sample'], "group": row['group'],
                             "time": float(row['time']),
                             "false_active_any_drug": any(x['accepted_as_active'] for x in flags),
                             "source_programs_checked": len(flags),
                             "max_gate_margin": float(max(x['gate_margin'] for x in flags))})
        n = len(predictions)
        if n == 0 or not negative:
            raise ValueError("NOT EVALUABLE: no heldout predictions")
        result = {
            "status": "EXPLORATORY_GROUPED_HOLDOUT",
            "n_active": n,
            "correct_and_accepted": sum(x['correct_and_accepted'] for x in predictions),
            "correct_duration_without_gate": sum(x['correct_time'] for x in predictions),
            "active_accepted": sum(x['accepted_active'] for x in predictions),
            "vehicle_only_time_correct": sum(x['vehicle_only_predicted_hours'] == x['true_hours'] for x in predictions),
            "n_vehicle": len(negative),
            "vehicle_rejected": sum(not x['false_active_any_drug'] for x in negative),
            "drugs_evaluated": selected,
            "candidate_times_h": [float(x) for x in times],
            "groups": groups,
            "folds": folds,
            "predictions": predictions,
            "vehicle_predictions": negative,
            "vehicle_drug_predictions": vehicle_drug_predictions,
            "v04_audit": build_v04_audit(
                predictions, negative, vehicle_drug_predictions,
                selected, groups, times,
            ),
            "method": {"top_genes": TOP_GENES, "variance_floor": VAR_FLOOR, "ridge_per_gene": RIDGE,
                       "model": "V120 two-stage architecture, training-only vehicle pilot"},
            "limitations": [
                "Known drug identity required for active queries; DMSO any-source false-positive rule is conservative",
                "No test-group vehicle specimen is used as a matching reference or in feature selection",
                "Original V115 trained on two experimental batches; a single training batch with two replicate pairs is a protocol adaptation",
                "The outcome is recorded exposure duration, NOT independently observed transcription onset",
                "No calibrated confidence sets, 95% specificity or methodological novelty can be claimed",
                "Batch and replicate labels do not independently establish biological exchangeability",
            ],
        }
        if self.demo:
            result['status'] = 'SYNTHETIC_DEMO_NOT_EVIDENCE'
        self.last_model = result
        return result

    def report(self) -> dict:
        return {
            "application": "OnsetAudit", "version": VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary(),
            "provenance": self.files,
            "methods": {"input_normalization": ("log2(1+CPM)" if self.matrix_kind=='counts' else "author-provided logCPM"),
                        "feature_filter": "≥10 counts in ≥3 training samples (only for raw counts)",
                        "gene_count": TOP_GENES, "ridge": RIDGE, "variance_floor": VAR_FLOOR},
            "evaluation": self.last_model,
            "nonclaims": ["Not a test of true biological onset", "Not a clinically validated test", "No guarantee of 95% coverage", "Training/test split depends on true batch groups"],
        }
