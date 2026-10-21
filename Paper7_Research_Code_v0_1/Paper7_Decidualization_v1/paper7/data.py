"""Explicit human-review and data-leakage gates for the endometriosis study."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from paper7_pilot import read_gmt

VERSION = "0.1.0"
PRIMARY = "GSE243039"
EXPECTED_PRIMARY_SAMPLES = 40
REVIEW_COLUMNS = ["gse", "gsm", "include", "reviewed", "donor_id", "group", "days",
                  "treatment", "exclusion_reason", "review_provenance"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_review(metadata: pd.DataFrame) -> pd.DataFrame:
    """Do NOT fill the verification fields from automated text heuristics."""
    if not {"gsm", "gse", "title"}.issubset(metadata):
        raise ValueError("Run metadata audit first; input needs gse/gsm/title")
    if metadata.gsm.duplicated().any():
        raise ValueError("Duplicate GSMs in metadata")
    result = pd.DataFrame({
        "gse": metadata.gse,
        "gsm": metadata.gsm,
        "include": "", "reviewed": "", "donor_id": "", "group": "", "days": "",
        "treatment": "", "exclusion_reason": "", "review_provenance": "",
        "GEO_title_UNVERIFIED": metadata.title,
        "suggested_donor_UNVERIFIED": metadata.get("donor_local", pd.Series([""]*len(metadata))).fillna(""),
        "suggested_group_UNVERIFIED": metadata.get("group", pd.Series([""]*len(metadata))).fillna(""),
        "suggested_days_UNVERIFIED": pd.to_numeric(metadata.get("time_hours", pd.Series([np.nan]*len(metadata))), errors="coerce") / 24,
    })
    return result


def is_one(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validated_review(path: Path, expected_gsm: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REVIEW_COLUMNS) - set(df)
    if missing:
        raise ValueError(f"Review missing fields: {sorted(missing)}")
    if df.gsm.duplicated().any() or df.gsm.eq("").any():
        raise ValueError("Each GSM must occur once and be nonempty")
    if expected_gsm is not None and set(df.gsm) != expected_gsm:
        raise ValueError("Review GSMs do not match audited primary metadata (including excluded samples)")
    if not df.gse.eq(PRIMARY).all():
        raise ValueError("Stage-1 fits only primary study GSE243039; do not pool protocols")
    if len(df) != EXPECTED_PRIMARY_SAMPLES and expected_gsm is None:
        raise ValueError("GSE243039 review should include all 40 GEO samples before exclusions")
    if not df.reviewed.map(is_one).all():
        raise ValueError("Every GSM requires reviewed=1, including excluded samples")
    df["include_bool"] = df.include.map(is_one)
    no_reason = (~df.include_bool) & df.exclusion_reason.str.strip().eq("")
    if no_reason.any():
        raise ValueError("Excluded samples must have exclusion_reason")
    yes = df.loc[df.include_bool].copy()
    if yes.empty:
        raise ValueError("No included samples")
    if yes[["donor_id", "group", "days", "treatment", "review_provenance"]].eq("").any().any():
        raise ValueError("Included samples need donor/group/days/treatment/review_provenance")
    if not yes.group.isin(["control", "endometriosis"]).all():
        raise ValueError("Verified group must be control or endometriosis")
    if not yes.treatment.eq("EPC").all():
        raise ValueError("This analysis is restricted to EPC-treated GSE243039; no protocol pooling")
    yes["days"] = pd.to_numeric(yes.days, errors="raise")
    if not np.isfinite(yes.days).all() or (yes.days < 0).any():
        raise ValueError("Non-finite or negative verified time")
    donors_to_groups = yes.groupby("donor_id").group.nunique()
    if donors_to_groups.max() > 1:
        raise ValueError("Donor IDs overlap across diagnosis groups: verify identity")
    if yes.duplicated(["donor_id", "days", "treatment"]).any():
        raise ValueError("More than one sample per donor/time/protocol: review technical replicates explicitly")
    for grp in ["control", "endometriosis"]:
        sub = yes[yes.group == grp]
        if sub.donor_id.nunique() < 3:
            raise ValueError(f"Need >=3 independently identified {grp} donors")
        if (sub.groupby("donor_id").days.nunique() < 5).any():
            raise ValueError(f"Each included {grp} donor needs 5 timepoints for primary LOTO comparison")
        if not set([0., 2., 4., 6., 8.]).issubset(set(sub.days)):
            raise ValueError(f"Group {grp} does not cover all original study timepoints")
    return yes


def create_lock(review_path: Path, gmt_path: Path, metadata_path: Path, out: Path,
                biological_note: str) -> dict:
    """Freeze inputs before examining case trajectories. Timestamp is not preregistration."""
    if len(biological_note.strip()) < 30:
        raise ValueError("Provide a >=30-character independent biological justification for programs")
    if "PROVISIONAL" in gmt_path.name.upper() or "DEMO" in gmt_path.name.upper():
        raise ValueError("Cannot freeze provisional/demo program file as a biological hypothesis")
    text = gmt_path.read_text(encoding="utf-8")
    if "EXPLORATORY" in text.upper() or "PROVISIONAL" in text.upper():
        raise ValueError("Explicitly curate gene sets and remove provisional/demo annotation before freezing")
    programs = read_gmt(gmt_path)
    if not programs or any(len(set(v)) < 3 for v in programs.values()):
        raise ValueError("Provide >=1 curated gene program with >=3 unique genes")
    meta = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    review = validated_review(review_path, expected_gsm=set(meta.gsm))
    # Guard against accidentally swapping disease labels or day fields during curation.
    # These are consistency checks, NOT verification of biological donor identity.
    if "group" in meta.columns:
        groups = meta.set_index("gsm")["group"].reindex(review.gsm)
        known = groups.notna() & groups.ne("unknown")
        contradictions = review.loc[known.to_numpy() & (review.group.to_numpy() != groups.to_numpy())]
        if not contradictions.empty:
            raise ValueError(f"Reviewed diagnosis contradicts GEO annotation for {list(contradictions.gsm)[:6]}")
    if "time_hours" in meta.columns:
        source_days = pd.to_numeric(meta.set_index("gsm")["time_hours"].reindex(review.gsm), errors="coerce") / 24
        mismatch = source_days.notna().to_numpy() & (np.abs(review.days.to_numpy() - source_days.to_numpy()) > 0.01)
        if mismatch.any():
            raise ValueError(f"Reviewed time contradicts GEO annotation for {list(review.loc[mismatch,'gsm'])[:6]}")
    lock = {
        "version": VERSION, "dataset": PRIMARY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "EPC in vitro donor-level response onset delay vs amplitude; no causal/clinical claims",
        "review_sha256": digest(review_path), "gmt_sha256": digest(gmt_path),
        "geo_audit_sha256": digest(metadata_path),
        "programs": sorted(programs), "n_curated_programs": len(programs),
        "n_included_samples": len(review), "n_controls": int(review.query('group=="control"').donor_id.nunique()),
        "n_cases": int(review.query('group=="endometriosis"').donor_id.nunique()),
        "biological_justification": biological_note,
        "model": {"lag_days_min": 0, "lag_days_max": 2, "grid_points": 81,
                  "amplitude_min": 0, "amplitude_max": 2.5,
                  "validation": "held-out timepoints aggregated by donor; reference LODO sensitivity",
                  "selection": "predefined programs; NO outcome-based gene selection"}
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Do not overwrite existing frozen plan: {out}")
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


def verify_lock(lock_path: Path, review_path: Path, gmt_path: Path, metadata_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("version") != VERSION or lock.get("dataset") != PRIMARY:
        raise ValueError("Wrong lock version/dataset")
    for k, p in [("review_sha256", review_path), ("gmt_sha256", gmt_path),
                 ("geo_audit_sha256", metadata_path)]:
        if lock[k] != digest(p):
            raise ValueError(f"Input changed after freeze: {p} ({k})")
    return lock


def score_all_samples(symbol_expr: pd.DataFrame, review: pd.DataFrame,
                      programs: dict[str, list[str]], kind: str = "normalized_counts",
                      train_control_gsms: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit *all* gene centering, scaling and inclusion on CONTROLS only.

    Input: HGNC-symbol-indexed gene x GSM matrix, already unambiguously mapped.
    Patient data do not affect feature scaling, coverage or reference trajectories.
    """
    all_ids = list(review.gsm)
    if len(all_ids) != len(set(all_ids)) or not set(all_ids).issubset(symbol_expr.columns):
        raise ValueError("Not every reviewed GSM has an unambiguous expression column")
    x = symbol_expr.loc[:, all_ids].apply(pd.to_numeric, errors="raise")
    if (x < 0).any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Counts matrix has invalid negative/non-finite values")
    if kind == "raw_counts":
        # For raw counts, normalize sample library sizes before learning anything on controls.
        sizes = x.sum(axis=0)
        if (sizes <= 0).any():
            raise ValueError("A sample has zero library size")
        logx = np.log2(x.div(sizes, axis=1) * 1e6 + 1)
    elif kind == "normalized_counts":
        logx = np.log2(x + 1)
    else:
        raise ValueError(f"Unknown expression type {kind}")
    controls = list(review.loc[review.group == "control", "gsm"]) if train_control_gsms is None else list(train_control_gsms)
    if not set(controls).issubset(set(review.loc[review.group == "control", "gsm"])):
        raise ValueError("Training IDs must refer exclusively to verified controls")
    if len(controls) < 12:
        raise ValueError("Need >=3 controls with five points for scaling")
    training = logx.loc[:, controls]
    means = training.mean(axis=1)
    std = training.std(axis=1, ddof=1)
    valid = training.notna().all(axis=1) & (std > 1e-8)
    z = logx.loc[valid].sub(means[valid], axis=0).div(std[valid], axis=0)
    z.index = z.index.astype(str).str.upper()
    # Duplicate gene symbols collapsed upstream in gene_symbols_and_expression.
    records, coverage = [], []
    for name, genes in programs.items():
        selected = sorted(set(g.upper() for g in genes) & set(z.index))
        coverage.append({"program": name, "n_requested": len(genes),
                         "n_present_in_controls": len(selected), "present_genes": ";".join(selected)})
        if len(selected) < 3:
            continue
        scored = z.loc[selected].mean(axis=0)
        records.extend({"gsm": gsm, "program": name, "score": float(val)} for gsm, val in scored.items())
    if not records:
        raise ValueError("None of the curated gene programs has >=3 measured genes")
    scores = pd.DataFrame(records).merge(review[["gsm", "donor_id", "group", "days"]], on="gsm", how="left", validate="many_to_one")
    norms = pd.DataFrame({"gene": z.index, "control_log_mean": means.loc[z.index].to_numpy(),
                          "control_log_sd": std.loc[z.index].to_numpy()})
    return scores, pd.DataFrame(coverage), norms
