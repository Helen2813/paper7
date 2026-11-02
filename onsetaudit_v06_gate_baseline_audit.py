#!/usr/bin/env python3
"""Paper 7 / OnsetAudit v0.6: frozen-gate pooled-versus-time-matched DMSO audit.

NO FLAGS | OFFLINE | MAIN ONLY | NO MODEL/GATE/FEATURE RETUNING.

This retrospective falsification test reconstructs the *existing* training-only
folds and checks the resulting held-out scores against the frozen v0.4 report.
It then substitutes the training-only DMSO reference at each sample's *known*
recorded time for the training-only pooled DMSO reference. The substitution is
ORACLE, post-hoc, and cannot be deployed for an unknown exposure duration.

Requires existing local GSE194200 import and v0.4/v0.5 reports. Produces HTML,
CSV, JSON, console log, SHA-256 manifest, and ZIP. Changes no existing files.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "onsetaudit/_generated/import_ready/GSE194200"
REPORT_BASE = ROOT / "onsetaudit/_generated/reports"
VERSION = "v0.6.0"
EXPECTED_METADATA_SHA = "ad34e6824ca5aa3a19fef28baf07ffbe759bf5360818f1ffdbebee6ad409256e"
EXPECTED_COUNTS_SHA = "3bc491413506f2787375d3447925952293f335fd1bb175f71686e43475700ab5"
DRUGS = ("SW016789", "Thapsigargin")
GROUPS = ("experiment_1", "experiment_2", "experiment_3")
TIMES = (1., 2., 6., 24.)
FPS = {"GSM5830673", "GSM5830674"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


class Log:
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8")

    def write(self, *parts) -> None:
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def verify_manifest(folder: Path) -> dict:
    file = folder / "sha256_manifest.json"
    require(file.is_file(), f"Missing hash manifest: {file}")
    manifest = json.loads(file.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and len(manifest) >= 2,
            f"Malformed SHA-256 manifest: {file}")
    for name, expected in manifest.items():
        require(Path(name).name == name, "Unsafe manifest entry")
        source = folder / name
        require(source.is_file() and digest(source) == expected,
                f"Frozen report file missing or modified: {source}")
    return manifest


def inputs_and_reports(log: Log):
    require((ROOT / ".git").is_dir(), "Expected an existing Git repository")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            check=True, text=True, capture_output=True).stdout.strip()
    require(branch == "main", "Only existing main branch supported")
    git_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, check=True, text=True,
                              capture_output=True).stdout.strip()
    log.write("[GIT]", branch, git_head)

    prior = sorted(REPORT_BASE.glob("v05_dmso_qc_*/audit_summary.json"),
                   key=lambda p: p.parent.name, reverse=True)
    require(prior, "Missing v0.5 audit: run onsetaudit_v05_qc_audit.py first")
    v05_file = prior[0]
    verify_manifest(v05_file.parent)
    v05 = json.loads(v05_file.read_text(encoding="utf-8"))
    require(v05.get("audit_version") == "v0.5.0", "Unexpected v0.5 audit version")
    require(v05.get("frozen_correct_and_accepted") == 22 and
            v05.get("frozen_vehicle_rejected") == 10 and
            set(v05.get("DMSO_false_samples", [])) == FPS,
            "v0.5 frozen performance/false IDs differ")

    v04_file = Path(v05.get("source_v04_report", "")).resolve()
    require(v04_file.is_relative_to(REPORT_BASE.resolve()) and v04_file.is_file(),
            "Invalid local v0.4 source path")
    verify_manifest(v04_file.parent)
    require(digest(v04_file) == v05.get("source_v04_report_sha256"),
            "v0.4 report differs from the verified v0.5 source")
    v04 = json.loads(v04_file.read_text(encoding="utf-8"))
    require(v04.get("version") == "0.4.0", "Expected OnsetAudit v0.4.0 report")
    ev = v04.get("evaluation") or {}
    require(ev.get("n_active") == 24 and ev.get("correct_and_accepted") == 22 and
            ev.get("active_accepted") == 24 and ev.get("n_vehicle") == 12 and
            ev.get("vehicle_rejected") == 10,
            "Unexpected v0.4 frozen benchmark; stop without re-tuning")
    require(ev.get("v04_audit", {}).get("frozen_gate_margin_threshold") == 0.,
            "Gate threshold differs from zero")
    require(ev.get("groups") == list(GROUPS) and
            ev.get("candidate_times_h") == list(TIMES) and
            set(ev.get("drugs_evaluated", [])) == set(DRUGS),
            "Frozen groups, drugs, or times differ")

    metadata_file = DATA / "metadata.csv"
    counts_file = DATA / "expression_counts.csv.gz"
    for path, expected, key in ((metadata_file, EXPECTED_METADATA_SHA, "metadata"),
                                (counts_file, EXPECTED_COUNTS_SHA, "expression")):
        require(path.is_file() and digest(path) == expected,
                f"Missing or changed original input: {path}")
        require(v04.get("provenance", {}).get(key, {}).get("sha256") == expected and
                v05.get("source_" + key + "_sha256") == expected,
                f"Source {key} SHA-256 differs between frozen audits")

    log.write("[VERIFY] v0.4 and v0.5 manifests + original data SHA-256: OK")
    log.write("[V04]", v04_file)
    log.write("[V05]", v05_file)
    return git_head, ev, v04_file, v05_file, metadata_file, counts_file


def dataset_and_replay(ev: dict, metadata_file: Path, counts_file: Path, log: Log):
    sys.path.insert(0, str(ROOT / "onsetaudit"))
    from core import Dataset, VERSION as CORE_VERSION  # imported only after checking paths
    require(CORE_VERSION == "0.4.0", "Current core is not frozen v0.4.0")

    ds = Dataset()
    ds.import_metadata(metadata_file.read_bytes(), metadata_file.name)
    summary = ds.import_matrix(counts_file.read_bytes(), counts_file.name, "counts")
    require(summary["genes"] == 24421 and summary["samples"] == 36 and
            summary["controls"] == 12, "Incorrect verified input dimensions")
    model = ds.evaluate()
    for key in ("n_active", "correct_and_accepted", "correct_duration_without_gate",
                "active_accepted", "vehicle_only_time_correct", "n_vehicle", "vehicle_rejected"):
        require(model[key] == ev[key], f"Frozen replay changed statistic: {key}")

    def by_key(items, fields):
        result = {}
        for row in items:
            key = tuple(row[f] for f in fields)
            require(key not in result, f"Duplicated frozen/replayed key: {key}")
            result[key] = row
        return result

    for name, fields, numeric, categorical in (
        ("predictions", ("sample",),
         ("gate_margin", "true_hours", "predicted_hours", "vehicle_only_predicted_hours"),
         ("group", "drug", "accepted_active", "correct_time", "correct_and_accepted")),
        ("vehicle_predictions", ("sample",),
         ("max_gate_margin", "time"), ("group", "false_active_any_drug")),
        ("vehicle_drug_predictions", ("sample", "drug"),
         ("gate_margin", "time"), ("group", "accepted_as_active")),
        ("folds", ("holdout_group", "drug"),
         ("training_replicate_pairs", "selected_genes"),
         ("selection_score_sha256",))):
        a = by_key(ev[name], fields)
        b = by_key(model[name], fields)
        require(a.keys() == b.keys(), f"Frozen versus replayed keys differ: {name}")
        for k in a:
            for attr in numeric:
                require(math.isclose(float(a[k][attr]), float(b[k][attr]),
                                     abs_tol=2e-9, rel_tol=2e-10),
                        f"Frozen replay mismatch {name}/{k}/{attr}")
            for attr in categorical:
                require(a[k][attr] == b[k][attr],
                        f"Frozen replay mismatch {name}/{k}/{attr}")
    log.write("[REPLAY] Exact metadata, selected-gene hashes, 24 active + 24 drug-specific control margins: OK")
    log.write("[FROZEN] 22/24 correct+accepted; 10/12 DMSO rejected; 2 unique false positives")
    return ds, model


def ridge_loss(residual: np.ndarray, effect: np.ndarray, lam: float) -> float:
    amplitude = max(0., float(residual @ effect + lam) / float(effect @ effect + lam))
    return float(((residual - amplitude * effect) @
                  (residual - amplitude * effect) + lam * (amplitude - 1.)**2) / len(residual))


def evaluate_gate(obs: np.ndarray, reference: np.ndarray,
                  scales: np.ndarray, effects: np.ndarray, lam: float) -> tuple[float, float, float, int]:
    base = (obs - reference) / scales
    null_loss = float(np.mean(base ** 2))
    losses = [ridge_loss(base, effects[:, j], lam) for j in range(effects.shape[1])]
    best_j = int(np.argmin(losses))
    alt_loss = float(losses[best_j])
    return null_loss - alt_loss, null_loss, alt_loss, best_j


def model_component_audit(ds, model, log: Log):
    from core import TOP_GENES, VAR_FLOOR, RIDGE

    meta, X, counts = ds.meta, ds.matrix, ds.counts
    require(X is not None and counts is not None, "Expected raw counts + normalized matrix")
    require((TOP_GENES, VAR_FLOOR, RIDGE) == (200, .25, .25),
            "Frozen gene selection/ridge constants differ")
    require(set(meta["group"]) == set(GROUPS) and set(meta["drug"]) == {"DMSO", *DRUGS},
            "Unexpected group/drug design")
    require(set(meta["time"]) == set(TIMES) and len(meta) == 36, "Unexpected time design")
    require(set(meta["replicate"]) == {"r1"}, "Frozen pairing differs from original import")

    lookup = {}
    for idx, row in meta.iterrows():
        key = (row["group"], row["replicate"], row["drug"], float(row["time"]))
        require(key not in lookup, f"Duplicate design cell: {key}")
        lookup[key] = idx

    frozen_active = {p["sample"]: p for p in model["predictions"]}
    frozen_neg = {(p["sample"], p["drug"]): p for p in model["vehicle_drug_predictions"]}
    frozen_folds = {(p["holdout_group"], p["drug"]): p for p in model["folds"]}
    programs = {}
    rows = []
    for held in GROUPS:
        for drug in DRUGS:
            train_groups = [g for g in GROUPS if g != held]
            units = [(grp, rep) for grp in train_groups
                     for rep in sorted(meta.loc[meta.group == grp, "replicate"].unique())
                     if all((grp, rep, d, t) in lookup for d in ("DMSO", drug) for t in TIMES)]
            require(len(units) == 2, f"Unexpected training pairs for {held}/{drug}")
            contrasts = np.stack([
                np.column_stack([X[:, lookup[(grp, rep, drug, t)]] -
                                 X[:, lookup[(grp, rep, "DMSO", t)]] for t in TIMES])
                for grp, rep in units], axis=1)
            response = contrasts.mean(axis=1)
            between = np.var(contrasts, axis=1, ddof=1).mean(axis=1)
            temporal_signal = np.var(response, axis=1)
            rank_score = temporal_signal / (between + VAR_FLOOR)
            eligible = np.isfinite(rank_score) & (temporal_signal > 1e-12)
            train_samples = [lookup[(grp, rep, d, t)] for grp, rep in units
                             for d in ("DMSO", drug) for t in TIMES]
            eligible &= ((counts[:, train_samples] >= 10).sum(axis=1) >= 3)
            selectable = np.flatnonzero(eligible)
            require(len(selectable) >= TOP_GENES, f"Insufficient genes in {held}/{drug}")
            genes = selectable[np.argsort(-rank_score[selectable], kind="stable")[:TOP_GENES]]
            ghash = hashlib.sha256(np.asarray(genes, dtype=np.int64).tobytes()).hexdigest()
            require(ghash == frozen_folds[(held, drug)]["selection_score_sha256"],
                    f"Gene selection differs from frozen v0.4: {held}/{drug}")
            scales = np.sqrt(between[genes] + VAR_FLOOR)
            template = response[genes, :]
            effect = template / scales[:, None]
            train_vehicle = np.column_stack([
                np.mean([X[genes, lookup[(grp, rep, "DMSO", t)]]
                         for grp, rep in units], axis=0) for t in TIMES])
            pooled = np.mean(train_vehicle, axis=1)
            programs[(held, drug)] = {
                "genes": genes, "scales": scales, "effects": effect,
                "template_unscaled": template, "vehicle_by_time": train_vehicle,
            }
            lam = RIDGE * TOP_GENES
            for treatment in ("DMSO", drug):
                for t in TIMES:
                    sample_idx = lookup[(held, "r1", treatment, t)]
                    sample = str(meta.loc[sample_idx, "sample"])
                    obs = X[genes, sample_idx]
                    time_i = TIMES.index(t)
                    original, null0, active0, best0 = evaluate_gate(obs, pooled, scales, effect, lam)
                    matched, null1, active1, best1 = evaluate_gate(obs, train_vehicle[:, time_i], scales, effect, lam)
                    frozen = (frozen_neg[(sample, drug)] if treatment == "DMSO"
                              else frozen_active[sample])
                    require(abs(original - float(frozen["gate_margin"])) < 2e-9,
                            f"Not an exact original-gate replay: {sample}/{drug}")
                    shift = (train_vehicle[:, time_i] - pooled) / scales
                    rows.append({
                        "sample": sample, "heldout_group": held, "treatment": treatment,
                        "program": drug, "true_time_h": t,
                        "is_vehicle": treatment == "DMSO",
                        "is_original_false_positive": bool(treatment == "DMSO" and original > 0),
                        "frozen_gate_margin": float(frozen["gate_margin"]),
                        "replayed_pooled_margin": original,
                        "oracle_known_time_matched_margin": matched,
                        "pooled_gate_positive": original > 0.,
                        "oracle_known_time_gate_positive": matched > 0.,
                        "margin_change_oracle_minus_pooled": matched - original,
                        "pooled_reference_null_loss": null0,
                        "pooled_reference_best_active_loss": active0,
                        "oracle_reference_null_loss": null1,
                        "oracle_reference_best_active_loss": active1,
                        "best_template_hour_pooled": TIMES[best0],
                        "best_template_hour_oracle": TIMES[best1],
                        "train_vehicle_temporal_shift_rms_scaled": float(np.sqrt(np.mean(shift ** 2))),
                        "train_vehicle_temporal_shift_approx_median_abs_scaled": float(np.median(np.abs(shift))),
                    })
    df = pd.DataFrame(rows)
    require(len(df) == 48 and df["sample"].nunique() == 36,
            "Expected exactly 48 held-out sample × drug program rows")
    require(int(df.is_original_false_positive.sum()) == 4,
            "Expected four dependent false positive program calls")
    actual_fp = set(df.loc[df.is_original_false_positive, "sample"])
    require(actual_fp == FPS, f"Unexpected false positives: {actual_fp}")
    log.write("[REPLAY] 6 feature sets and all 48 original margins match frozen report")

    vehicle = df[df.is_vehicle].copy()
    summary_rows = []
    for sample, g in vehicle.groupby("sample", sort=True):
        require(len(g) == 2 and set(g.program) == set(DRUGS), "Unpaired DMSO scores")
        x, y = [g.loc[g.program == drug].iloc[0] for drug in DRUGS]
        summary_rows.append({
            "sample": sample, "heldout_group": x.heldout_group, "true_time_h": x.true_time_h,
            "pooled_false_any_program": bool(g.pooled_gate_positive.any()),
            "oracle_matched_false_any_program": bool(g.oracle_known_time_gate_positive.any()),
            "pooled_margin_SW016789": float(x.replayed_pooled_margin),
            "pooled_margin_Thapsigargin": float(y.replayed_pooled_margin),
            "oracle_margin_SW016789": float(x.oracle_known_time_matched_margin),
            "oracle_margin_Thapsigargin": float(y.oracle_known_time_matched_margin),
            "pooled_max_margin": float(g.replayed_pooled_margin.max()),
            "oracle_matched_max_margin": float(g.oracle_known_time_matched_margin.max()),
            "sw_temporal_vehicle_shift_rms": float(x.train_vehicle_temporal_shift_rms_scaled),
            "thapsigargin_temporal_vehicle_shift_rms": float(y.train_vehicle_temporal_shift_rms_scaled),
        })
    sample_df = pd.DataFrame(summary_rows)
    require(len(sample_df) == 12 and int(sample_df.pooled_false_any_program.sum()) == 2,
            "Frozen controls not preserved in sample summary")

    align_rows = []
    for held in GROUPS:
        a, b = programs[(held, DRUGS[0])], programs[(held, DRUGS[1])]
        left, right = set(a["genes"]), set(b["genes"])
        common = sorted(left & right)
        overlap = len(common)
        align = {
            "heldout_group": held,
            "gene_overlap_count": overlap,
            "gene_overlap_jaccard": overlap / len(left | right),
        }
        a_idx = {int(g): i for i, g in enumerate(a["genes"])}
        b_idx = {int(g): i for i, g in enumerate(b["genes"])}
        for j, t in enumerate(TIMES):
            if len(common) >= 2:
                vec_a = np.array([a["effects"][a_idx[g], j] for g in common])
                vec_b = np.array([b["effects"][b_idx[g], j] for g in common])
                denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                cos = float(vec_a @ vec_b / denominator) if denominator > 1e-12 else None
            else:
                cos = None
            align[f"common_gene_template_cosine_{int(t)}h"] = cos
        align_rows.append(align)
    align_df = pd.DataFrame(align_rows)
    log.write("[DIAGNOSTIC] Known-time oracle is post-hoc; original model is not changed")
    log.write("[ORACLE DMSO] Any-program false calls", int(sample_df.oracle_matched_false_any_program.sum()),
              "/12 vs frozen 2/12")
    for sample in sorted(FPS):
        r = sample_df.loc[sample_df["sample"] == sample].iloc[0]
        log.write(f"[FP] {sample} | pooled SW={r.pooled_margin_SW016789:+.4f} Thap={r.pooled_margin_Thapsigargin:+.4f}"
                  f" | known-time SW={r.oracle_margin_SW016789:+.4f} Thap={r.oracle_margin_Thapsigargin:+.4f}"
                  f" | oracle any-program false={r.oracle_matched_false_any_program}")
    for row in align_df.itertuples():
        log.write(f"[GENE OVERLAP] {row.heldout_group}: {row.gene_overlap_count}/200 shared;"
                  f" Jaccard={row.gene_overlap_jaccard:.3f}")
    return df, sample_df, align_df


def table_html(df: pd.DataFrame) -> str:
    temp = df.copy()
    for col in temp.columns:
        if pd.api.types.is_bool_dtype(temp[col]):
            temp[col] = temp[col].map({True: "Yes", False: "No"})
        elif pd.api.types.is_float_dtype(temp[col]):
            temp[col] = temp[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    return temp.to_html(index=False, border=0, classes="data", escape=True)


def scatter_svg(vehicle: pd.DataFrame) -> str:
    width, height = 850, 440
    xl, xr, yt, yb = 85, 50, 40, 70
    vals = pd.concat([vehicle.replayed_pooled_margin, vehicle.oracle_known_time_matched_margin])
    lo, hi = float(min(vals.min(), 0)), float(max(vals.max(), 0))
    pad = max(.09, (hi - lo) * .15)
    lo, hi = lo - pad, hi + pad
    fx = lambda z: xl + (z - lo) / (hi - lo) * (width - xl - xr)
    fy = lambda z: height - yb - (z - lo) / (hi - lo) * (height - yt - yb)
    zero_x, zero_y = fx(0), fy(0)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pooled versus known-time DMSO margins">',
           f'<line x1="{zero_x:.1f}" y1="{yt}" x2="{zero_x:.1f}" y2="{height-yb}" stroke="#94a3b8" stroke-dasharray="5 5"/>',
           f'<line x1="{xl}" y1="{zero_y:.1f}" x2="{width-xr}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-dasharray="5 5"/>']
    for p in vehicle.itertuples():
        x, y = fx(p.replayed_pooled_margin), fy(p.oracle_known_time_matched_margin)
        color = "#7966e6" if p.program == DRUGS[0] else "#15aaa4"
        radius = 7 if p.is_original_false_positive else 4
        stroke = "#d02b47" if p.is_original_false_positive else "white"
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="2"><title>{html.escape(p.sample)} {html.escape(p.program)}: pooled {p.replayed_pooled_margin:+.4f}, oracle {p.oracle_known_time_matched_margin:+.4f}</title></circle>')
        if p.is_original_false_positive:
            out.append(f'<text x="{x+10:.1f}" y="{y-5:.1f}" font-size="11" fill="#17283e">{html.escape(p.sample[-4:])}</text>')
    out.extend([f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="14">Frozen pooled-control margin (original)</text>',
                f'<text transform="translate(22,{height/2}) rotate(-90)" text-anchor="middle" font-size="14">Known-time matched margin (oracle diagnostic)</text>',
                f'<text x="{xl}" y="20" font-size="12">Purple = SW016789 · Teal = Thapsigargin · Red outline = frozen false positive</text>',
                '</svg>'])
    return "".join(out)


def html_dashboard(samples: pd.DataFrame, long: pd.DataFrame, align: pd.DataFrame,
                   metadata_sha: str, counts_sha: str, v04_path: Path) -> str:
    false = samples.loc[samples.pooled_false_any_program]
    new_false = samples.loc[samples.oracle_matched_false_any_program]
    changed = int((samples.pooled_false_any_program != samples.oracle_matched_false_any_program).sum())
    body = '''<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OnsetAudit v0.6 | Pooled versus time-matched DMSO reference</title><style>
body{margin:0;background:#f3f5fa;color:#17263c;font-family:system-ui,-apple-system,Arial,sans-serif} main{max-width:1150px;margin:32px auto;padding:0 20px} h1{font-size:26px;margin:0 0 6px}h2{font-size:20px;margin:25px 0 10px}p{line-height:1.55;color:#536277}small{color:#6b7a90}.card{background:#fff;border:1px solid #dce4f0;border-radius:13px;padding:19px;margin:12px 0;overflow-x:auto}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.stats .card{margin:0}.stats b{font-size:28px;display:block}.warning{border-left:5px solid #d99b2b;background:#fff8e8;padding:15px 19px;border-radius:10px;margin:18px 0;line-height:1.5}.data{font-size:12px;border-collapse:collapse;width:100%}.data td,.data th{padding:9px;text-align:left;border-bottom:1px solid #e5eaf3;white-space:nowrap}.data th{background:#eff3fa}svg{width:100%;height:auto}.sub{color:#6a788d}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:0 10px}.stats{grid-template-columns:1fr}}
</style></head><body><main>'''
    body += '<h1>OnsetAudit v0.6 · Gate baseline falsification</h1>'
    body += '<p class="sub">GSE194200 · verified replay of v0.4 · retrospective descriptive only</p>'
    body += f'<div class="stats"><div class="card"><b>22/24</b><small>Frozen active correct + accepted</small></div><div class="card"><b>10/12</b><small>Frozen DMSO rejected</small></div><div class="card"><b>{len(new_false)}/12</b><small>Oracle known-time DMSO false positives (NOT model performance)</small></div></div>'
    body += '<div class="warning"><b>CRITICAL: This is NOT an improved classifier.</b> The alternate gate uses each sample\'s recorded true time to pick the corresponding DMSO baseline. For unknown duration, that time is unavailable. This is a post-hoc test of whether the pooled DMSO reference might contribute to errors. No feature, threshold, prediction, or model parameters were changed; no calibration or causal inference is implied.</div>'
    body += f'<p>Frozen false-positive samples: {", ".join(html.escape(s) for s in false["sample"])}. Any-program control decisions differing under oracle substitution: {changed}/12. The two drug programs reuse the same samples.</p>'
    body += '<h2>Control margins before and after known-time baseline</h2><div class="card">' + scatter_svg(long.loc[long.is_vehicle]) + '</div>'
    body += '<h2>All 12 control samples</h2><div class="card">' + table_html(samples) + '</div>'
    body += '<h2>Training-fold gene overlap & program alignment</h2><div class="card">' + table_html(align) + '</div>'
    body += '<h2>All 48 sample × program gate decompositions</h2><div class="card">' + table_html(long[["sample", "heldout_group", "treatment", "program", "true_time_h", "frozen_gate_margin", "oracle_known_time_matched_margin", "margin_change_oracle_minus_pooled", "pooled_reference_null_loss", "pooled_reference_best_active_loss", "oracle_reference_null_loss", "oracle_reference_best_active_loss", "train_vehicle_temporal_shift_rms_scaled"]]) + '</div>'
    body += '<div class="warning">The original gate uses pooled TRAINING DMSO over all four times. The oracle diagnostic instead uses the TRAINING DMSO reference at the sample\'s KNOWN time. Both compare with the existing four drug-response templates. Group holdout is maintained; model selection and reported frozen prediction are unchanged. Observed differences must not be turned into a tuned gate based on this cohort.</div>'
    body += f'<p><small>Frozen v0.4: <code>{html.escape(str(v04_path))}</code><br>Metadata SHA-256: <code>{metadata_sha}</code><br>Counts SHA-256: <code>{counts_sha}</code></small></p>'
    return body + '</main></body></html>'


def run(out: Path, log: Log) -> None:
    log.write("=" * 79)
    log.write("PAPER 7 | ONSETAUDIT v0.6 | FROZEN GATE BASELINE FALSIFICATION")
    log.write("OFFLINE | NO FLAGS | MAIN ONLY | NO GENE/THRESHOLD RETUNING")
    log.write("=" * 79)
    head, ev, path04, path05, metadata_path, counts_path = inputs_and_reports(log)
    ds, model = dataset_and_replay(ev, metadata_path, counts_path, log)
    long, samples, align = model_component_audit(ds, model, log)
    long.to_csv(out / "all_48_gate_decompositions.csv", index=False)
    samples.to_csv(out / "all_12_dmso_pooled_vs_oracle.csv", index=False)
    align.to_csv(out / "training_gene_overlap_and_template_alignment.csv", index=False)
    (out / "onsetaudit_v06_dashboard.html").write_text(
        html_dashboard(samples, long, align, EXPECTED_METADATA_SHA, EXPECTED_COUNTS_SHA, path04),
        encoding="utf-8")
    original = int(samples.pooled_false_any_program.sum())
    oracle = int(samples.oracle_matched_false_any_program.sum())
    fp_rows = samples.loc[samples["sample"].isin(FPS)]
    summary = {
        "audit_version": VERSION,
        "status": "RETROSPECTIVE_KNOWN_TIME_ORACLE_DIAGNOSTIC_NOT_A_MODEL",
        "dataset": "GSE194200",
        "git_commit_at_run": head,
        "frozen_v04_report_path": str(path04), "frozen_v04_report_sha256": digest(path04),
        "frozen_v05_audit_path": str(path05), "frozen_v05_audit_sha256": digest(path05),
        "metadata_sha256": EXPECTED_METADATA_SHA, "counts_sha256": EXPECTED_COUNTS_SHA,
        "n_active": 24, "n_distinct_vehicle": 12, "n_drug_program_control_calls": 24,
        "frozen_correct_and_accepted": 22, "frozen_vehicle_rejected": 10,
        "frozen_false_positive_distinct": original,
        "oracle_known_time_false_positive_distinct": oracle,
        "frozen_fp_that_remain_oracle_positive": int(fp_rows.oracle_matched_false_any_program.sum()),
        "oracle_differs_from_original_any_control": int((samples.pooled_false_any_program != samples.oracle_matched_false_any_program).sum()),
        "frozen_selected_gene_hashes_reproduced": True,
        "all_48_original_margins_reproduced": True,
        "holdout_experiment_exclusion_replayed": True,
        "diagnostic_warning": "Known TRUE time is used for matched reference; NOT an operational model, prospective validation, or calibrated classification; never tune using these observed holdout results",
        "limitations": [
            "Pooled-vs-true-time-conditioned baseline is post-hoc; changes are not an external test",
            "The oracle reference uses the recorded TRUE hour and cannot be deployed when duration is unknown",
            "Gene alignment on two treatment programs may reflect a shared biological response or preprocessing artifacts",
            "Three experimental groups, 12 shared controls and only one cell line are insufficient for generalization",
            "Biological molecular onset is not measured by this experiment",
            "No threshold, gene, model change or performance improvement has been made",
        ],
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
    }
    save_json(out / "audit_summary.json", summary)
    (out / "README.txt").write_text(
        "OnsetAudit v0.6: frozen pooled DMSO versus KNOWN-TIME DMSO diagnostic.\n"
        "Replay checks frozen model, gene hashes, 24 active and 24 drug-specific control margins.\n"
        "This is NOT a deployable time inference model: the alternate baseline uses the true hour.\n"
        "Do NOT choose new gate thresholds or feature sets based on these holdouts.\n"
        "The ZIP includes results, report and code; raw counts remain local and are SHA-256 identified.\n",
        encoding="utf-8")
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    log.write("[OUTPUT]", out)
    log.write("[DASHBOARD]", out / "onsetaudit_v06_dashboard.html")
    log.write("[POLICY] Retrospective oracle baseline only; original app v0.4 unchanged")


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_BASE.mkdir(parents=True, exist_ok=True)
    out = REPORT_BASE / f"v06_gate_baseline_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    log = Log(out / "console.log")
    try:
        run(out, log)
        log.write("[READY] v0.6 diagnostic complete; all existing code/models/data unchanged")
    except BaseException:
        log.write("[STOP] v0.6 aborted; original app, data and previous reports unchanged")
        log.write(traceback.format_exc())
        raise
    finally:
        log.close()
    paths = sorted(p for p in out.iterdir() if p.is_file() and
                   p.name not in {"sha256_manifest.json", "onsetaudit_v06_gate_audit.zip"})
    save_json(out / "sha256_manifest.json", {p.name: digest(p) for p in paths})
    zpath = out / "onsetaudit_v06_gate_audit.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != zpath:
                z.write(p, arcname=p.name)
    print("[ZIP]", zpath, flush=True)
    if sys.platform == "darwin":
        subprocess.run(["open", str(out / "onsetaudit_v06_dashboard.html")], check=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[STOP]", type(exc).__name__ + ":", exc, file=sys.stderr, flush=True)
        sys.exit(1)
