#!/usr/bin/env python3
"""Paper 7 / OnsetAudit v0.5: frozen-prediction DMSO quality/confounding audit.

Standalone, no flags. Reads the verified GSE194200 imports and most recent
v0.4 negative-control report. Never retrains/retunes the onset-duration model,
changes app/core.py, or alters the frozen acceptance threshold (0.0).

Run: python onsetaudit_v05_qc_audit.py from ~/Desktop/paper7
Outputs reports, manifest, full console log, and ZIP under onsetaudit/_generated.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "onsetaudit/_generated/import_ready/GSE194200"
REPORT_BASE = ROOT / "onsetaudit/_generated/reports"
VERSION = "v0.5.0"
EXPECTED = {"n_active": 24, "n_vehicle": 12, "correct_and_accepted": 22,
            "vehicle_rejected": 10}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def finite_float(v, name):
    val = float(v)
    require(math.isfinite(val), f"Nonfinite {name}")
    return val


def as_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise RuntimeError(f"Unexpected boolean value: {value!r}")


class Log:
    def __init__(self, path):
        self.path = path
        self.fd = path.open("w", encoding="utf-8")

    def write(self, *values):
        msg = " ".join(str(v) for v in values)
        print(msg, flush=True)
        self.fd.write(msg + "\n")
        self.fd.flush()

    def close(self):
        self.fd.close()


def find_previous_report() -> Path:
    candidates = sorted(REPORT_BASE.glob("v04_negative_controls_*/report.json"),
                        key=lambda p: p.parent.name, reverse=True)
    require(bool(candidates), "No v0.4 local report. Run onsetaudit_v04_upgrade.py first; no new model run performed.")
    return candidates[0]


def verify_previous_report(path: Path, obj: dict) -> None:
    require(obj.get("version") == "0.4.0", "Only frozen OnsetAudit v0.4.0 reports are accepted")
    m = path.parent / "sha256_manifest.json"
    require(m.is_file(), "v0.4 report is missing SHA-256 manifest")
    expected = json.loads(m.read_text(encoding="utf-8"))
    require(expected.get("report.json") == digest(path), "Prior report SHA-256 mismatch")
    for name, original_hash in expected.items():
        source = path.parent / name
        require(source.is_file() and digest(source) == original_hash,
                f"Previous v0.4 file missing/altered: {name}")


def check_git():
    require((ROOT / ".git").is_dir(), "Not a Git checkout")
    status = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    require(status.stdout.strip() == "main", "Only use the existing main branch")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    return commit.stdout.strip()


def check_inputs(report):
    meta_file, expr_file = DATA / "metadata.csv", DATA / "expression_counts.csv.gz"
    for f in (meta_file, expr_file):
        require(f.is_file(), f"Missing verified data file: {f}")
    source = report.get("provenance", {})
    for f, key in ((meta_file, "metadata"), (expr_file, "expression")):
        expect_sha = source.get(key, {}).get("sha256")
        require(expect_sha and digest(f) == expect_sha,
                f"GSE194200 {key} SHA-256 does not match frozen v0.4 report")
    meta = pd.read_csv(meta_file, dtype={"sample_column_name": str})
    needed = {"sample_column_name", "drug", "timepoint", "library", "replicate"}
    require(needed.issubset(meta.columns), f"Metadata missing {needed - set(meta)}")
    require(len(meta) == 36 and meta.sample_column_name.is_unique,
            "Expected 36 unique sample IDs")
    require(set(meta.drug) == {"DMSO", "SW016789", "Thapsigargin"}, "Unexpected treatment schema")
    require(set(float(x) for x in meta.timepoint) == {1., 2., 6., 24.}, "Unexpected timing schema")
    require(set(meta.library) == {f"experiment_{i}" for i in (1, 2, 3)}, "Unexpected group schema")
    expected_cells = {(g, d, t) for g in meta.library.unique()
                      for d in meta.drug.unique() for t in (1., 2., 6., 24.)}
    got = {(r.library, r.drug, float(r.timepoint)) for r in meta.itertuples()}
    require(got == expected_cells and len(got) == 36, "Missing/duplicate time×group×drug cell")
    matrix = pd.read_csv(expr_file, compression="gzip")
    require(matrix.shape == (24421, 37), "Unexpected gene × sample matrix shape")
    require(matrix.columns[0] == "gene_id", "Expected first expression column gene_id")
    require(matrix.columns[1:].tolist() == meta.sample_column_name.tolist(),
            "Sample column order differs from metadata")
    c = matrix.iloc[:, 1:].to_numpy(dtype=np.float64)
    require(np.isfinite(c).all() and (c >= 0).all(), "Counts are missing, negative, or nonfinite")
    require(np.all(c == np.floor(c)), "Noninteger counts")
    depth = c.sum(axis=0)
    require((depth > 0).all(), "Zero-depth sample")
    logcpm = np.log2(1.0 + (c / depth[None, :]) * 1e6)
    qc = pd.DataFrame({
        "sample": meta.sample_column_name.to_numpy(),
        "group": meta.library.to_numpy(),
        "drug": meta.drug.to_numpy(),
        "true_time_h": meta.timepoint.astype(float).to_numpy(),
        "library_size_counts": depth.astype(np.int64),
        "log10_library_size": np.log10(depth),
        "genes_with_positive_counts": (c > 0).sum(axis=0),
        "genes_with_at_least_10_counts": (c >= 10).sum(axis=0),
        "top_10_genes_count_fraction": np.sort(c, axis=0)[-10:, :].sum(axis=0) / depth,
    })
    return qc, logcpm, meta, meta_file, expr_file


def vehicle_rows(report, qc, logcpm, meta):
    ev = report["evaluation"]
    for k, target in EXPECTED.items():
        require(ev.get(k) == target, f"Frozen v0.3/v0.4 result changed: {k}={ev.get(k)} (expected {target})")
    require(ev.get("active_accepted") == 24 and ev.get("correct_duration_without_gate") == 22,
            "Frozen acceptance or time classifier changed")
    prior_audit = ev.get("v04_audit", {})
    require(prior_audit.get("frozen_baseline_reproduced") is True, "v0.4 baseline not verified")
    require(prior_audit.get("frozen_gate_margin_threshold") == 0., "Gate was not frozen at zero")
    negatives = ev.get("vehicle_predictions", [])
    controls = ev.get("vehicle_drug_predictions", [])
    predictions = ev.get("predictions", [])
    require(len(negatives) == 12 and len(controls) == 24 and len(predictions) == 24,
            "Unexpected number of active or control observations")
    require(len({p["sample"] for p in predictions}) == 24 and
            len({n["sample"] for n in negatives}) == 12, "Duplicate prediction IDs")
    require(set(p["sample"] for p in predictions).isdisjoint({n["sample"] for n in negatives}),
            "Active and DMSO IDs overlap")
    controls_map = {}
    for record in controls:
        sid, drug = str(record["sample"]), str(record["drug"])
        require(drug in ("SW016789", "Thapsigargin"), "Unrecognized drug program")
        key = sid, drug
        require(key not in controls_map, f"Duplicate drug margin for {key}")
        margin = finite_float(record["gate_margin"], f"gate_margin {key}")
        require(as_bool(record["accepted_as_active"]) == (margin > 0),
                f"Gate/margin mismatch for {key}")
        controls_map[key] = margin
    require(len(controls_map) == 24, "Missing DMSO × drug comparisons")
    meta_ix = meta.set_index("sample_column_name")
    controls_ix = {r["sample"]: r for r in negatives}
    features = []
    DMSO_ixs = np.flatnonzero(meta.drug.to_numpy() == "DMSO")
    # All control samples are diagnostic material only. The selection below is
    # a descriptive, post-hoc QC filter; it is NEVER used for model training.
    selected = ((logcpm[:, DMSO_ixs] > 0).sum(axis=1) >= 3)
    # Unlike feature selection for classification, this filter is not used for
    # any prediction, gate, calibration, or performance claim.
    require(int(selected.sum()) >= 200, "Too few observed genes for profile QC")
    for neg in negatives:
        sid = str(neg["sample"])
        require(sid in meta_ix.index, f"DMSO sample not in metadata: {sid}")
        mi = meta_ix.loc[sid]
        require(mi["drug"] == "DMSO", f"Control in wrong treatment: {sid}")
        require(str(mi["library"]) == str(neg["group"]) and
                float(mi["timepoint"]) == float(neg["time"]),
                f"Control time or group mismatch: {sid}")
        a = controls_map[(sid, "SW016789")]
        b = controls_map[(sid, "Thapsigargin")]
        fp = int(a > 0) + int(b > 0)
        require((fp > 0) == as_bool(neg["false_active_any_drug"]), "Frozen control mismatch")
        sidx = meta_ix.index.get_loc(sid)
        peers = [int(i) for i in DMSO_ixs if i != sidx and
                 float(meta.iloc[i]["timepoint"]) == float(mi["timepoint"]) and
                 str(meta.iloc[i]["library"]) != str(mi["library"])]
        require(len(peers) == 2, f"Expected 2 same-time controls from other groups: {sid}")
        y = logcpm[selected, sidx]
        reference = np.mean(logcpm[selected][:, peers], axis=1)
        diffs = y - reference
        profile_rmse = float(np.sqrt(np.mean(diffs ** 2)))
        med_abs = float(np.median(np.abs(diffs)))
        corr = float(np.corrcoef(y, reference)[0, 1])
        require(math.isfinite(corr), "Nonfinite expression correlation")
        base = qc.loc[qc["sample"] == sid].iloc[0].to_dict()
        base.update({
            "false_active_any_drug": fp > 0,
            "number_positive_drug_programs": fp,
            "gate_margin_SW016789": a,
            "gate_margin_Thapsigargin": b,
            "max_gate_margin": max(a, b),
            "time_matched_cross_group_logcpm_RMSE": profile_rmse,
            "time_matched_cross_group_logcpm_median_abs": med_abs,
            "time_matched_cross_group_logcpm_pearson": corr,
            "diagnostic_genes_in_expression_distance": int(selected.sum()),
        })
        features.append(base)
    diag = pd.DataFrame(features).sort_values(["group", "true_time_h"]).reset_index(drop=True)
    require(diag.false_active_any_drug.sum() == 2, "Wrong number of sample-level false calls")
    require((diag.number_positive_drug_programs == 2).sum() == 2 and
            (diag.number_positive_drug_programs == 1).sum() == 0,
            "Unexpected drug-program overlap; inspect before writing conclusions")
    return diag


def compare_fp_to_tn(df):
    metrics = ["library_size_counts", "genes_with_positive_counts",
               "genes_with_at_least_10_counts", "top_10_genes_count_fraction",
               "time_matched_cross_group_logcpm_RMSE",
               "time_matched_cross_group_logcpm_median_abs",
               "time_matched_cross_group_logcpm_pearson"]
    positives = df.loc[df.false_active_any_drug]
    negatives = df.loc[~df.false_active_any_drug]
    result = []
    for metric in metrics:
        fp_m = float(positives[metric].median())
        tn_m = float(negatives[metric].median())
        ranks = df[metric].rank(pct=True, method="average")
        fp_ranks = ranks.loc[positives.index].to_list()
        result.append({
            "metric": metric,
            "false_positive_n": len(positives),
            "true_negative_n": len(negatives),
            "false_positive_median": fp_m,
            "true_negative_median": tn_m,
            "descriptive_difference": fp_m - tn_m,
            "fp_rank_percentiles_all12": ";".join(f"{100*x:.1f}" for x in fp_ranks),
            "interpretation": "POST_HOC_DESCRIPTIVE_ONLY_NO_P_VALUES_OR_THRESHOLD_CHANGES",
        })
    return pd.DataFrame(result)


def group_time_tables(df):
    by_group = df.groupby("group", sort=True).agg(
        DMSO_samples=("sample", "size"),
        false_positive_samples=("false_active_any_drug", "sum"),
        both_drug_false_calls=("number_positive_drug_programs", lambda v: int((v == 2).sum())),
        median_library_size=("library_size_counts", "median"),
        median_time_matched_logcpm_RMSE=("time_matched_cross_group_logcpm_RMSE", "median"),
    ).reset_index()
    by_time = df.groupby("true_time_h", sort=True).agg(
        DMSO_samples=("sample", "size"),
        false_positive_samples=("false_active_any_drug", "sum"),
        both_drug_false_calls=("number_positive_drug_programs", lambda v: int((v == 2).sum())),
        median_library_size=("library_size_counts", "median"),
        median_time_matched_logcpm_RMSE=("time_matched_cross_group_logcpm_RMSE", "median"),
    ).reset_index()
    return by_group, by_time


def margin_correlation(df):
    x = df.gate_margin_SW016789.to_numpy(dtype=float)
    y = df.gate_margin_Thapsigargin.to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return {"n_shared_DMSO_samples": 12,
            "margin_pearson_correlation": pearson,
            "margin_spearman_correlation": spearman,
            "correlated_tests_warning": "The 24 drug comparisons reuse the same 12 DMSO samples; they are not independent"}


def add_section_table(lines, title, df, showcols=None):
    lines.append(f"<h2>{html.escape(title)}</h2>")
    if showcols:
        df = df.loc[:, showcols]
    table = df.copy()
    for col in table:
        if pd.api.types.is_bool_dtype(table[col]):
            table[col] = table[col].map({True: "Yes", False: "No"})
        elif pd.api.types.is_numeric_dtype(table[col]):
            if not pd.api.types.is_integer_dtype(table[col]):
                table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    lines.append(table.to_html(index=False, border=0, classes="data", escape=True))


def svg_margin_plot(df):
    # Static readable SVG diagnostic; values are *already frozen* held-out margins.
    width, height = 940, 370
    l, r, t, b = 75, 30, 30, 85
    low = min(0., float(df[["gate_margin_SW016789", "gate_margin_Thapsigargin"]].min().min()))
    high = max(0., float(df[["gate_margin_SW016789", "gate_margin_Thapsigargin"]].max().max()))
    pad = max(0.1, (high - low) * 0.12)
    low -= pad
    high += pad
    fx = lambda j: l + (j + 0.5) * (width - l - r) / len(df)
    fy = lambda v: t + (high - v) * (height - t - b) / (high - low)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Frozen DMSO gate margins for both drug programs">']
    zero = fy(0)
    pieces.append(f'<line x1="{l}" y1="{zero:.1f}" x2="{width-r}" y2="{zero:.1f}" stroke="#9aa5b1" stroke-dasharray="5,5"/>')
    pieces.append(f'<text x="{l-14}" y="{zero+4:.1f}" text-anchor="end" font-size="12">0</text>')
    for j, row in df.iterrows():
        x = fx(j)
        shade = "#b91c1c" if row.false_active_any_drug else "#53657a"
        pieces.append(f'<text transform="translate({x:.1f},{height-b+13}) rotate(48)" font-size="10" fill="{shade}">{html.escape(str(row["sample"]))}</text>')
        for shift, field, color in ((-8, "gate_margin_SW016789", "#6156dd"),
                                    (8, "gate_margin_Thapsigargin", "#10a5a0")):
            y = fy(float(row[field]))
            pieces.append(f'<circle cx="{x+shift:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
    pieces.append('<text x="80" y="17" font-size="12" fill="#53657a">Positive margin = false active call (frozen gate threshold 0)</text>')
    pieces.append('</svg>')
    return "".join(pieces)


def dashboard(df, contrasts, by_group, by_time, correlation, source_report, sha_inputs):
    fp = df.loc[df.false_active_any_drug]
    lines = ["""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OnsetAudit v0.5 · DMSO diagnostic</title>
<style>body{margin:0;background:#f1f4f9;color:#12203a;font-family:system-ui,-apple-system,Arial,sans-serif}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px;margin-bottom:5px}h2{font-size:19px;margin:26px 0 9px}.subtitle,.note{color:#617087}.banner{padding:14px 17px;background:white;border-left:4px solid #6b5fe6;border-radius:7px;line-height:1.5}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0}.card,.panel{background:white;border:1px solid #dce3ef;border-radius:12px;padding:16px}.card b{font-size:27px;display:block}.card small{color:#617087}.panel{overflow-x:auto;margin-bottom:15px}.data{border-collapse:collapse;width:100%;font-size:12px}.data th{background:#eef1f8;text-align:left}.data td,.data th{padding:9px 10px;border-bottom:1px solid #e5eaf2;white-space:nowrap}.warning{background:#fff7ed;border:1px solid #fed7aa;padding:15px;border-radius:9px}.foot{font-size:12px;color:#66748a;line-height:1.6;overflow-wrap:anywhere}svg{width:100%;height:auto}code{overflow-wrap:anywhere}@media(max-width:780px){.stats{grid-template-columns:repeat(2,1fr)}main{padding:12px}}</style></head><body><main>
<h1>OnsetAudit v0.5 · DMSO quality / confounding diagnostics</h1><p class="subtitle">GSE194200 · frozen v0.4 predictions · retrospective descriptive audit · not independent validation</p>"""]
    lines.append('<div class="banner">The two false-positive DMSO samples each trigger <b>both</b> drug programs. The false calls are in <b>experiment_1, at 1h and 2h</b>. This is compatible with a shared sample/time/batch effect; the audit alone does <b>not</b> identify its cause.</div>')
    lines.append(f'<div class="stats"><div class="card"><b>22/24</b><small>Duration + gate correct (unchanged)</small></div><div class="card"><b>10/12</b><small>DMSO samples rejected (unchanged)</small></div><div class="card"><b>2/12</b><small>DMSO samples falsely accepted</small></div><div class="card"><b>4/24</b><small>Drug-program false calls, dependent</small></div></div>')
    lines.append('<div class="panel"><h2>Gate margins · fixed at 0</h2>' + svg_margin_plot(df) + '<div style="font-size:13px;color:#60708a">Purple = SW016789 program · teal = Thapsigargin program · dashed line = zero threshold. Same 12 DMSO samples scored twice.</div></div>')
    lines.append('<div class="panel">')
    add_section_table(lines, "False-positive controls", fp,
                      ["sample", "group", "true_time_h", "gate_margin_SW016789",
                       "gate_margin_Thapsigargin", "library_size_counts",
                       "genes_with_positive_counts", "time_matched_cross_group_logcpm_RMSE"])
    lines.append('</div><div class="panel">')
    add_section_table(lines, "All 12 DMSO controls", df,
                      ["sample", "group", "true_time_h", "false_active_any_drug",
                       "gate_margin_SW016789", "gate_margin_Thapsigargin",
                       "library_size_counts", "genes_with_positive_counts",
                       "top_10_genes_count_fraction", "time_matched_cross_group_logcpm_RMSE",
                       "time_matched_cross_group_logcpm_pearson"])
    lines.append('</div><div class="panel">')
    add_section_table(lines, "Confusion concentration by experiment", by_group)
    lines.append('</div><div class="panel">')
    add_section_table(lines, "Confusion concentration by time", by_time)
    lines.append('</div><div class="panel">')
    add_section_table(lines, "False-positive versus other DMSO samples · descriptive", contrasts)
    lines.append('</div>')
    lines.append(f'<p class="note">Across 12 shared DMSO samples, the two program margins have descriptive Pearson correlation {correlation["margin_pearson_correlation"]:.3f} and Spearman correlation {correlation["margin_spearman_correlation"]:.3f}. These are not independent measurements, and n=12 is small.</p>')
    lines.append('<div class="warning"><b>Interpretation limits.</b> Cross-group same-time profile distances use only two other DMSO samples and all controls were inspected after seeing errors. QC contrasts and margin correlations must not be used to tune feature selection, gate threshold or calibration. Three experimental groups do not support a stable population specificity estimate. The target is elapsed exposure time, not directly observed molecular onset.</div>')
    lines.append(f'<p class="foot">Frozen source: {html.escape(str(source_report))}<br>Metadata SHA-256: <code>{sha_inputs["metadata"]}</code><br>Counts SHA-256: <code>{sha_inputs["expression"]}</code></p>')
    lines.append('</main></body></html>')
    return "\n".join(lines)


def make_manifest(path):
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name not in ("sha256_manifest.json",) and not p.name.endswith(".zip"))
    manifest = {p.name: digest(p) for p in files}
    (path / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(output, log):
    log.write("=" * 78)
    log.write("PAPER 7 | ONSETAUDIT v0.5 | DMSO QUALITY + SHARED ERROR AUDIT")
    log.write("OFFLINE | NO FLAGS | MAIN ONLY | FROZEN MODEL AND GATE | DESCRIPTIVE")
    log.write("=" * 78)
    git = check_git()
    source = find_previous_report()
    report = json.loads(source.read_text(encoding="utf-8"))
    verify_previous_report(source, report)
    log.write("[GIT] main", git)
    log.write("[INPUT REPORT]", source)
    log.write("[VERIFY] v0.4 report and file manifest hashes: OK")
    qc, logcpm, meta, meta_path, counts_path = check_inputs(report)
    log.write("[VERIFY] Raw metadata + counts match frozen report SHA-256")
    log.write("[REAL DATA]", len(qc), "samples; 24,421 genes; 12 shared DMSO controls")
    diag = vehicle_rows(report, qc, logcpm, meta)
    contrasts = compare_fp_to_tn(diag)
    by_group, by_time = group_time_tables(diag)
    correlation = margin_correlation(diag)
    fp = diag.loc[diag.false_active_any_drug]
    ids = fp["sample"].tolist()
    log.write("[FROZEN MODEL] 22/24 correct+accepted; 10/12 DMSO rejected")
    log.write("[FALSE DMSO] 2 distinct samples, 4 dependent drug-program calls")
    for row in fp.itertuples():
        log.write(f"[FP] {row.sample} | {row.group} | {row.true_time_h:g}h | "
                  f"SW margin={row.gate_margin_SW016789:.4f} | "
                  f"Thapsigargin margin={row.gate_margin_Thapsigargin:.4f} | "
                  f"depth={row.library_size_counts:,} | "
                  f"cross-group same-time RMSE={row.time_matched_cross_group_logcpm_RMSE:.4f}")
    log.write(f"[MARGIN CORRELATION] Pearson={correlation['margin_pearson_correlation']:.4f} "
              f"Spearman={correlation['margin_spearman_correlation']:.4f} (n=12, shared controls)")
    for row in by_group.itertuples():
        log.write(f"[GROUP] {row.group}: {row.false_positive_samples}/{row.DMSO_samples} false DMSO calls")
    log.write("[POLICY] QC is post-hoc; no threshold/gene tuning, no independent validation claim")

    files = {
        "vehicle_sample_QC_and_gate_margins.csv": diag.to_csv(index=False),
        "sample_QC_all_36.csv": qc.to_csv(index=False),
        "false_positive_vs_other_DMSO_posthoc.csv": contrasts.to_csv(index=False),
        "by_group.csv": by_group.to_csv(index=False),
        "by_time.csv": by_time.to_csv(index=False),
    }
    for name, payload in files.items():
        (output / name).write_text(payload, encoding="utf-8")
    hashes = {"metadata": digest(meta_path), "expression": digest(counts_path)}
    html_doc = dashboard(diag, contrasts, by_group, by_time, correlation, source, hashes)
    (output / "onsetaudit_v05_dashboard.html").write_text(html_doc, encoding="utf-8")
    result = {
        "audit_version": VERSION,
        "status": "DESCRIPTIVE_NO_RETRAINING_NO_CALIBRATION",
        "source_dataset": "GSE194200",
        "git_main_commit_at_analysis": git,
        "source_v04_report": str(source),
        "source_v04_report_sha256": digest(source),
        "source_metadata_sha256": hashes["metadata"],
        "source_expression_sha256": hashes["expression"],
        "frozen_gate_threshold": 0.0,
        "frozen_correct_and_accepted": 22,
        "frozen_active_total": 24,
        "frozen_vehicle_rejected": 10,
        "frozen_vehicle_total": 12,
        "DMSO_false_samples": ids,
        "distinct_false_sample_count": 2,
        "drug_program_false_call_count_correlated": 4,
        "drug_margin_association": correlation,
        "qc_contrast_rows": contrasts.to_dict(orient="records"),
        "warnings": [
            "Post-hoc QC associations are descriptive and not validated explanatory mechanisms",
            "Both drug scores are calculated on the SAME control samples, so 24 comparisons are not 24 independent trials",
            "Only three experimental groups; replication and uncertainty are limited",
            "Other two same-time DMSO controls used as diagnostic reference, not an external standard",
            "No gate, feature selection, model parameter, or exposure-duration predictions were changed",
            "Elapsed exposure time is not independently measured biological onset",
        ],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "audit_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "OnsetAudit v0.5: frozen-prediction DMSO quality/confounding audit.\n"
        "No new model evaluation, thresholds, selected genes or duration predictions.\n"
        "Read the generated HTML dashboard and JSON/CSV, not just the headline counts.\n"
        "The research ZIP does NOT contain raw counts; see source hashes.\n"
        "CAUTION: post-hoc diagnostics do not validate onset, causality, batch\n"
        "explanations or specific target specificity.\n",
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__), output / "onsetaudit_v05_qc_audit.py")
    log.write("[OUTPUT]", output)
    log.write("[DASHBOARD]", output / "onsetaudit_v05_dashboard.html")


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = REPORT_BASE / f"v05_dmso_qc_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    log = Log(output / "console.log")
    try:
        run(output, log)
        log.write("[READY] v0.5 audit complete, original OnsetAudit v0.4 app unchanged")
    except BaseException:
        log.write("[STOP] v0.5 aborted with an error. Previous data/application unchanged.")
        log.write(traceback.format_exc())
        raise
    finally:
        log.close()
    make_manifest(output)
    zip_path = output / "onsetaudit_v05_qc_audit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file != zip_path:
                archive.write(file, arcname=file.name)
    print("[ZIP]", zip_path, flush=True)
    print("[OPEN DASHBOARD] open", output / "onsetaudit_v05_dashboard.html", flush=True)
    # Open a local diagnostic HTML report; do not require a second server.
    if sys.platform == "darwin":
        subprocess.run(["open", str(output / "onsetaudit_v05_dashboard.html")], check=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[STOP]", type(exc).__name__ + ":", exc, file=sys.stderr, flush=True)
        sys.exit(1)
