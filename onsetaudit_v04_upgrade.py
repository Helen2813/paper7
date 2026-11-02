#!/usr/bin/env python3
"""OnsetAudit v0.4.0 safe in-place upgrade + frozen-gate negative-control audit.

Usage (no flags): python onsetaudit_v04_upgrade.py
Run from ~/Desktop/paper7. No network, no Git branch creation or push.
Backs up and patches the current v0.3.0 app only after exact anchor checks.
Retains the V120-family model, gene selection, normalization, and 0.0 gate.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone


AUDIT_MODULE = r'''"""OnsetAudit v0.4 descriptive negative-control diagnostics.

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
'''


UI_FUNCTION = r'''function renderV04Audit(root,r){
  const a=r.v04_audit;if(!a)return;
  const wrap=text('div','','panel');wrap.style.marginTop='18px';
  const heading=text('h3','v0.4 · Negative-control audit');wrap.append(heading);
  const status=text('p','DESCRIPTIVE ONLY · Frozen v0.3 gate margin > 0 · Not calibrated.','hint');status.style.margin='7px 0 14px';wrap.append(status);
  const stats=text('div','','modelstats');
  for(const [name,value] of [['Vehicle false-positive samples',a.vehicle_false_positive_samples_any_drug+'/'+a.n_vehicle_samples],['False-positive drug calls',a.vehicle_false_positive_drug_calls+'/'+a.n_vehicle_drug_comparisons]]){
    const card=text('div','','modelstat');card.append(text('strong',value),text('small',name));stats.append(card);
  }
  wrap.append(stats);
  const makeTable=(headers,rows)=>{const sc=text('div','','table-scroll');sc.style.marginTop='12px';const t=document.createElement('table');const head=document.createElement('tr');headers.forEach(x=>head.append(text('th',x)));t.append(head);for(const arr of rows){const tr=document.createElement('tr');arr.forEach(x=>tr.append(text('td',x)));t.append(tr);}sc.append(t);return sc;};
  const byDrug=a.by_drug||[];
  wrap.append(text('h4','DMSO rejection by drug · held-out experiments'));
  wrap.append(makeTable(['Drug','Controls','False calls','Control FPR','Active correct'],byDrug.map(x=>[x.drug,x.control_n,x.false_active_calls,x.control_n?(100*x.false_active_calls/x.control_n).toFixed(1)+'%':'—',x.duration_correct+'/'+x.active_n])));
  const fp=(r.vehicle_drug_predictions||[]).filter(x=>x.accepted_as_active);
  wrap.append(text('h4','Which control samples triggered a drug program?'));
  if(fp.length){wrap.append(makeTable(['DMSO sample','Held-out group','Time h','Drug program','Gate margin'],fp.map(x=>[x.sample,x.group,x.time,x.drug,Number(x.gate_margin).toFixed(4)])));}
  else{wrap.append(text('p','No positive DMSO-by-drug calls in this evaluation.','hint'));}
  wrap.append(text('h4','Threshold sensitivity · retrospective, not calibrated'));
  wrap.append(makeTable(['Margin threshold','Active accepted','Correct + accepted','DMSO rejected'],(a.threshold_sweep||[]).map(x=>[(x.is_frozen_v03_gate?'FROZEN ':'')+x.threshold,x.active_accepted+'/'+x.active_n,x.correct_and_accepted+'/'+x.active_n,x.vehicle_rejected_any_drug+'/'+x.vehicle_n])));
  const caution=text('div','Do not choose a threshold from this table: the same held-out samples are used here for descriptive sensitivity. This is NOT independent calibration, an onset-time measurement, or evidence of 95% specificity.','notice');wrap.append(caution);
  root.append(wrap);
}
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_once(source: str, old: str, description: str) -> None:
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"Patch anchor {description!r}: expected once, found {n}. No files changed.")


def replace_once(source: str, old: str, new: str, description: str) -> str:
    assert_once(source, old, description)
    return source.replace(old, new, 1)


def make_core_patch(content: str) -> str:
    content = replace_once(content, 'VERSION = "0.3.0"', 'VERSION = "0.4.0"', 'v0.3.0 version')
    content = replace_once(
        content,
        'import pandas as pd\n',
        'import pandas as pd\nfrom v04_audit import build_v04_audit\n',
        'audit module import',
    )
    content = replace_once(
        content,
        '        negative_source_calls: dict[int, list[bool]] = {}\n',
        '        negative_source_calls: dict[int, list[dict[str, Any]]] = {}\n',
        'negative call structure',
    )
    content = replace_once(
        content,
        '''                for i in test_controls:
                    negative_source_calls.setdefault(i, []).append(scores(X[selected_genes, i])[0])
        negative = []
        for i, flags in negative_source_calls.items():
            row = meta.iloc[i]
            negative.append({"sample": row['sample'], "group": row['group'],
                             "time": float(row['time']), "false_active_any_drug": any(flags),
                             "source_programs_checked": len(flags)})
''',
        '''                for i in test_controls:
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
''',
        'full per-drug heldout control margins',
    )
    content = replace_once(
        content,
        '''            "vehicle_predictions": negative,
            "method": {"top_genes": TOP_GENES,''',
        '''            "vehicle_predictions": negative,
            "vehicle_drug_predictions": vehicle_drug_predictions,
            "v04_audit": build_v04_audit(
                predictions, negative, vehicle_drug_predictions,
                selected, groups, times,
            ),
            "method": {"top_genes": TOP_GENES,''',
        'v0.4 diagnostics in evaluation',
    )
    compile(content, 'onsetaudit/core.py', 'exec')
    return content


def make_app_patch(content: str) -> str:
    content = replace_once(
        content,
        "('folds','holdout_folds.csv')):",
        "('folds','holdout_folds.csv'),\n"
        "                                            ('vehicle_drug_predictions','vehicle_by_drug.csv')):",
        'v0.4 vehicle by drug CSV export',
    )
    content = replace_once(
        content,
        "                                    z.writestr(fn, pd.DataFrame(table).to_csv(index=False))\n"
        "                        manifest = {'version': VERSION, 'generated_utc': stamp,",
        "                                    z.writestr(fn, pd.DataFrame(table).to_csv(index=False))\n"
        "                            diagnostics = evaluation.get('v04_audit', {})\n"
        "                            for key, fn in ((\"by_drug\", \"v04_by_drug.csv\"),\n"
        "                                            (\"by_group\", \"v04_by_group.csv\"),\n"
        "                                            (\"by_time\", \"v04_by_time.csv\"),\n"
        "                                            (\"threshold_sweep\", \"v04_threshold_sweep_posthoc.csv\")):\n"
        "                                rows = diagnostics.get(key, [])\n"
        "                                if rows:\n"
        "                                    z.writestr(fn, pd.DataFrame(rows).to_csv(index=False))\n"
        "                        manifest = {'version': VERSION, 'generated_utc': stamp,",
        'v0.4 report diagnostic CSV exports',
    )
    compile(content, 'onsetaudit/app.py', 'exec')
    return content


def make_html_patch(content: str) -> str:
    content = replace_once(
        content,
        'Research prototype · v0.3',
        'Research prototype · v0.4',
        'sidebar version display',
    )
    content = replace_once(
        content,
        'OnsetAudit 0.3 · Paper 7',
        'OnsetAudit 0.4 · Paper 7',
        'footer version display',
    )
    content = replace_once(
        content,
        'function viewResult(r){',
        UI_FUNCTION + '\nfunction viewResult(r){',
        'v0.4 interface',
    )
    content = replace_once(
        content,
        "preview.append(t);root.append(preview);$('viewPredictions').disabled=false;}",
        "preview.append(t);root.append(preview);renderV04Audit(root,r);$('viewPredictions').disabled=false;}",
        'v0.4 interface rendering call',
    )
    return content


def validate_git_repo(root: Path) -> None:
    if not (root / '.git').exists():
        raise RuntimeError('Expected ~/Desktop/paper7 Git repository; no files changed')
    p = subprocess.run(['git', 'branch', '--show-current'], cwd=root,
                       text=True, capture_output=True, check=True)
    branch = p.stdout.strip()
    if branch != 'main':
        raise RuntimeError(f"Current Git branch is {branch!r}; requested main. No branch switched.")
    touched = ['onsetaudit/core.py', 'onsetaudit/app.py', 'onsetaudit/web/index.html']
    p = subprocess.run(['git', 'status', '--porcelain', '--', *touched], cwd=root,
                       text=True, capture_output=True, check=True)
    if p.stdout.strip():
        raise RuntimeError('Uncommitted edits exist in application files; refusing to overwrite:\n' + p.stdout)
    print('[GIT] main; application files have no uncommitted changes', flush=True)


def backup_manifest(root: Path, source: dict[Path, str], stamp: str) -> Path:
    backup = root / 'onsetaudit' / '_generated' / 'backups' / f'pre_v04_{stamp}'
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for file, content in source.items():
        target = backup / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        manifest[str(file.relative_to(root))] = sha(file)
    (backup / 'original_sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return backup


def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(path.parent),
                                     prefix='.onsetaudit_v04_', delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def install(root: Path) -> tuple[bool, Path | None]:
    files = [root / x for x in ('onsetaudit/core.py', 'onsetaudit/app.py', 'onsetaudit/web/index.html')]
    for file in files:
        if not file.is_file():
            raise RuntimeError(f'Missing application file: {file}')
    version = files[0].read_text('utf-8')
    marker = root / 'onsetaudit/v04_audit.py'
    if 'VERSION = "0.4.0"' in version and marker.exists() and 'build_v04_audit' in version:
        print('[UPGRADE] v0.4 already installed. No files changed.', flush=True)
        return False, None
    validate_git_repo(root)
    old = {p: p.read_text('utf-8') for p in files}
    new = {
        files[0]: make_core_patch(old[files[0]]),
        files[1]: make_app_patch(old[files[1]]),
        files[2]: make_html_patch(old[files[2]]),
        marker: AUDIT_MODULE,
    }
    compile(AUDIT_MODULE, 'onsetaudit/v04_audit.py', 'exec')
    # All anchors and Python syntax checked BEFORE touching working application.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = backup_manifest(root, old, stamp)
    try:
        for path, content in new.items():
            write_atomic(path, content)
        print('[UPGRADE] OnsetAudit v0.4 files written', flush=True)
        print('[BACKUP]', backup, flush=True)
    except BaseException:
        for path, original in old.items():
            write_atomic(path, original)
        marker.unlink(missing_ok=True)
        raise
    return True, backup


def verify_core(root: Path) -> None:
    sys.path.insert(0, str(root / 'onsetaudit'))
    importlib.invalidate_caches()
    from core import Dataset, VERSION
    from v04_audit import build_v04_audit
    assert VERSION == '0.4.0'
    result = Dataset()
    result.demonstration()
    ev = result.evaluate()
    audit = ev['v04_audit']
    assert audit['frozen_baseline_reproduced']
    assert audit['vehicle_false_positive_samples_any_drug'] == ev['n_vehicle'] - ev['vehicle_rejected']
    assert ev['correct_and_accepted'] == next(x for x in audit['threshold_sweep']
                                              if x['threshold'] == 0.0)['correct_and_accepted']
    assert len(ev['vehicle_drug_predictions']) == ev['n_vehicle'] * len(ev['drugs_evaluated'])
    print(f"[TEST] Synthetic unit smoke: OK, n_active={ev['n_active']}, n_vehicle={ev['n_vehicle']}", flush=True)


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b''
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def real_data_audit(root: Path) -> None:
    source = root / 'onsetaudit/_generated/import_ready/GSE194200'
    meta_path, counts_path = source / 'metadata.csv', source / 'expression_counts.csv.gz'
    if not meta_path.is_file() or not counts_path.is_file():
        print('[REAL DATA] GSE194200 not available on this Mac; skipped (no synthetic substitution)', flush=True)
        return
    from core import Dataset
    ds = Dataset()
    ds.import_metadata(meta_path.read_bytes(), meta_path.name)
    ds.import_matrix(counts_path.read_bytes(), counts_path.name, 'counts')
    summary = ds.summary()
    print(f"[REAL DATA] {summary['genes']} genes, {summary['samples']} samples", flush=True)
    ev = ds.evaluate()
    audit = ev['v04_audit']
    print(f"[FROZEN v0.3] Duration correct {ev['correct_duration_without_gate']}/{ev['n_active']}; "
          f"DMSO rejected {ev['vehicle_rejected']}/{ev['n_vehicle']}", flush=True)
    print('[CONTROL AUDIT] Drug-specific false calls:', flush=True)
    for drug in audit['by_drug']:
        print(f"  {drug['drug']}: {drug['false_active_calls']}/{drug['control_n']}; "
              f"IDs: {drug['false_positive_samples'] or 'none'}", flush=True)
    if summary['samples'] == 36 and summary['genes'] == 24421:
        expected = {'n_active':24, 'correct_and_accepted':22, 'n_vehicle':12, 'vehicle_rejected':10}
        for key, value in expected.items():
            if ev.get(key) != value:
                raise RuntimeError(f'Frozen v0.3 GSE194200 benchmark drift: {key}: '
                                   f'{ev.get(key)} != {value}. STOP; inspect before commit.')
        print('[REGRESSION] v0.3 benchmark 22/24 and 10/12 reproduced exactly', flush=True)
    else:
        print('[REGRESSION] Dataset differs from the previous 36 × 24421 GSE194200; comparison not enforced', flush=True)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = root / 'onsetaudit/_generated/reports' / f'v04_negative_controls_{stamp}'
    target.mkdir(parents=True, exist_ok=False)
    report = ds.report()
    report['runtime_provenance'] = {
        'python': sys.version.split()[0],
        'core_source_sha256': sha(root / 'onsetaudit/core.py'),
        'app_source_sha256': sha(root / 'onsetaudit/app.py'),
        'v04_audit_source_sha256': sha(root / 'onsetaudit/v04_audit.py'),
        'upgrade_script_sha256': sha(root / 'onsetaudit_v04_upgrade.py'),
    }
    records = {
        'report.json': json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'),
        'active_predictions.csv': csv_bytes(ev['predictions']),
        'vehicle_predictions.csv': csv_bytes(ev['vehicle_predictions']),
        'vehicle_by_drug.csv': csv_bytes(ev['vehicle_drug_predictions']),
        'holdout_folds.csv': csv_bytes(ev['folds']),
        'v04_by_drug.csv': csv_bytes(audit['by_drug']),
        'v04_by_group.csv': csv_bytes(audit['by_group']),
        'v04_by_time.csv': csv_bytes(audit['by_time']),
        'v04_threshold_sweep_posthoc.csv': csv_bytes(audit['threshold_sweep']),
        'README.txt': (
            'OnsetAudit v0.4 negative-control diagnostic. No raw counts are included.\n'
            'Gate threshold zero and original model frozen; other thresholds are POST-HOC descriptions.\n'
            'Do not claim biological onset, independent calibration, or 95% specificity.\n'
        ).encode('utf-8'),
    }
    records['sha256_manifest.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
                                                    for name, data in records.items()}, indent=2).encode('utf-8')
    for name, data in records.items():
        (target / name).write_bytes(data)
    bundle = target / 'onsetaudit_v04_negative_control_audit.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in records.items():
            archive.writestr(name, data)
    print('[OUTPUT] CSV + report:', target, flush=True)
    print('[ZIP]', bundle, flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    if root != Path.home() / 'Desktop/paper7':
        print('[NOTE] Running from', root, flush=True)
    print('=' * 72, flush=True)
    print('PAPER 7 | ONSETAUDIT v0.4 | FROZEN-GATE NEGATIVE CONTROL AUDIT', flush=True)
    print('OFFLINE | NO FLAGS | MAIN ONLY | NO GENE/THRESHOLD RETUNING', flush=True)
    print('=' * 72, flush=True)
    created, backup = install(root)
    try:
        verify_core(root)
        real_data_audit(root)
    except BaseException:
        if created and backup is not None:
            print('[ROLLBACK] Restoring v0.3 application from local backup', flush=True)
            for relative in ('onsetaudit/core.py', 'onsetaudit/app.py', 'onsetaudit/web/index.html'):
                old = backup / relative
                if old.is_file():
                    write_atomic(root / relative, old.read_text(encoding='utf-8'))
            (root / 'onsetaudit/v04_audit.py').unlink(missing_ok=True)
        raise
    print('[READY] v0.4 update and validation complete; new server can be started.', flush=True)
    print('[GIT] Changes are on local main; NO automatic commit/push.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('[STOP]', type(error).__name__ + ':', error, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
