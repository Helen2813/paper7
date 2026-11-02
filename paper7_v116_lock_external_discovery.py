#!/usr/bin/env python3
"""Paper 7 V116: freeze V115 evidence; discover independent GSE149428.
No fitting or outcome-driven tuning on GSE149428.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import csv, gzip, json, math, re, shutil, sys, time, traceback, urllib.request, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'GSE149428'
OUT = ROOT / 'paper7_v116_external_lock' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149428/'
FILES = {
    'GSE149428_series_matrix.txt.gz':
        BASE + 'matrix/GSE149428_series_matrix.txt.gz',
    'GSE149428_Source_Data_File_1_log_cpm_viab.csv.gz':
        BASE + 'suppl/GSE149428_Source_Data_File_1_log_cpm_viab.csv.gz',
    'GSE149428_Source_Data_File_2_lncap_expression.csv.gz':
        BASE + 'suppl/GSE149428_Source_Data_File_2_lncap_expression.csv.gz'
}

DRUGS = ('T', 'M', 'W')
TIMES = (0, 3, 6, 9, 12, 24)
CELL_LINES = ('MCF7', 'LNCaP')

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def require(ok, s):
    if not bool(ok):
        raise RuntimeError(s)

def sha(p):
    h = sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def save(name, rows):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(
        OUT / name,
        index=False,
        encoding='utf-8-sig',
        float_format='%.15g'
    )
    say('[OUTPUT] {} | {} rows'.format(name, len(d)))
    return d

def freeze_spec():
    spec = {
        'version': 'v116',
        'status': 'FROZEN_BEFORE_EXTERNAL_EXPRESSION_AUDIT',
        'development_dataset': 'GSE194200',
        'external_study': 'GSE149428',
        'target': (
            'experimentally recorded active-compound exposure duration, '
            'not molecular response onset'
        ),
        'development_algorithm': 'V115 paired_effect_ridge_amplitude',
        'gene_count': 200,
        'variance_floor': 0.25,
        'ridge_penalty_per_gene': 0.25,
        'feature_rule': (
            'train-only across-time mean drug-vs-DMSO contrast variance '
            '/ (two-replicate between-contrast variance + 0.25)'
        ),
        'model_score': (
            'candidate active-minus-vehicle residual vs learned response '
            'template, ridge amplitude centered at 1, nonnegative amplitude'
        ),
        'required_query_context': (
            'time-labelled DMSO reference samples in the same cell line '
            'and replicate grouping'
        ),
        'negative_comparator': (
            'vehicle_reference_only with identical test-side controls'
        ),
        'exposure_hours_GSE149428': [3, 6, 9, 12, 24],
        'control_hours_GSE149428': [3, 6, 9, 12, 24],
        'external_subsets': {
            'cell_lines': ['MCF7', 'LNCaP'],
            'single_drug_codes': ['T', 'M', 'W'],
            'replicate_labels': ['A', 'B', 'C']
        },
        'external_evaluation_plan': (
            'within-cell-line leave-one-replicate-label-out; '
            'per-treatment and cross-treatment; no training from heldout '
            'active samples; compare paired vehicle-only baseline; '
            'report per-hour accuracy and MAE'
        ),
        'external_data_type': (
            'processed log CPM; normalization and logarithm base '
            'must be verified before applying the model'
        ),
        'unverified': (
            'A/B/C biological independence; cross-file sample mapping; '
            'normalization log base; drug identities; availability of '
            'all matched DMSO samples'
        ),
        'failure_policy': (
            'halt before prediction if sample mapping, measurement scale '
            'or required controls cannot be verified'
        ),
        'novelty_not_claimed': True,
        'external_results_unseen_at_freeze': True,
        'not_per_sample_deployable_without_time_labeled_vehicle_controls': True,
        'independent_validation_of_pretrained_mouse_weights': False,
        'algorithm_external_retrain_with_frozen_settings_only': True,
        'study_selection_post_GSE194200_development': True,
        'no_git': True,
        'script_sha256': sha(__file__)
    }

    (OUT / '01_frozen_external_protocol.json').write_text(
        json.dumps(spec, indent=2) + '\n',
        encoding='utf-8'
    )

    say(
        '[FROZEN] V115 architecture | 200 genes | '
        'variance floor 0.25 | ridge 0.25 | no external tuning'
    )

def verify_v115():
    base = ROOT / 'paper7_gse194200_paired_vehicle_v115'
    expected = (
        '13_sha256_inventory.csv',
        '12_manifest.json',
        '02_paired_vehicle_predictions.csv',
        '10_end_to_end_gate_metrics.csv',
        'script_snapshot_v115.py',
        'input_counts_logCPM.npz'
    )

    runs = sorted(
        (
            r for r in base.glob('run_*')
            if all((r / n).is_file() for n in expected)
        ),
        reverse=True
    )
    require(runs, 'Complete V115 archive not found')
    run = runs[0]

    inventory = pd.read_csv(
        run / '13_sha256_inventory.csv',
        encoding='utf-8-sig'
    )
    require(
        {'filename', 'bytes', 'sha256'}.issubset(inventory.columns)
        and inventory.filename.is_unique,
        'V115 inventory malformed'
    )

    for item in inventory.itertuples(index=False):
        name = str(item.filename)
        p = run / name
        require(
            Path(name).name == name
            and p.is_file()
            and p.stat().st_size == int(item.bytes)
            and sha(p) == str(item.sha256),
            'V115 modified artifact: ' + name
        )

    manifest = json.loads(
        (run / '12_manifest.json').read_text(encoding='utf-8')
    )
    require(
        manifest.get('version') == 'v115'
        and manifest.get('n_real_samples') == 39
        and manifest.get('n_outcomes') == 144,
        'V115 manifest mismatch'
    )

    original_code = ROOT / 'paper7_v115_control_anchored_pilot.py'
    require(
        original_code.is_file()
        and sha(original_code) == sha(run / 'script_snapshot_v115.py'),
        'V115 script differs from archived snapshot'
    )

    predictions = pd.read_csv(
        run / '02_paired_vehicle_predictions.csv',
        encoding='utf-8-sig',
        dtype={'gsm': str}
    )

    require(
        len(predictions) == 144
        and not predictions.duplicated(
            ['gsm', 'method', 'source_drug']
        ).any(),
        'V115 predictions duplicated'
    )

    predictions['correct'] = (
        predictions.correct.astype(str).str.lower().eq('true')
    )

    expected_correct = {
        ('paired_effect_ridge_amplitude', 'same_drug'): 21,
        ('paired_effect_ridge_amplitude', 'cross_drug'): 18,
        ('paired_effect_fixed', 'same_drug'): 19,
        ('paired_effect_fixed', 'cross_drug'): 12,
        ('vehicle_reference_only', 'same_drug'): 14,
        ('vehicle_reference_only', 'cross_drug'): 13
    }

    metrics = []

    for (method, transfer), ncorrect in expected_correct.items():
        subset = predictions[
            (predictions.method == method)
            & (predictions.transfer == transfer)
        ]

        require(
            len(subset) == 24
            and int(subset.correct.sum()) == ncorrect,
            'V115 result replay failed: ' + method + '/' + transfer
        )

        metrics.append(dict(
            method=method,
            transfer=transfer,
            n=24,
            correct=ncorrect,
            MAE_hours=float(subset.absolute_error_hours.mean())
        ))

    save('02_v115_frozen_metrics.csv', metrics)

    paired = []

    for transfer in ('same_drug', 'cross_drug'):
        a = predictions[
            (predictions.method == 'paired_effect_ridge_amplitude')
            & (predictions.transfer == transfer)
        ].set_index('gsm')

        b = predictions[
            (predictions.method == 'vehicle_reference_only')
            & (predictions.transfer == transfer)
        ].set_index('gsm')

        require(
            len(a) == len(b) == 24
            and set(a.index) == set(b.index),
            'The two methods did not evaluate the same samples'
        )

        wins = int((a.correct & ~b.correct).sum())
        losses = int((~a.correct & b.correct).sum())
        both = int((a.correct & b.correct).sum())
        neither = int((~a.correct & ~b.correct).sum())

        paired.append(dict(
            transfer=transfer,
            n=24,
            ridge_only_correct=wins,
            vehicle_only_correct=losses,
            both_correct=both,
            neither_correct=neither,
            net_gain=wins-losses
        ))

        say(
            '[PAIRED AUDIT | {}] ridge-only={} | vehicle-only={} '
            '| both={} | neither={} | net={:+d}/24'
            .format(transfer, wins, losses, both, neither, wins-losses)
        )

    save('03_v115_paired_discordances.csv', paired)

    gate = pd.read_csv(
        run / '10_end_to_end_gate_metrics.csv',
        encoding='utf-8-sig'
    )
    ridge = gate[
        gate.method == 'paired_effect_ridge_amplitude'
    ]

    require(
        len(ridge) == 1
        and int(ridge.iloc[0].DMSO_rejected) == 11
        and int(ridge.iloc[0].active_accepted) == 19
        and int(ridge.iloc[0].time_correct_and_accepted) == 16,
        'V115 DMSO gate mismatch'
    )

    say(
        '[DMSO REPLAY] DMSO rejected=11/12 '
        '| active accepted=19/24 '
        '| correct-and-accepted=16/24'
    )

    for old, new in (
        ('02_paired_vehicle_predictions.csv', 'input_v115_predictions.csv'),
        ('10_end_to_end_gate_metrics.csv', 'input_v115_gate.csv'),
        ('12_manifest.json', 'input_v115_manifest.json'),
        ('script_snapshot_v115.py', 'input_v115_script_snapshot.py')
    ):
        shutil.copy2(run / old, OUT / new)

    say('[V115 VERIFIED] {} checksum-verified artifacts'.format(
        len(inventory)
    ))

    return run

def download(name, url):
    dest = DATA / name

    if dest.is_file() and dest.stat().st_size > 1000:
        try:
            with gzip.open(dest, 'rb') as stream:
                stream.read(2048)

            say('[CACHE] {} | {:.1f} MiB | SHA256={}'.format(
                name, dest.stat().st_size / 1048576, sha(dest)
            ))
            return dest
        except Exception:
            dest.unlink()

    temp = DATA / (name + '.partial')

    for attempt in (1, 2, 3):
        try:
            say('[DOWNLOAD] {} | attempt {}/3'.format(
                name, attempt
            ))

            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Paper7-External-Locked-Audit/1.0'
                }
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                with temp.open('wb') as output:
                    total = 0
                    next_report = 5 * 1048576

                    while True:
                        block = response.read(1048576)
                        if not block:
                            break
                        output.write(block)
                        total += len(block)

                        if total >= next_report:
                            say('[DOWNLOADING] {} | {:.1f} MiB'.format(
                                name, total / 1048576
                            ))
                            next_report += 5 * 1048576

            require(
                temp.stat().st_size > 1000,
                'Downloaded file too small'
            )

            with gzip.open(temp, 'rb') as stream:
                stream.read(4096)

            temp.replace(dest)

            say('[DOWNLOADED] {} | {:.2f} MiB | SHA256={}'.format(
                name, dest.stat().st_size / 1048576, sha(dest)
            ))
            return dest

        except Exception as error:
            say('[RETRY] {} | {}: {}'.format(
                name, type(error).__name__, str(error)[:160]
            ))
            if temp.exists():
                temp.unlink()
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError('Download failed: ' + name)

def geo_metadata(source):
    fields = {
        '!Sample_geo_accession': [],
        '!Sample_title': []
    }

    with gzip.open(
        source, 'rt', encoding='utf-8', errors='replace'
    ) as stream:
        for line in stream:
            if not line.startswith('!Sample_'):
                continue

            row = next(csv.reader(
                [line.rstrip('\r\n')],
                delimiter='\t',
                quotechar='"'
            ))
            if row[0] in fields:
                fields[row[0]] = row[1:]

    ids = fields['!Sample_geo_accession']
    titles = fields['!Sample_title']

    require(
        len(ids) == len(titles) == 282
        and len(set(ids)) == 282,
        'GSE149428 GEO sample inventory mismatch'
    )

    expression = re.compile(
        r'^\s*(MCF7|LNCaP)\s+([A-Za-z0-9]+)_([0-9]+)([ABC])\s*$',
        re.IGNORECASE
    )

    rows = []

    for gsm, title in zip(ids, titles):
        match = expression.fullmatch(title)

        rows.append(dict(
            gsm=gsm,
            title=title,
            cell_line=match.group(1) if match else '',
            condition_code=match.group(2) if match else '',
            hours=int(match.group(3)) if match else np.nan,
            replicate=match.group(4).upper() if match else '',
            parsed=bool(match)
        ))

    metadata = save(
        '04_external_geo_sample_metadata.csv',
        rows
    )

    eligible = metadata[
        metadata.parsed
        & metadata.cell_line.isin(CELL_LINES)
        & metadata.condition_code.isin(('DMSO',) + DRUGS)
        & metadata.hours.isin(TIMES)
    ]

    for cell in CELL_LINES:
        subset = eligible[
            eligible.cell_line == cell
        ]

        expected = {
            (condition, hour, replicate)
            for condition in ('DMSO',) + DRUGS
            for hour in TIMES
            for replicate in ('A', 'B', 'C')
        }

        observed = set(zip(
            subset.condition_code,
            subset.hours.astype(int),
            subset.replicate
        ))

        require(
            len(subset) == 72
            and len(observed) == 72
            and observed == expected,
            'DMSO and single-drug time grid incomplete for ' + cell
        )

        say(
            '[EXTERNAL DESIGN VERIFIED | {}] '
            'DMSO + T/M/W | 6 times | A/B/C | 72 samples'
            .format(cell)
        )

    save(
        '05_external_eligible_timecourse.csv',
        eligible
    )

    say(
        '[EXTERNAL GEO] 282 samples | eligible={} '
        '| unparsed labels={}'
        .format(len(eligible), int((~metadata.parsed).sum()))
    )

    return metadata, eligible

def inspect_processed(path, metadata):
    csv.field_size_limit(10**7)

    with gzip.open(
        path, 'rt', encoding='utf-8-sig',
        errors='replace', newline=''
    ) as stream:
        reader = csv.reader(stream)
        head = []

        for _ in range(4):
            try:
                head.append(next(reader))
            except StopIteration:
                break

        require(
            head and len(head[0]) >= 4,
            'Empty expression table: ' + path.name
        )

        width = len(head[0])
        data_rows = len(head) - 1

        sample_ids = set(metadata.gsm.astype(str))
        sample_titles = set(metadata.title.astype(str))

        def norm(value):
            return re.sub(
                '[^a-z0-9]', '', str(value).lower()
            )

        gsm_matches = sum(
            str(v).strip() in sample_ids
            for v in head[0]
        )
        normalized_titles = {
            norm(v) for v in sample_titles
        }
        title_matches = sum(
            norm(v) in normalized_titles
            for v in head[0]
        )

        for row_number, row in enumerate(reader, start=5):
            require(
                len(row) == width,
                'CSV row {} has unexpected width in {}'.format(
                    row_number, path.name
                )
            )
            data_rows += 1

        say(
            '[EXTERNAL TABLE] {} | columns={} | data rows={} '
            '| GSM header matches={} | title matches={}'
            .format(
                path.name,
                width,
                data_rows,
                gsm_matches,
                title_matches
            )
        )

        for i, row in enumerate(head[:3]):
            say(
                '[EXTERNAL HEADER {} | row {}] first={} | last={}'
                .format(
                    path.name,
                    i + 1,
                    ' ; '.join(str(v)[:55] for v in row[:8]),
                    ' ; '.join(str(v)[:45] for v in row[-4:])
                )
            )

        return dict(
            file=path.name,
            bytes=path.stat().st_size,
            sha256=sha(path),
            columns=width,
            rows_below_header=data_rows,
            exact_GSM_header_matches=gsm_matches,
            normalized_title_header_matches=title_matches,
            header_0_to_11=' ; '.join(
                str(v)[:90] for v in head[0][:12]
            )
        )

def main():
    say('=' * 94)
    say(
        'PAPER 7 | V116 | FREEZE V115 AND '
        'DISCOVER INDEPENDENT GSE149428'
    )
    say(
        'NO NEW MODEL FIT | GEO METADATA AND FILE '
        'HEADER AUDIT | NO GIT'
    )
    say('=' * 94)

    require(
        len(sys.argv) == 1,
        'Run Python script.py with no flags'
    )

    freeze_spec()
    v115_run = verify_v115()

    paths = {
        name: download(name, url)
        for name, url in FILES.items()
    }

    metadata, eligible = geo_metadata(
        paths['GSE149428_series_matrix.txt.gz']
    )

    profiles = [
        inspect_processed(paths[name], metadata)
        for name in FILES
        if name.endswith('.csv.gz')
    ]

    save(
        '06_external_file_profiles.csv',
        profiles
    )

    manifest = dict(
        version='v116',
        source_GSE149428=FILES,
        source_file_sha256={
            name: sha(path)
            for name, path in paths.items()
        },
        geo_sample_count=len(metadata),
        eligible_sample_count=len(eligible),
        v115_manifest_sha256=sha(
            v115_run / '12_manifest.json'
        ),
        v115_code_sha256=sha(
            v115_run / 'script_snapshot_v115.py'
        ),
        script_sha256=sha(__file__),
        external_prediction_performed=False,
        external_labels_not_used_for_model_selection=True,
        no_git=True,
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__
    )

    (OUT / '07_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
        encoding='utf-8'
    )

    (OUT / '08_methods_limits.md').write_text(
        """# V116 — external discovery and method freeze

GSE194200 was already examined in V113-V115 and is
a development dataset, not untouched external validation.

GSE149428 is a separate published study. Its expression
headers are audited here without fitting or evaluating a model.

The planned evaluation retrains the same fixed algorithm
on the new study's training subset. It does not transfer
the original mouse gene coefficients into human cells.

Replicate letters A/B/C are not automatically proof of
independent biological donors or experimental batches.
That status requires verification.

The target is duration of exposure, not biological onset.

All candidate methods must receive access to the same
time-labelled DMSO references in the test experiment.

Processed expression must not be treated as raw counts.
Its log scale and normalization need verification first.

Gene count, feature rule and regularization are frozen
from V115 before examination of external performance.

Novelty and validated coverage are not established.
""",
        encoding='utf-8'
    )

    shutil.copy2(
        Path(__file__),
        OUT / 'script_snapshot_v116.py'
    )

    say(
        '[GATE] External expression sample-column mapping '
        'NOT YET VERIFIED; model evaluation NOT RUN'
    )
    say(
        '[COMPLETE] Frozen V115 and external GEO '
        'source audit archived'
    )

def finish(ok):
    say('[RESULT] ' + ('SUCCESS' if ok else 'FAILED'))
    say('[REPORT] ' + str(OUT / 'console_summary.txt'))

    archive = OUT / 'portable_report_v116.zip'
    say('[ZIP] ' + str(archive))

    (OUT / 'console_summary.txt').write_text(
        '\n'.join(LOG) + '\n',
        encoding='utf-8'
    )

    items = []

    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            archive.name, '09_sha256_inventory.csv'
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT / '09_sha256_inventory.csv',
        index=False,
        encoding='utf-8-sig'
    )

    with zipfile.ZipFile(
        archive,
        'w',
        zipfile.ZIP_DEFLATED,
        compresslevel=4
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

        for p in sorted(DATA.iterdir()):
            if p.is_file() and p.name in FILES:
                z.write(p, 'external_source/' + p.name)

    say(
        '[ZIP COMPLETE] frozen evidence, '
        'source files, metadata, code and hashes'
    )

if __name__ == '__main__':
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say('[ERROR]\n' + traceback.format_exc())

    try:
        finish(ok)
    except Exception:
        ok = False
        print(
            '[ARCHIVE ERROR]\n' + traceback.format_exc(),
            flush=True
        )

    if not ok:
        sys.exit(1)
