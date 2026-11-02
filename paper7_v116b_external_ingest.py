#!/usr/bin/env python3
"""Paper 7 V116B: rescue external GEO ingest. No model fitting."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import csv, gzip, json, re, shutil, sys, time, traceback
import urllib.request, zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'GSE149428'
OUT = ROOT / 'paper7_v116b_external_ingest' / (
    'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
)
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=False)
LOG = []

BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149428/'
SUPPS = {
    'GSE149428_Source_Data_File_1_log_cpm_viab.csv.gz':
        BASE + 'suppl/GSE149428_Source_Data_File_1_log_cpm_viab.csv.gz',
    'GSE149428_Source_Data_File_2_lncap_expression.csv.gz':
        BASE + 'suppl/GSE149428_Source_Data_File_2_lncap_expression.csv.gz'
}
MATRIX = 'GSE149428_series_matrix.txt.gz'
SOFT = 'GSE149428_family.soft.gz'

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def check(x, msg):
    if not bool(x):
        raise RuntimeError(msg)

def digest(p):
    h = sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def save(name, columns, rows):
    with (OUT / name).open(
        'w', encoding='utf-8-sig', newline=''
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    say('[OUTPUT] {} | {} rows'.format(name, len(rows)))

def freeze():
    parent = ROOT / 'paper7_v116_external_lock'
    candidates = sorted((
        r for r in parent.glob('run_*')
        if r.is_dir()
        and (r/'01_frozen_external_protocol.json').is_file()
        and (r/'input_v115_script_snapshot.py').is_file()
        and (r/'02_v115_frozen_metrics.csv').is_file()
    ), reverse=True)

    check(candidates, 'Original V116 frozen protocol not found')
    previous = candidates[0]
    spec_file = previous/'01_frozen_external_protocol.json'
    spec = json.loads(spec_file.read_text(encoding='utf-8'))

    check(
        spec.get('status') ==
        'FROZEN_BEFORE_EXTERNAL_EXPRESSION_AUDIT'
        and spec.get('external_study') == 'GSE149428'
        and spec.get('development_algorithm') ==
        'V115 paired_effect_ridge_amplitude'
        and spec.get('gene_count') == 200
        and spec.get('variance_floor') == 0.25
        and spec.get('ridge_penalty_per_gene') == 0.25,
        'Original model lock differs: DO NOT automatically revise'
    )

    for name in (
        '01_frozen_external_protocol.json',
        '02_v115_frozen_metrics.csv',
        '03_v115_paired_discordances.csv',
        'input_v115_script_snapshot.py'
    ):
        shutil.copy2(previous/name, OUT/name)

    say(
        '[FREEZE VERIFIED] SHA256={} | '
        'genes=200, ridge=0.25, variance floor=0.25'
        .format(digest(spec_file))
    )
    say('[MODEL] No fitting or test-label inspection')
    return spec_file

def gzip_ok(path):
    try:
        with gzip.open(path, 'rb') as f:
            return bool(f.read(1))
    except (OSError, EOFError):
        return False

def download(name, url, optional=False):
    path = DATA / name

    if path.is_file() and gzip_ok(path):
        say(
            '[CACHE] {} | {} bytes'
            .format(name, path.stat().st_size)
        )
        return path

    errors = []
    partial = DATA / (name+'.partial')

    for trial in (1,2,3):
        try:
            say(
                '[DOWNLOAD] {} | attempt {}/3'
                .format(name,trial)
            )

            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent':
                    'Paper7-GEO-External-ReadOnly/1.0'
                }
            )

            next_report = 5 * 1048576

            with urllib.request.urlopen(
                req, timeout=120
            ) as source:
                with partial.open('wb') as target:
                    total = 0
                    while True:
                        buf = source.read(1048576)
                        if not buf:
                            break
                        target.write(buf)
                        total += len(buf)

                        if total >= next_report:
                            say(
                                '[DOWNLOADING] {} | {:.1f} MiB'
                                .format(name,total/1048576)
                            )
                            next_report += 5*1048576

            check(
                partial.is_file() and gzip_ok(partial),
                'No valid gzip content'
            )

            partial.replace(path)

            say(
                '[DOWNLOADED] {} | {} bytes | SHA256={}'
                .format(
                    name,
                    path.stat().st_size,
                    digest(path)
                )
            )
            return path

        except Exception as e:
            errors.append(
                '{}: {}'.format(
                    type(e).__name__,str(e)[:160]
                )
            )
            if partial.exists():
                partial.unlink()

            say('[RETRY] '+errors[-1])

            if trial < 3:
                time.sleep(trial*2)

    if optional:
        say(
            '[OPTIONAL UNAVAILABLE] {} | {}'
            .format(name,' / '.join(errors))
        )
        return None

    raise RuntimeError(
        'Required download failed: '+name+
        ' | '+' / '.join(errors)
    )

def matrix_metadata(path):
    if path is None:
        return []

    ids, names = [], []

    with gzip.open(
        path,'rt',encoding='utf-8',errors='replace'
    ) as f:
        for line in f:
            if line.startswith('!Sample_geo_accession\t'):
                ids = next(csv.reader(
                    [line.rstrip('\r\n')],
                    delimiter='\t'
                ))[1:]

            elif line.startswith('!Sample_title\t'):
                names = next(csv.reader(
                    [line.rstrip('\r\n')],
                    delimiter='\t'
                ))[1:]

    say(
        '[MATRIX] compressed={} bytes | '
        'accession labels={} | sample titles={}'
        .format(
            path.stat().st_size,len(ids),len(names)
        )
    )

    if len(ids)==len(names)==282:
        return list(zip(ids,names))

    say(
        '[MATRIX] Insufficient sample metadata; '
        'attempting family SOFT'
    )
    return []

def soft_metadata(path):
    records = []
    gsm, title = None, None

    with gzip.open(
        path,'rt',encoding='utf-8',errors='replace'
    ) as f:
        for line in f:
            if line.startswith('^SAMPLE = '):
                if gsm is not None:
                    records.append((gsm,title or ''))
                gsm = line.split('=',1)[1].strip()
                title = None

            elif line.startswith('!Sample_title = ') and gsm is not None:
                title = line.split('=',1)[1].strip()

    if gsm is not None:
        records.append((gsm,title or ''))

    say(
        '[SOFT] sample accessions={} | titles={}'
        .format(
            len(records),
            sum(bool(t) for _,t in records)
        )
    )
    return records

def metadata_rows(pairs, origin):
    check(
        len(pairs)==282,
        'GEO must contain 282 individual GSM accessions'
    )

    ids = [s for s,_ in pairs]
    check(
        len(set(ids))==282
        and all(re.fullmatch(r'GSM\d+',s) for s in ids),
        'GEO GSM inventory malformed'
    )

    regex = re.compile(
        r'^(MCF7|LNCaP)\s+([A-Z0-9]+)_(\d+)([ABC])$',
        re.I
    )

    rows = []
    for gsm,title in pairs:
        m = regex.fullmatch(title.strip())

        if m:
            cell,trt,hour,rep = m.groups()
            cell = (
                'MCF7' if cell.upper()=='MCF7'
                else 'LNCaP'
            )
            trt = trt.upper()
            hour = int(hour)
            rep = rep.upper()
        else:
            cell,trt,hour,rep = '','','',''

        eligible = (
            cell in ('MCF7','LNCaP')
            and trt in ('DMSO','T','M','W')
            and hour in (0,3,6,9,12,24)
        )

        rows.append(dict(
            gsm=gsm,
            title=title,
            cell_line=cell,
            condition_code=trt,
            hours=hour,
            replicate=rep,
            eligible=eligible,
            metadata_source=origin
        ))

    save(
        '04_GEO_verified_sample_metadata.csv',
        [
            'gsm','title','cell_line',
            'condition_code','hours','replicate',
            'eligible','metadata_source'
        ],
        rows
    )

    for cell in ('MCF7','LNCaP'):
        part = [
            r for r in rows
            if r['eligible'] and r['cell_line']==cell
        ]
        obs = {
            (r['condition_code'],r['hours'],r['replicate'])
            for r in part
        }
        exp = {
            (drug,t,rep)
            for drug in ('DMSO','T','M','W')
            for t in (0,3,6,9,12,24)
            for rep in 'ABC'
        }

        check(
            len(part)==72 and obs==exp,
            'Missing/duplicated required samples: '+cell
        )

        say(
            '[DESIGN VERIFIED | {}] '
            '72/72 DMSO + three monotherapies; '
            'A/B/C x six times'
            .format(cell)
        )

    say(
        '[METADATA] 282 GSM, 144 eligible '
        'time-course samples, {} other'
        .format(len(rows)-144)
    )
    return rows

def profile(path, metadata):
    csv.field_size_limit(10**8)

    norm = lambda s: re.sub(
        '[^a-z0-9]','',str(s).lower()
    )

    title_map = {
        norm(r['title']) for r in metadata
    }
    gsm_map = {
        r['gsm'] for r in metadata
    }

    rows = []

    with gzip.open(
        path,'rt',encoding='utf-8-sig',
        errors='replace',newline=''
    ) as f:
        reader = csv.reader(f)

        for i,row in enumerate(reader):
            if i < 4:
                sample = (
                    list(row[:7])+['...']+list(row[-4:])
                    if len(row)>11 else row
                )

                say(
                    '[HEADER | {} | row {} | width={}] {}'
                    .format(
                        path.name,i+1,len(row),
                        ' ; '.join(
                            str(s)[:55] for s in sample
                        )
                    )
                )

                n_gsm = sum(
                    v.strip() in gsm_map for v in row
                )
                n_titles = sum(
                    norm(v) in title_map for v in row
                )

                rows.append(dict(
                    file=path.name,
                    row=i+1,
                    columns=len(row),
                    n_exact_gsm=n_gsm,
                    n_title_matches=n_titles,
                    first_values=';'.join(
                        str(x)[:80] for x in row[:8]
                    )
                ))

            if i == 4:
                say(
                    '[CSV] {} | scanning remaining '
                    'rows for gzip integrity'
                    .format(path.name)
                )

            if i%10000==0 and i>0:
                say(
                    '[CSV] {} | {} records scanned'
                    .format(path.name,i)
                )

        line_count = i+1

    say(
        '[EXTERNAL FILE] {} | gzip validated; '
        'rows={} | size={:.1f} MiB'
        .format(
            path.name,
            line_count,
            path.stat().st_size/1048576
        )
    )

    return rows,dict(
        file=path.name,
        rows=line_count,
        bytes=path.stat().st_size,
        sha256=digest(path)
    )

def main():
    say('='*89)
    say(
        'PAPER 7 | V116B | REPAIR GEO '
        'GSE149428 DOWNLOAD & METADATA'
    )
    say(
        'ORIGINAL V115 FROZEN | '
        'NO FITTING | NO EXTERNAL SCORES | NO GIT'
    )
    say('='*89)

    check(
        len(sys.argv)==1,
        'Run as python script.py without flags'
    )

    locked = freeze()

    processed = {
        name:download(name,url)
        for name,url in SUPPS.items()
    }

    matrix = download(
        MATRIX,
        BASE+'matrix/'+MATRIX,
        optional=True
    )

    pairs = matrix_metadata(matrix)
    metadata_origin = 'series_matrix'

    if not pairs:
        soft = download(
            SOFT,
            BASE+'soft/'+SOFT
        )
        pairs = soft_metadata(soft)
        metadata_origin = 'family_SOFT'

    metadata = metadata_rows(
        pairs,metadata_origin
    )

    heads,profiles = [],[]

    for path in processed.values():
        h,p = profile(path,metadata)
        heads.extend(h)
        profiles.append(p)

    save(
        '05_external_CSV_header_audit.csv',
        [
            'file','row','columns','n_exact_gsm',
            'n_title_matches','first_values'
        ],
        heads
    )

    save(
        '06_external_file_profiles.csv',
        ['file','rows','bytes','sha256'],
        profiles
    )

    shutil.copy2(
        Path(__file__),
        OUT/'script_snapshot_v116b.py'
    )

    manifest = dict(
        version='v116b',
        original_lock_SHA256=digest(locked),
        original_V115_unmodified=True,
        source_accession='GSE149428',
        metadata_provenance=metadata_origin,
        metadata_GSM=282,
        eligible_samples=144,
        external_model_predictions=0,
        frozen_settings_not_retuned=True,
        source_files={
            p.name:dict(
                sha256=digest(p),
                bytes=p.stat().st_size
            )
            for p in DATA.iterdir()
            if p.is_file() and p.suffix=='.gz'
        },
        script_sha256=digest(__file__)
    )

    (OUT/'07_manifest.json').write_text(
        json.dumps(manifest,indent=2)+'\n',
        encoding='utf-8'
    )

    say(
        '[GATE] External data acquired and GEO '
        'metadata checked; expression-column '
        'mapping remains TO VERIFY'
    )
    say(
        '[NEXT] Examine headers; do NOT fit '
        'external data until mapping and '
        'log2CPM scale verified'
    )

def finish(ok):
    say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
    say('[REPORT] '+str(OUT/'console_summary.txt'))

    zip_path = OUT/'portable_report_v116b.zip'
    say('[ZIP] '+str(zip_path))

    (OUT/'console_summary.txt').write_text(
        '\n'.join(LOG)+'\n',
        encoding='utf-8'
    )

    inv = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            '08_sha256_inventory.csv',zip_path.name
        ):
            inv.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    save(
        '08_sha256_inventory.csv',
        ['filename','bytes','sha256'],
        inv
    )

    with zipfile.ZipFile(
        zip_path,'w',zipfile.ZIP_DEFLATED,
        compresslevel=3
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=zip_path:
                z.write(p,p.name)

        for p in sorted(DATA.iterdir()):
            if (
                p.is_file()
                and p.suffix=='.gz'
                and gzip_ok(p)
            ):
                z.write(
                    p,'external_source/'+p.name
                )

    say(
        '[ZIP COMPLETE] Frozen specification, '
        'external sources, headers, hashes, logs'
    )

if __name__=='__main__':
    ok=True

    try:
        main()
    except Exception:
        ok=False
        say('[ERROR]\n'+traceback.format_exc())

    try:
        finish(ok)
    except Exception:
        ok=False
        print(
            '[ARCHIVE ERROR]\n'+traceback.format_exc(),
            flush=True
        )

    if not ok:
        sys.exit(1)
