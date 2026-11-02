"""Local-only OnsetAudit HTTP interface. Start: python onsetaudit/app.py"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import sys
import threading
import traceback
import webbrowser
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, unquote

from core import Dataset, VERSION

HERE = Path(__file__).resolve().parent
WEB = HERE / 'web' / 'index.html'
GENERATED = HERE / '_generated'
INPUTS = GENERATED / 'inputs'
REPORTS = GENERATED / 'reports'
for d in (INPUTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
DATA = Dataset()
LOCK = threading.RLock()
TOKEN = secrets.token_urlsafe(24)
MAX_UPLOAD = 150 * 1024 * 1024


def safe_name(s: str) -> str:
    name = Path(str(s).replace('\\', '/')).name
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)[:100] or 'source.csv'


class Handler(BaseHTTPRequestHandler):
    server_version = f'OnsetAudit/{VERSION}'
    def log_message(self, fmt, *args):
        print('[OnsetAudit]', fmt % args, flush=True)

    def _origin_ok(self) -> bool:
        # Standard browsers send Origin on fetch POST. Reject cross-site requests.
        origin = self.headers.get('Origin')
        if origin is not None:
            return origin in {f'http://127.0.0.1:{self.server.server_port}',
                              f'http://localhost:{self.server.server_port}'}
        return True

    def _send(self, code: int, body: bytes, content_type='application/json', filename=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy',
                         "default-src 'self' 'unsafe-inline' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{safe_name(filename)}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, value, code=200):
        self._send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))

    def _protected(self):
        if not self._origin_ok() or self.headers.get('X-OnsetAudit-Token') != TOKEN:
            self._json({'error': 'Invalid local session or forbidden origin'}, 403)
            return False
        return True

    def _payload(self) -> bytes:
        raw_len = self.headers.get('Content-Length')
        if not raw_len or not raw_len.isdigit():
            raise ValueError('Missing Content-Length')
        n = int(raw_len)
        if n > MAX_UPLOAD:
            raise ValueError('Upload larger than 150 MiB not supported')
        data = self.rfile.read(n)
        if len(data) != n:
            raise ValueError('Incomplete upload')
        return data

    def do_GET(self):
        route = urlsplit(self.path)
        try:
            if route.path in ('/', '/index.html'):
                content = WEB.read_text('utf-8').replace('ONSET_AUDIT_SESSION_TOKEN', TOKEN)
                return self._send(200, content.encode('utf-8'), 'text/html; charset=utf-8')
            if route.path == '/favicon.ico':
                return self._send(404, b'', 'text/plain')
            if route.path == '/api/health':
                return self._json({'version': VERSION, 'local_only': True, 'model_status': 'EXPERIMENTAL'})
            if route.path == '/api/summary':
                with LOCK:
                    return self._json(DATA.summary())
            if route.path == '/api/genes':
                q = parse_qs(route.query).get('q', [''])[0]
                with LOCK:
                    return self._json({'genes': DATA.genes(q)})
            if route.path == '/api/curve':
                p = parse_qs(route.query)
                gene = p.get('gene', [''])[0]
                drugs = p.get('drug') or None
                with LOCK:
                    return self._json(DATA.curve(gene, drugs))
            if route.path == '/api/design':
                p = parse_qs(route.query)
                ta = float(p.get('earlier', ['0'])[0])
                tb = float(p.get('later', ['6'])[0])
                with LOCK:
                    return self._json(DATA.design_audit(ta, tb))
            if route.path == '/api/report':
                with LOCK:
                    report = DATA.report()
                    import numpy as np
                    import pandas as pd
                    report["runtime_provenance"] = {
                        "python": sys.version,
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "core_source_sha256": hashlib.sha256((HERE / "core.py").read_bytes()).hexdigest(),
                        "server_source_sha256": hashlib.sha256((HERE / "app.py").read_bytes()).hexdigest(),
                    }
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                        report_data = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                        z.writestr('report.json', report_data)
                        z.writestr('README.txt',
                            'OnsetAudit research artifact. No raw counts included.\n'
                            'Recorded exposure duration is NOT biological transcription onset.\n'
                            'Source SHA256 and analysis settings are in report.json.\n')
                        evaluation = report.get('evaluation')
                        if evaluation:
                            import pandas as pd
                            for key, fn in (('predictions','active_predictions.csv'),
                                            ('vehicle_predictions','vehicle_predictions.csv'),
                                            ('folds','holdout_folds.csv'),
                                            ('vehicle_drug_predictions','vehicle_by_drug.csv')):
                                table = evaluation.get(key, [])
                                if table:
                                    z.writestr(fn, pd.DataFrame(table).to_csv(index=False))
                            diagnostics = evaluation.get('v04_audit', {})
                            for key, fn in (("by_drug", "v04_by_drug.csv"),
                                            ("by_group", "v04_by_group.csv"),
                                            ("by_time", "v04_by_time.csv"),
                                            ("threshold_sweep", "v04_threshold_sweep_posthoc.csv")):
                                rows = diagnostics.get(key, [])
                                if rows:
                                    z.writestr(fn, pd.DataFrame(rows).to_csv(index=False))
                        manifest = {'version': VERSION, 'generated_utc': stamp,
                                    'report_sha256': hashlib.sha256(report_data.encode('utf-8')).hexdigest()}
                        z.writestr('sha256_manifest.json', json.dumps(manifest, indent=2))
                    dest = REPORTS / f'onsetaudit_{stamp}.zip'
                    dest.write_bytes(buf.getvalue())
                    return self._send(200, buf.getvalue(), 'application/zip', filename=dest.name)
            return self._json({'error': 'Not found'}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            return self._json({'error': str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            return self._json({'error': 'Internal processing error: ' + type(exc).__name__}, 500)

    def do_POST(self):
        route = urlsplit(self.path)
        if not self._protected():
            return
        try:
            if route.path in ('/api/import/metadata', '/api/import/matrix'):
                payload = self._payload()
                filename = safe_name(unquote(self.headers.get('X-File-Name', 'upload.csv')))
                digest = hashlib.sha256(payload).hexdigest()
                with LOCK:
                    if route.path.endswith('metadata'):
                        summary = DATA.import_metadata(payload, filename)
                    else:
                        params = parse_qs(route.query)
                        kind = params.get('kind', ['counts'])[0]
                        summary = DATA.import_matrix(payload, filename, kind)
                    dest = INPUTS / (digest[:20] + '_' + filename)
                    if not dest.exists():
                        dest.write_bytes(payload)
                    return self._json({'summary': summary, 'source_saved_locally': str(dest), 'sha256': digest})
            if route.path == '/api/demo':
                with LOCK:
                    return self._json({'summary': DATA.demonstration()})
            if route.path == '/api/reset':
                with LOCK:
                    DATA.__dict__.update(Dataset().__dict__)
                    return self._json({'summary': DATA.summary()})
            if route.path == '/api/evaluate':
                body = self._payload()
                obj = json.loads(body or b'{}')
                drugs = obj.get('drugs')
                if drugs is not None and (not isinstance(drugs, list) or not all(isinstance(x, str) for x in drugs)):
                    raise ValueError('drugs must be a list of treatment names')
                with LOCK:
                    result = DATA.evaluate(drugs=drugs)
                    small = {k:v for k,v in result.items() if k not in ('predictions','vehicle_predictions')}
                    small['preview_active'] = result['predictions'][:25]
                    small['preview_vehicle'] = result['vehicle_predictions'][:15]
                    return self._json(small)
            return self._json({'error': 'Unknown action'}, 404)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self._json({'error': str(exc)}, 400)
        except Exception:
            traceback.print_exc()
            return self._json({'error': 'Internal processing error'}, 500)


def main():
    if not WEB.is_file():
        raise SystemExit('Missing onsetaudit/web/index.html')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    address = f'http://127.0.0.1:{server.server_port}/'
    print(f'[ONSetAudit] v{VERSION} — LOCAL ONLY', flush=True)
    print('[OPEN]', address, flush=True)
    print('[DATA] Uploaded files stay locally under', INPUTS, flush=True)
    print('[STOP] Press Ctrl+C in Terminal', flush=True)
    if '--no-browser' not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print('[STOPPED] Local server closed', flush=True)

if __name__ == '__main__':
    main()
