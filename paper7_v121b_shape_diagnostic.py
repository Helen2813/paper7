from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from collections import Counter
import csv, gzip, json, sys, traceback, zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE/'data'/'GSE214965'
OUT = BASE/'paper7_v121b_shape_diagnostic'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
OUT.mkdir(parents=True, exist_ok=False)
LOG=[]
SOURCES = {
 'metadata': ('GSE214965_sample_metadata_LN229_DRUG_Seq.txt.gz','89a69aa0a3d8fc25147bd1009aaebb5621ac67af73e5ef33f7204349a2ace28d'),
 'counts': ('GSE214965_rawcounts_LN229_DRUG_Seq.txt.gz','82446902882766ce0ae8aff69ba42edf8951b7f974125fb52a7c9b56de0e5db6'),
 'normalized': ('GSE214965_normalizedcounts_LN229_DRUG_Seq.txt.gz',None)
}
def say(s):
 s=str(s);print(s,flush=True);LOG.append(s)
def sha(p):
 h=sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def save(n,d):
 pd.DataFrame(d).to_csv(OUT/n,index=False,encoding='utf-8-sig');say('[OUTPUT] '+n+' | '+str(len(d))+' rows')
def audit(name,path):
 sizes=Counter(); exceptions=[]; cols=None; nonempty=0; blank=0; leading=[]
 with gzip.open(path,'rt',encoding='utf-8-sig',errors='replace',newline='') as f:
  for line,row in enumerate(csv.reader(f,delimiter='\t'),1):
   if not row or all(not str(v).strip() for v in row):
    blank+=1
    if len(exceptions)<12: exceptions.append(dict(file=name,line=line,width=len(row),problem='blank',first='',last=''))
    continue
   nonempty+=1; sizes[len(row)]+=1
   if cols is None:
    cols=row
    say(f'[HEADER | {name}] width={len(cols)} | first={repr(cols[:4])} | last={repr(cols[-3:])}')
   elif len(row)!=len(cols) and len(exceptions)<25:
    exceptions.append(dict(file=name,line=line,width=len(row),problem='width_mismatch',first=repr(row[:4])[:180],last=repr(row[-4:])[:180]))
   if len(leading)<3:leading.append(dict(file=name,line=line,width=len(row),first=repr(row[:5])[:240],last=repr(row[-5:])[:240]))
 say(f'[WIDTH AUDIT | {name}] nonempty={nonempty} blank={blank} distributions={dict(sorted(sizes.items()))}')
 bad=sum(n for k,n in sizes.items() if k!=len(cols))
 say(f'[WIDTH DIAGNOSIS | {name}] header={len(cols)} bad_nonempty_rows={bad} first_anomalies={sum(r["problem"]=="width_mismatch" for r in exceptions)}')
 return cols,exceptions,leading,[dict(file=name,width=k,n_rows=n) for k,n in sorted(sizes.items())], dict(file=name,bytes=path.stat().st_size,sha256=sha(path),header_width=len(cols),nonempty_rows=nonempty,blank_rows=blank,mismatch_nonempty_rows=bad)
def main():
 say('PAPER 7 | V121B | LOCAL TSV SHAPE DIAGNOSTIC | READ ONLY | NO GIT')
 if len(sys.argv)!=1:raise RuntimeError('No flags')
 profiles=[];anomalies=[];first=[];dist=[];headers={}
 for name,(filename,expected) in SOURCES.items():
  p=DATA/filename
  if not p.exists():
   say('[NOT DOWNLOADED] '+filename);continue
  current=sha(p)
  if expected and current!=expected:raise RuntimeError('Source hash changed: '+filename)
  say('[SOURCE VERIFIED] '+filename+' | SHA256='+current)
  cols,a,b,d,profile=audit(name,p)
  headers[name]=cols;anomalies+=a;first+=b;dist+=d;profiles.append(profile)
 if not {'metadata','counts'}.issubset(headers):raise RuntimeError('Need cached rawcounts and metadata')
 meta=pd.read_csv(DATA/SOURCES['metadata'][0],sep='\t',dtype=str)
 for col in ('sample_column_name','library','drug','timepoint','condition'):
  if col not in meta.columns:raise RuntimeError('Metadata missing '+col)
 say(f'[METADATA] rows={len(meta)} unique_sample_columns={meta.sample_column_name.nunique()}')
 groups=meta.groupby(['library','timepoint','drug'],dropna=False).size().reset_index(name='n_samples')
 save('04_library_time_drug_counts.csv',groups)
 samples=set(meta.sample_column_name.astype(str))
 matched=set(str(v) for v in headers['counts'] if str(v) in samples)
 say(f'[HEADER CROSSCHECK] counts columns matching metadata={len(matched)}/{len(samples)} | unmatched metadata={len(samples-matched)}')
 for value in sorted(samples-matched)[:8]: say('[UNMATCHED METADATA] '+value)
 save('01_width_distribution.csv',dist)
 save('02_anomalous_rows.csv',anomalies)
 save('03_first_rows.csv',first)
 save('05_file_provenance.csv',profiles)
 (OUT/'06_manifest.json').write_text(json.dumps(dict(version='v121b',source_files=profiles,no_model_fitting=True,read_only=True,original_v121_unchanged=True,script_sha256=sha(Path(__file__))),indent=2)+'\n',encoding='utf-8')
 say('[GATE] Only formatting / sample metadata audited. No imputation or row dropping.')
def finish(ok):
 say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
 (OUT/'console_summary.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
 (OUT/'script_snapshot_v121b.py').write_bytes(Path(__file__).read_bytes())
 items=[dict(filename=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('07_sha256_inventory.csv','portable_report_v121b.zip')]
 save('07_sha256_inventory.csv',items)
 z=OUT/'portable_report_v121b.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in sorted(OUT.iterdir()):
   if p.is_file() and p!=z:f.write(p,p.name)
 say('[ZIP] '+str(z))
if __name__=='__main__':
 ok=True
 try:main()
 except Exception:ok=False;say('[ERROR]\n'+traceback.format_exc())
 try:finish(ok)
 except Exception:ok=False;print('[ARCHIVE ERROR]\n'+traceback.format_exc(),flush=True)
 if not ok:sys.exit(1)
