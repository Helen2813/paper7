from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
from itertools import product
import json,sys,traceback,zipfile,shutil
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper7_v120_twostage_audit'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
OUT.mkdir(parents=True,exist_ok=False)
LOG=[]
MODES=('training_time_matched','training_pooled_vehicle')
KEY=['cell_line','replicate','source_drug','target_drug','true_hour']
QUERY=['cell_line','replicate','target_drug','true_hour']

def say(s):
    s=str(s);print(s,flush=True);LOG.append(s)

def need(x,s):
    if not bool(x):raise RuntimeError(s)

def sha(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.15g')
    say('[OUTPUT] {} | {} rows'.format(name,len(d)))
    return d

def boolify(v):
    s=v.astype(str).str.lower().str.strip()
    need(s.isin(['true','false','1','0']).all(),'Bad boolean value')
    return s.isin(['true','1'])

def main():
    say('='*83)
    say('PAPER 7 | V120 | POSTHOC TWO-STAGE AUDIT OF GSE149428')
    say('READ-ONLY V119 | NO MODEL FIT | NO NEW EXTERNAL VALIDATION | NO GIT')
    say('='*83)
    need(len(sys.argv)==1,'Run without flags')
    base=ROOT/'paper7_v119_noncircular_vehicle_gate'
    needed=('10_sha256_inventory.csv','09_manifest.json','01_protocol.json',
            '02_noncircular_per_source_decisions.csv','script_snapshot_v119.py')
    runs=sorted((r for r in base.glob('run_*') if r.is_dir() and
                 all((r/f).is_file() for f in needed)),reverse=True)
    need(runs,'Complete V119 run not found')
    run=runs[0]
    inv=pd.read_csv(run/'10_sha256_inventory.csv',encoding='utf-8-sig')
    need({'filename','bytes','sha256'}.issubset(inv.columns) and inv.filename.is_unique,
         'Invalid V119 checksum inventory')
    for r in inv.itertuples(index=False):
        p=run/str(r.filename)
        need(p.name==str(r.filename) and p.is_file() and
             p.stat().st_size==int(r.bytes) and sha(p)==str(r.sha256),
             'Archive integrity failure: '+str(r.filename))
    man=json.loads((run/'09_manifest.json').read_text('utf-8'))
    proto=json.loads((run/'01_protocol.json').read_text('utf-8'))
    need(man.get('version')=='v119' and man.get('decisions')==720 and
         proto.get('reference_modes')==list(MODES),'V119 provenance mismatch')
    script=ROOT/'paper7_v119_noncircular_vehicle_gate.py'
    need(script.is_file() and sha(script)==sha(run/'script_snapshot_v119.py'),
         'V119 script was modified')
    say('[V119 VERIFIED] {} files | SHA256 code={}'.format(len(inv),sha(script)))

    df=pd.read_csv(run/'02_noncircular_per_source_decisions.csv',encoding='utf-8-sig')
    fields=KEY+['reference_mode','same_drug','is_DMSO','accepted_active',
                'predicted_hour','correct_time','correct_and_accepted']
    need(len(df)==720 and set(fields).issubset(df.columns) and
         not df.duplicated(KEY+['reference_mode']).any(),
         'V119 decisions missing or duplicated')
    for c in ('same_drug','is_DMSO','accepted_active','correct_time','correct_and_accepted'):
        df[c]=boolify(df[c])
    expected=set(product(('MCF7','LNCaP'),('A','B','C'),('T','M','W'),
                         ('DMSO','T','M','W'),(3,6,9,12,24),MODES))
    got=set(map(tuple,df[KEY+['reference_mode']].itertuples(index=False,name=None)))
    need(got==expected,'Missing/extra experimental conditions')
    need((df.same_drug==(df.source_drug==df.target_drug)).all() and
         (df.is_DMSO==(df.target_drug=='DMSO')).all(),
         'Inconsistent treatment flags')
    correct=(~df.is_DMSO)&(df.predicted_hour==df.true_hour)
    need((df.correct_time==correct).all() and
         (df.correct_and_accepted==(df.correct_time&df.accepted_active)).all(),
         'Saved decision flags inconsistent')

    active=df[(df.target_drug!='DMSO') & df.same_drug]
    m=active[active.reference_mode==MODES[0]][KEY+['accepted_active','correct_time','predicted_hour']].rename(columns={
        'accepted_active':'matched_accept','correct_time':'matched_correct','predicted_hour':'matched_prediction'})
    p=active[active.reference_mode==MODES[1]][KEY+['accepted_active','correct_time','predicted_hour']].rename(columns={
        'accepted_active':'pooled_accept','correct_time':'pooled_correct','predicted_hour':'pooled_prediction'})
    a=m.merge(p,on=KEY,validate='one_to_one')
    need(len(a)==90 and not a.duplicated(QUERY).any(),'Active sample pairing failed')
    a['hybrid_accept']=a.pooled_accept
    a['hybrid_prediction_if_accepted']=a.matched_prediction.where(a.hybrid_accept,np.nan)
    a['hybrid_correct_and_accepted']=a.pooled_accept&a.matched_correct
    a=save('02_active_hybrid_predictions.csv',a)

    v=df[(df.target_drug=='DMSO')&(df.reference_mode==MODES[1])]
    need(len(v)==90,'Pooled DMSO source calls missing')
    u=v.groupby(QUERY,as_index=False).agg(
        n_source_models_positive=('accepted_active','sum'),
        any_model_positive=('accepted_active','any'))
    u['rejected_DMSO']=~u.any_model_positive
    need(len(u)==30,'DMSO query pairing failed')
    save('03_DMSO_unique_control_decisions.csv',u)

    n_correct=int(a.matched_correct.sum())
    n_accept=int(a.pooled_accept.sum())
    n_joint=int(a.hybrid_correct_and_accepted.sum())
    n_control=int(u.rejected_DMSO.sum())
    lower=max(0,n_correct+n_accept-90)
    upper=min(n_correct,n_accept)
    need((n_correct,n_accept,n_control,int(a.pooled_correct.sum()))==(84,85,29,63),
         'V119 aggregate replay mismatch')
    need(lower<=n_joint<=upper,'Joint success bound failed')
    save('04_hybrid_summary.csv',[dict(
        n_active=90,n_DMSO=30,pooled_gate_accepted=n_accept,
        time_matched_correct_before_gate=n_correct,
        hybrid_correct_and_accepted=n_joint,
        active_rejected=90-n_accept,
        accepted_wrong_time=int((a.pooled_accept&~a.matched_correct).sum()),
        DMSO_rejected=n_control,DMSO_false_active=30-n_control,
        lower_bound=lower,upper_bound=upper,
        assessment='POSTHOC_DEVELOPMENT_NOT_EXTERNAL_VALIDATION')])
    say('[V120 HYBRID] correct AND accepted={}/90 | rejected active={}/90 | accepted wrong time={}/90'.format(
        n_joint,90-n_accept,int((a.pooled_accept&~a.matched_correct).sum())))
    say('[V120 DMSO] rejected={}/30 | false positive={}/30 | any-source rule'.format(n_control,30-n_control))
    say('[V120 BOUNDS] required 79..84/90 | observed={}/90'.format(n_joint))

    summaries=[]
    for c in ('cell_line','target_drug','replicate','true_hour'):
        for k,g in a.groupby(c):
            summaries.append(dict(partition=c,value=k,n=len(g),
                                  accepted=int(g.pooled_accept.sum()),
                                  joint_correct=int(g.hybrid_correct_and_accepted.sum()),
                                  accepted_wrong=int((g.pooled_accept&~g.matched_correct).sum())))
    save('05_hybrid_active_strata.csv',summaries)
    save('06_hybrid_errors.csv',a[~a.hybrid_correct_and_accepted])
    save('07_hybrid_DMSO_false_alarms.csv',u[u.any_model_positive])

    for name,decisions in (
        ('pooled_one_stage',a.pooled_accept&a.pooled_correct),
        ('matched_one_stage',a.matched_accept&a.matched_correct),
        ('two_stage_posthoc',a.hybrid_correct_and_accepted)):
        say('[V120 COMPARISON] {} correct-and-accepted={}/90'.format(name,int(decisions.sum())))

    (OUT/'01_protocol.json').write_text(json.dumps(dict(
        version='v120',source_V119_manifest_sha256=sha(run/'09_manifest.json'),
        action='join existing V119 predictions without fitting or tuning',
        gate='training_pooled_vehicle',time='training_time_matched',
        active_queries_require_known_source_drug=True,
        DMSO_unknown_drug_rule='any-source active -> false positive',
        architecture_selected_after_V119_outcomes=True,
        independent_validation=False,novelty_unproven=True,
        biological_onset_truth=False,author_normalization_precedes_splits=True,
        replicate_batch_independence_unverified=True,
        no_git=True,script_sha256=sha(__file__)),indent=2)+'\n',encoding='utf-8')
    (OUT/'08_caveats.md').write_text('''# V120 limitations

This hybrid was designed AFTER inspecting V119 outcomes and must not be
presented as independent external validation. Both components are existing
V119 decisions and no model was refitted or retuned.

Active queries assume the assigned drug identity is known and use its source
program. DMSO controls are conservatively called active if ANY of three drug
programs calls active. This is not a symmetric drug-blind detection task.

Publisher logCPM was normalized and plate-adjusted before splitting.
Replicate labels A/B/C are not proven independent biological batches.
The target is experimentally recorded exposure duration, not true onset.

A genuinely untouched third study is required for the locked hybrid, as well
as a formal novelty comparison and a realistic control-availability protocol.
''',encoding='utf-8')
    shutil.copy2(__file__,OUT/'script_snapshot_v120.py')
    for name in ('01_protocol.json','02_noncircular_per_source_decisions.csv',
                 '05_aggregate_gate_metrics.csv','09_manifest.json','script_snapshot_v119.py'):
        shutil.copy2(run/name,OUT/('input_v119_'+name))
    say('[BOUNDARY] This is posthoc design on previously seen GSE149428, not confirmatory')
    say('[COMPLETE] Verified two-stage outcome and negative controls archived')

def finish(ok):
    say('[RESULT] '+('SUCCESS' if ok else 'FAILED'))
    archive=OUT/'portable_report_v120.zip'
    say('[REPORT] '+str(OUT/'console_summary.txt'))
    say('[ZIP] '+str(archive))
    (OUT/'console_summary.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
    items=[dict(filename=p.name,bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(OUT.iterdir()) if p.is_file() and
           p.name not in ('09_sha256_inventory.csv',archive.name)]
    pd.DataFrame(items).to_csv(OUT/'09_sha256_inventory.csv',index=False,encoding='utf-8-sig')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)

if __name__=='__main__':
    ok=True
    try:main()
    except Exception:
        ok=False;say('[ERROR]\n'+traceback.format_exc())
    try:finish(ok)
    except Exception:
        ok=False;print('[ARCHIVE ERROR]\n'+traceback.format_exc(),flush=True)
    if not ok:sys.exit(1)
