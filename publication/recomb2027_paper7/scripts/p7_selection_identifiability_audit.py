#!/usr/bin/env python3
"""Paper 7: one-pass selection/identifiability audit of previously computed GSE246591 geometry.

NO FLAGS, NO NEW GEO DOWNLOAD, NO REFITTING, NO GROUND-TRUTH LAG CLAIMS.
Uses the existing per-feature per-culture per-shift export from the verified
GSE246591 experiment. Neither independent replicates nor calibrated p-values
are inferred. Produces a new result folder and a supplementary LaTeX table.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, math, shutil, sys, traceback, zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'analysis_output'
INPUT = BASE / 'gse246591_delta_gene_level.csv.gz'
SUMMARY = ROOT / 'evidence/gse246591_delta_summary.csv'
PROVENANCE = ROOT / 'evidence/gse246591_delta_provenance.json'
RAW = ROOT / 'data/GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz'
EXPECTED = '1d340b52b0eac564f73fe892ab28560748a122a7d3d51e21aa3c13e8a667cb08'
CULTURES = ('S169','S170','S506','S508')
TAUS = (3,6,12,24)
N = 60676
THRESHOLD = 0.5
NEAR = 0.05

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def ensure(condition,msg):
    if not condition:raise RuntimeError(msg)

def num(x):
    return None if not np.isfinite(x) else float(x)

def median(x):
    return num(np.median(x)) if len(x) else None

def frac(x):
    return num(np.mean(x)) if len(x) else None

def verify_identities():
    t=np.array([0.,3.,6.,9.,12.]); h=3.; k=.07; a=1.5;b=.9
    f=np.exp(k*t); shifted=np.exp(k*(t-h))
    ensure(np.allclose(shifted,np.exp(-k*h)*f,atol=1e-13),'Exponential affine identity failed')
    ensure(np.isclose(np.ptp(shifted),np.exp(-k*h)*np.ptp(f),rtol=1e-12),'Range identity failed')
    y=b+a*np.exp(k*(t-h))
    kk=np.log((y[2]-y[1])/(y[1]-y[0]))/3.
    ensure(abs(kk-k)<1e-12,'Noiseless exponential-rate identity failed')
    x=np.column_stack((np.ones(len(t)),f))
    dt=-k*a*np.exp(k*(t-h))
    dk=a*(t-h)*np.exp(k*(t-h))
    resid_tau=dt-x@np.linalg.lstsq(x,dt,rcond=None)[0]
    resid_k=dk-x@np.linalg.lstsq(x,dk,rcond=None)[0]
    ensure(np.linalg.norm(resid_tau)<1e-12 and np.linalg.norm(resid_k)>1e-5,
           'Delay/rate local sensitivity check failed')
    # Popoviciu: sd(v) <= range(v)/2; projection cannot exceed constant-only residual.
    example=np.array([2.,3.,4.,9.,11.]); shifted_example=np.array([1.,2.,2.1,2.2,2.4]);
    u=shifted_example-shifted_example.mean(); x=example-example.mean()
    proj=u-x*np.dot(x,u)/np.dot(x,x)
    ensure(np.linalg.norm(proj)/np.sqrt(len(proj))<=np.ptp(shifted_example)/2.+1e-12,
           'Range/RMS upper bound failed')

def main():
    print('PAPER 7 | SELECTION-CONDITIONAL IDENTIFIABILITY | READ-ONLY SOURCE AUDIT',flush=True)
    for f in (INPUT,SUMMARY,PROVENANCE,RAW):ensure(f.is_file(),'Required archived evidence missing: '+str(f))
    prov=json.loads(PROVENANCE.read_text(encoding='utf-8'))
    ensure(prov.get('source_sha256')==EXPECTED,'Provenance source hash mismatch')
    ensure(digest(RAW)==EXPECTED,'Original GEO count-file hash mismatch')
    verify_identities()
    print('[ALGEBRA] range identity, exponential-rate combination, sensitivities, RMS bound: PASS',flush=True)
    expected=pd.read_csv(SUMMARY)
    must=['culture','feature_id','tau_h','range_log2cpm','shifted_range_log2cpm',
          'included_in_summary','delta_relative_to_range','fitted_unconstrained_amplitude']
    detail=pd.read_csv(INPUT,usecols=must)
    ensure(len(detail)==4*4*N,'Unexpected number of detailed feature records')
    ensure(detail['tau_h'].notna().all(),'Missing tau')
    # Row positions are aligned explicitly below; duplicate source IDs are permitted.
    ensure(set(detail.culture)==set(CULTURES) and set(detail.tau_h)==set(TAUS),
           'Culture/shift mismatch')
    for col in ['range_log2cpm','shifted_range_log2cpm','fitted_unconstrained_amplitude']:
        ensure(np.isfinite(detail[col]).all(),'Non-finite numeric field '+col)
    ensure(detail['included_in_summary'].dtype==bool,'Eligibility flag not Boolean')
    out=BASE/('selection_identifiability_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for culture in CULTURES:
        g=detail.loc[detail.culture==culture]
        curves={int(t):g.loc[g.tau_h==t].reset_index(drop=True) for t in TAUS}
        ids=curves[TAUS[0]].feature_id.to_numpy()
        obs=curves[TAUS[0]].range_log2cpm.to_numpy(float)
        orig=obs>=THRESHOLD
        keep_by_tau={}
        for tau in TAUS:
            f=curves[tau]
            ensure(len(f)==N and np.array_equal(f.feature_id.to_numpy(),ids),
                   'Feature row ordering is not aligned across shifts: '+culture)
            ensure(np.allclose(obs,f.range_log2cpm.to_numpy(float),atol=1e-10,rtol=0),
                   'Observed ranges differ across shifts: '+culture)
            sh=f.shifted_range_log2cpm.to_numpy(float)
            keep=orig & (sh>=THRESHOLD)
            keep_by_tau[tau]=keep
            ensure(np.array_equal(keep,f.included_in_summary.to_numpy()),
                   'Archived dynamic eligibility inconsistent: '+culture+'/'+str(tau))
            old=expected[(expected.culture==culture)&(expected.tau_h==tau)]
            ensure(len(old)==1 and int(old.n_included.iloc[0])==int(keep.sum()),
                   'Historical included count changed')
            rel=f.delta_relative_to_range.to_numpy(float)
            ensure(np.isfinite(rel[orig]).all(),'Nonfinite relative residual for original-range eligible feature')
            ensure(abs(np.median(rel[keep])-float(old.median_delta_over_range.iloc[0]))<2e-8,
                   'Historical median delta changed')
        shared=np.logical_and.reduce([keep_by_tau[t] for t in TAUS])
        for tau in TAUS:
            f=curves[tau]
            keep=keep_by_tau[tau]
            excluded=orig&~keep
            rel=f.delta_relative_to_range.to_numpy(float)
            slopes=f.fitted_unconstrained_amplitude.to_numpy(float)
            sh=f.shifted_range_log2cpm.to_numpy(float)
            # Universal bound for any curve: delta_RMS <= range(f_shifted)/2.
            # The normalized residual is formed relative to ORIGINAL observed range.
            bound=np.divide(sh,2*obs,out=np.full_like(obs,np.nan),where=obs>0)
            ensure((rel[orig] <= bound[orig]+1e-8).all(),
                   'Projection/RMS theoretical bound violated; inspect provenance')
            r=dict(culture=culture,tau_h=tau,source_features=N,
                observed_range_eligible=int(orig.sum()),
                retained_both_gates=int(keep.sum()),
                excluded_only_by_shift=int(excluded.sum()),
                shift_exclusion_fraction_given_observed_gate=frac(excluded[orig]),
                median_delta_retained=median(rel[keep]),
                median_delta_shift_excluded=median(rel[excluded]),
                nearzero_frac_retained=frac(rel[keep]<NEAR),
                nearzero_frac_shift_excluded=frac(rel[excluded]<NEAR),
                nearzero_frac_observed_gate_unfiltered_for_shift=frac(rel[orig]<NEAR),
                fixed_four_shift_gene_count=int(shared.sum()),
                median_delta_fixed_four_shift_genes=median(rel[shared]),
                negative_slope_frac_retained=frac(slopes[keep]<0),
                negative_slope_frac_shift_excluded=frac(slopes[excluded]<0))
            rows.append(r)
            print('[SELECTION]',culture,'tau',tau,'observed_gate',int(orig.sum()),
                  'retained',int(keep.sum()),'shift_drop',int(excluded.sum()),
                  'near_retained',f'{r["nearzero_frac_retained"]:.4f}',
                  'near_excluded',r['nearzero_frac_shift_excluded'],
                  'fixed4',int(shared.sum()),flush=True)
    res=pd.DataFrame(rows)
    ensure(len(res)==16,'Expected 16 audit settings')
    res.to_csv(out/'selection_full_16_settings.csv',index=False,float_format='%.10g')
    # Concise LaTeX table, 3h and 24h only, not a null-hypothesis test.
    cells=[]
    for r in res.itertuples():
        if r.tau_h not in (3,24):continue
        fracdrop=r.shift_exclusion_fraction_given_observed_gate
        fm=r.nearzero_frac_retained
        fd=r.nearzero_frac_shift_excluded
        cells.append(f'{r.culture} & {r.tau_h} & {r.observed_range_eligible} & '
                     f'{r.excluded_only_by_shift} ({100*fracdrop:.1f}\\%) & '
                     f'{100*fm:.1f}\\% & '+('---' if fd is None else f'{100*fd:.1f}\\%')+r' \\')
    tex=r'''\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Exploratory selection-conditioned shift geometry on archived GSE246591 trajectories. Among features passing the observed-range gate (\(\geq0.5\) log$_2$CPM), the shift-only exclusions are counted for modelled shifts; ``near zero'' means \(\delta_{\rm RMS}/\mathrm{range}(f)<0.05\) and is not a calibrated identifiability test. All 16 rows, including the fixed-four-shift cohort, are archived in CSV.}
\label{tab:selection-audit}
\begin{tabular}{lrrrrr}\toprule
Culture & Shift h & Original eligible & Shift-only excluded & Near-zero retained & Near-zero excluded \\\midrule
'''+ '\n'.join(cells)+ '\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
    (out/'selection_audit_table.tex').write_text(tex,encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Paper7 selection audit</title></head><body>'
        '<h2>GSE246591 shift-dependent selection audit</h2><p>Descriptive, retrospective; no onset ground truth, no p-values. Source '+EXPECTED+'</p>'
        +res.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</body></html>',encoding='utf-8')
    summary={'run_utc':datetime.now(timezone.utc).isoformat(),'source':'GSE246591','source_sha256':EXPECTED,
             'per_feature_export_sha256':digest(INPUT),'existing_summary_sha256':digest(SUMMARY),
             'provenance_sha256':digest(PROVENANCE),'n_settings':16,
             'filter_observed_range':THRESHOLD,'filter_shifted_range':THRESHOLD,'nearzero_threshold':NEAR,
             'no_new_model':True,'no_independent_validation':True,'no_calibrated_noise':True,
             'interpretation':'Separate original-range eligibility from tau-dependent shifted-range exclusions; no universal direction is assumed.'}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'SHA256.txt').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in sorted(out.iterdir()) if p.is_file()),encoding='utf-8')
    archive=out/'selection_audit.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    print('[READY]',out,'\n[ZIP]',archive,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[STOP]',type(exc).__name__+':',str(exc),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(1)
