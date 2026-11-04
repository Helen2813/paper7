#!/usr/bin/env python3
"""Paper 7 | One final two-component sensitivity audit.

Read-only input: SHA256-pinned GSE246591 GEO counts, existing per-feature
conditional projection export, and verified previous selection summary.
Output: a new timestamped, separate directory; no model fitting, data download,
Git changes, or edits to earlier results.

R = range(shifted curve) / range(observed curve) -- dynamic signal retention.
S = ||(I-P_[1,f]) shifted|| / ||shifted-centered|| -- affine shape separation.
S is undefined for a constant shifted curve, not zero.
All t<0 shifted values are fixed to f(0), using piecewise-linear interpolation.
"""
from __future__ import annotations
import gzip, hashlib, html, json, math, shutil, sys, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis_output'
RAW = ROOT / 'data/GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz'
PRIOR = ANALYSIS / 'gse246591_delta_gene_level.csv.gz'
PRIOR_SUMMARY = ROOT / 'evidence/gse246591_delta_summary.csv'
SHA = '1d340b52b0eac564f73fe892ab28560748a122a7d3d51e21aa3c13e8a667cb08'
TIMES = np.array([0.,3.,6.,9.,12.,18.,24.,36.,48.,96.])
CULTURES = ('S169','S170','S506','S508')
TAUS = (3,6,12,24)
BASE_GATE = 0.5
SHIFT_GATE = 0.5
NEAR = 0.05          # existing descriptive threshold (not calibrated)
SMALL_SHAPE = 0.20   # new DISPLAY-ONLY heuristic, not a statistical cutoff
SMALL_RATIO = 0.20   # new DISPLAY-ONLY heuristic, not a statistical cutoff
EXPECTED_OBS = {'S169':14766, 'S170':14871, 'S506':12745, 'S508':14606}
EXPECTED_FIXED = {'S169':8796, 'S170':9960, 'S506':7858, 'S508':9068}
LOG=[]
def say(*parts):
    message=' '.join(map(str,parts))
    LOG.append(message)
    print(message,flush=True)

EXPECTED_KEEP = {'S169':(14019,11746,10931,9116),
                 'S170':(13694,13584,12173,10281),
                 'S506':(11422,11254,10300,8225),
                 'S508':(13114,12775,11629,9690)}


def must(test, msg):
    if not bool(test): raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def savejson(p,obj):
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def med(arr):
    a=np.asarray(arr,dtype=float)
    a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None

def rate(cond,den):
    return float(np.count_nonzero(cond)/den) if den else None


def interpolated_shift(F,tau):
    """Exactly reproduce the 2026-11-03 measured-source finite-grid model."""
    t=np.maximum(TIMES-tau,0.)
    right=np.searchsorted(TIMES,t,side='right')
    right=np.minimum(right,len(TIMES)-1)
    left=np.maximum(right-1,0)
    d=TIMES[right]-TIMES[left]
    weight=np.divide(t-TIMES[left],d,out=np.zeros_like(t),where=d>0)
    return F[:,left]*(1-weight)[None,:]+F[:,right]*weight[None,:]


def get_metrics(F,G):
    rngF=np.ptp(F,axis=1)
    rngG=np.ptp(G,axis=1)
    x=F-F.mean(axis=1,keepdims=True)
    y=G-G.mean(axis=1,keepdims=True)
    x2=np.sum(x*x,axis=1)
    y2=np.sum(y*y,axis=1)
    slope=np.divide(np.sum(x*y,axis=1),x2,
                    out=np.zeros(len(F)),where=x2>1e-12)
    residual=y-slope[:,None]*x
    rms=np.sqrt(np.mean(residual**2,axis=1))
    delta=np.divide(rms,rngF,out=np.full_like(rms,np.nan),where=rngF>1e-12)
    R=np.divide(rngG,rngF,out=np.full_like(rngG,np.nan),where=rngF>1e-12)
    # Denominator is the centered shifted curve. If flat, S is UNDEFINED.
    S=np.divide(np.sqrt(np.sum(residual**2,axis=1)),np.sqrt(y2),
                out=np.full_like(rms,np.nan),where=y2>1e-16)
    flat=(y2<=1e-16)
    # Identity: delta = std(G)/range(F) * S (whenever S defined).
    identity=np.divide(np.sqrt(y2/len(TIMES)),rngF,
                       out=np.full_like(rngF,np.nan),where=rngF>1e-12)*S
    mask=np.isfinite(S)&(rngF>=BASE_GATE)
    must(np.allclose(delta[mask],identity[mask],atol=5e-11,rtol=2e-8),
         'Delta/shape identity failed')
    bound=np.divide(rngG,2*rngF,out=np.full_like(rngF,np.nan),where=rngF>1e-12)
    observed=(rngF>=BASE_GATE)
    must(np.all(delta[observed]<=bound[observed]+1e-9),
         'Range-based RMS bound failed')
    must(np.all(S[np.isfinite(S)]<=1+1e-9), 'Shape ratio outside [0,1]')
    must(np.all(S[np.isfinite(S)]>=-1e-9), 'Negative shape ratio')
    return rngF,rngG,rms,delta,R,S,slope,flat


def check_algebra():
    t=TIMES
    f=np.exp(.04*t)
    delayed=np.exp(.04*(t-6.))
    x=np.column_stack([np.ones(len(t)),f])
    must(np.linalg.norm(delayed-x@np.linalg.lstsq(x,delayed,rcond=None)[0])<1e-10,
         'Exact affine alias identity failed')
    must(np.isclose(np.ptp(delayed),np.exp(-.24)*np.ptp(f),rtol=1e-11),
         'Range alias identity failed')
    # A decreasing transient can be flat after a large causal shift; S undefined.
    F=np.stack([np.arange(10,dtype=float),np.ones(10)])
    G=np.stack([np.ones(10),np.ones(10)])
    _,_,_,_,_,S,_,flat=get_metrics(F,G)
    must(flat[0] and np.isnan(S[0]), 'Flat shifted response must have undefined S')
    say('[ALGEBRA] projection identity / range bound / flat S=undefined: PASS')


def summarise(culture,tau,label,mask,metrics):
    f,g,rms,delta,R,S,slope,flat,near=metrics
    n=int(mask.sum()); nnear=int((near&mask).sum())
    nnf=int(np.count_nonzero(mask&flat))
    valid=mask&~flat
    return dict(culture=culture,tau_h=tau,stratum=label,n=n,
        nearzero_n=nnear,nearzero_pct=(100*nnear/n if n else None),
        shifted_flat_n=nnf,
        median_dynamic_ratio_R=med(R[mask]),
        median_shape_ratio_S=med(S[valid]),
        median_delta_normalized=med(delta[mask]),
        median_R_among_nearzero=med(R[mask&near]),
        median_S_among_nearzero_defined=med(S[mask&near&~flat]),
        nearzero_R_below_0p2_n=int(np.count_nonzero(mask&near&(R<SMALL_RATIO))),
        nearzero_S_below_0p2_n=int(np.count_nonzero(mask&near&~flat&(S<SMALL_SHAPE))),
        nearzero_R_and_S_below_0p2_n=int(np.count_nonzero(mask&near&~flat&(R<SMALL_RATIO)&(S<SMALL_SHAPE))),
        nearzero_flat_shift_n=int(np.count_nonzero(mask&near&flat)),
        nearzero_retained_dynamic_and_low_S_n=int(np.count_nonzero(mask&near&(g>=SHIFT_GATE)&~flat&(S<SMALL_SHAPE))),
        negative_slope_n=int(np.count_nonzero(mask&(slope<0))))


def table_html(df):
    return df.to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}',na_rep='—')


def figure_svg(frame):
    # Four panels: near-zero percentages vs candidate shifts on three denominators.
    width,height=1020,690
    pal={'observed_dynamic':'#64748b','retained_shift':'#0f766e',
         'excluded_shift':'#d97706','fixed4':'#6d28d9'}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
          '<rect width="100%" height="100%" fill="white"/>',
          '<text x="30" y="28" font-size="20" font-weight="bold" fill="#132c40">Near-zero projection residuals: denominator matters</text>',
          '<text x="30" y="48" font-size="12" fill="#64748b">GSE246591 | delta / observed range &lt; 0.05 (descriptive; no p-values)</text>']
    def sx(t,x0): return x0+64+(t-3)/(24-3)*367
    for i,c in enumerate(CULTURES):
        x0=22+(i%2)*508;y0=65+(i//2)*306
        parts += [f'<rect x="{x0}" y="{y0}" width="485" height="284" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>',
                  f'<text x="{x0+22}" y="{y0+25}" font-size="17" font-weight="bold" fill="#1e293b">{c}</text>']
        ytop=y0+43;ybottom=y0+241
        d=frame[frame.culture.eq(c)]
        maxval=max(1.0,float(d.nearzero_pct.max())*1.08)
        step=math.ceil(maxval/4)
        ymax=max(2,step*4)
        def sy(v): return ybottom-(v/ymax)*(ybottom-ytop)
        for j in range(5):
            yy=sy(ymax*j/4)
            parts.append(f'<line x1="{x0+64}" y1="{yy:.1f}" x2="{x0+447}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{x0+55}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{ymax*j/4:g}%</text>')
        for t in TAUS:
            xx=sx(t,x0)
            parts.append(f'<text x="{xx:.1f}" y="{ybottom+20}" text-anchor="middle" font-size="11" fill="#475569">{t}h</text>')
        for lab,color in pal.items():
            rows=d[d.stratum.eq(lab)].sort_values('tau_h')
            pts=' '.join(f'{sx(int(r.tau_h),x0):.1f},{sy(float(r.nearzero_pct)):.1f}' for r in rows.itertuples(index=False))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for r in rows.itertuples(index=False):
                parts.append(f'<circle cx="{sx(int(r.tau_h),x0):.1f}" cy="{sy(float(r.nearzero_pct)):.1f}" r="3.1" fill="{color}"/>')
    for i,(k,col) in enumerate(pal.items()):
        x=40+i*240;y=684
        parts.append(f'<line x1="{x}" y1="{y-4}" x2="{x+23}" y2="{y-4}" stroke="{col}" stroke-width="3"/>')
        parts.append(f'<text x="{x+29}" y="{y}" font-size="12" fill="#334155">{html.escape(k.replace("_"," "))}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    say('PAPER 7 | FINAL R/S DUAL-COMPONENT IDENTIFIABILITY AUDIT')
    say('OFFLINE | FROZEN GEO | NO NEW MODEL | NO FLAGS | NO GIT')
    check_algebra()
    for p in [RAW,PRIOR,PRIOR_SUMMARY]:must(p.is_file(),'Missing frozen input: '+str(p))
    must(sha(RAW)==SHA,'Source GEO hash mismatch. No results generated.')
    expected=pd.read_csv(PRIOR_SUMMARY)
    must(len(expected)==16,'Historical 16-setting summary absent')
    # Load and validate all 40 columns exactly. Import is read only.
    df=pd.read_csv(RAW,compression='gzip',sep='\t',low_memory=False)
    must(len(df)==60676 and len(df.columns)==41,'Wrong original source dimensions')
    names=[f'{c}_{int(t)}' for c in CULTURES for t in TIMES]
    must(df.columns[0]=='Ensemvl.ID' and set(df.columns[1:])==set(names),
         'GEO count columns differ from verified source')
    counts=df[names].apply(pd.to_numeric,errors='raise').to_numpy(np.float64)
    must(np.isfinite(counts).all() and (counts>=0).all() and
         np.allclose(counts,np.rint(counts),rtol=0,atol=1e-8),'Invalid source counts')
    lib=counts.sum(axis=0)
    must((lib>1000).all(),'Missing/empty count sample')
    Z=np.log2(1+counts/lib[None,:]*1e6)
    features=df.iloc[:,0].astype(str).to_numpy()
    say('[SOURCE] Original GEO SHA256',SHA,'| 60676 features x 40 named samples')
    # Prior per-feature export provides independent row-wise parity for all tau.
    prior=pd.read_csv(PRIOR,usecols=['culture','tau_h','feature_id','delta_relative_to_range','range_log2cpm','shifted_range_log2cpm','included_in_summary'])
    must(len(prior)==60676*16,'Frozen per-gene delta export incomplete')
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    OUT=ANALYSIS/f'dual_component_identifiability_{now}'
    OUT.mkdir(parents=True,exist_ok=False)
    say('[OUTPUT]',OUT)
    results=[]
    per_gene_out=OUT/'per_feature_R_S.csv.gz'
    with gzip.open(per_gene_out,'wt',encoding='utf-8',newline='') as fcsv:
        for i,c in enumerate(CULTURES):
            F=Z[:,i*10:(i+1)*10]
            rgF=np.ptp(F,axis=1)
            base=rgF>=BASE_GATE
            must(int(base.sum())==EXPECTED_OBS[c],f'Observed gate differs for {c}')
            # Compute all shifts so fixed4 denominator can be constructed before summaries.
            cache={}
            for tau in TAUS:
                G=interpolated_shift(F,tau)
                metrics=get_metrics(F,G)
                rgG=metrics[1]
                keep=base&(rgG>=SHIFT_GATE)
                cache[tau]=(metrics,keep)
            fixed=np.logical_and.reduce([cache[t][1] for t in TAUS])
            must(int(fixed.sum())==EXPECTED_FIXED[c],f'fixed4 mismatch: {c}')
            for j,tau in enumerate(TAUS):
                (rngF,rngG,rms,delta,R,S,slope,flat),keep=cache[tau]
                must(int(keep.sum())==EXPECTED_KEEP[c][j],f'Historical selection mismatch: {c}/{tau}')
                old=expected[(expected.culture==c)&(expected.tau_h==tau)].iloc[0]
                must(np.isclose(float(old.median_delta_over_range),np.median(delta[keep]),atol=3e-8,rtol=0),
                     f'Historical delta median mismatch: {c}/{tau}')
                # Independent per-feature comparison against prior export.
                q=prior[(prior.culture==c)&(prior.tau_h==tau)]
                must(len(q)==len(features) and np.array_equal(q.feature_id.astype(str).to_numpy(),features),
                     f'Gene order/ID mismatch: {c}/{tau}')
                must(np.allclose(q.range_log2cpm.to_numpy(float),rngF,rtol=0,atol=1e-8) and
                     np.allclose(q.shifted_range_log2cpm.to_numpy(float),rngG,rtol=0,atol=1e-8),
                     f'Frozen range parity failed: {c}/{tau}')
                must(np.array_equal(q.included_in_summary.to_numpy(bool),keep),
                     f'Frozen inclusion parity failed: {c}/{tau}')
                must(np.allclose(q.delta_relative_to_range.to_numpy(float)[base],delta[base],atol=1e-8,rtol=0),
                     f'Frozen delta parity failed: {c}/{tau}')
                near=delta<NEAR
                mat=(rngF,rngG,rms,delta,R,S,slope,flat,near)
                strata={
                    'observed_dynamic':base,
                    'retained_shift':keep,
                    'excluded_shift':base&~keep,
                    'fixed4':fixed
                }
                for k,mask in strata.items(): results.append(summarise(c,tau,k,mask,mat))
                detail=pd.DataFrame(dict(
                    culture=c,feature_id=features,tau_h=tau,
                    observed_range=rngF,shifted_range=rngG,
                    observed_dynamic=base,retained_shift=keep,fixed4=fixed,
                    R_dynamic_ratio=R,S_shape_ratio=S,
                    delta_normalized=delta,delta_RMS=rms,
                    affine_amplitude=slope,shifted_flat=flat,
                    nearzero_delta=near
                ))
                detail.to_csv(fcsv,index=False,header=(i==0 and j==0),float_format='%.9g')
                say(f'[CHECK] {c} tau={tau} observed={base.sum()} retained={keep.sum()} excluded={(base&~keep).sum()} fixed4={fixed.sum()} flat_shifted_on_observed={np.count_nonzero(base&flat)}')
    summaries=pd.DataFrame(results)
    must(len(summaries)==64,'Expected 64 summary rows')
    summaries.to_csv(OUT/'summary_64_strata.csv',index=False,float_format='%.10g')
    # Direct comparison for the 24h shift, retaining the four original culture units.
    subset=summaries[summaries.tau_h.eq(24)].copy()
    cols=['culture','stratum','n','nearzero_n','nearzero_pct','median_dynamic_ratio_R',
          'median_shape_ratio_S','median_R_among_nearzero','median_S_among_nearzero_defined',
          'nearzero_flat_shift_n','nearzero_R_below_0p2_n',
          'nearzero_S_below_0p2_n','nearzero_retained_dynamic_and_low_S_n']
    subset[cols].to_csv(OUT/'table_24h_two_component.csv',index=False,float_format='%.10g')
    (OUT/'figure_selection_denominator.svg').write_text(figure_svg(summaries),encoding='utf-8')
    # Overleaf-ready table: the original dynamic denominator, shifted-selected,
    # and fixed4 cohorts must not be interpreted as independent samples.
    texrows=[]
    for c in CULTURES:
        for label in ['observed_dynamic','retained_shift','excluded_shift','fixed4']:
            z=subset[(subset.culture==c)&(subset.stratum==label)].iloc[0]
            pct=f'{z.nearzero_pct:.2f}' if z.nearzero_pct is not None else '---'
            rmed=f'{z.median_dynamic_ratio_R:.3f}' if z.median_dynamic_ratio_R is not None else '---'
            smed=f'{z.median_shape_ratio_S:.3f}' if z.median_shape_ratio_S is not None else '---'
            texrows.append(f'{c} & {label.replace("_", " ")} & {z.n} & {pct} & {rmed} & {smed} '+'\\\\')
    latex=(r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}'+'\n'
           +r'\caption{Two-component audit at a modelled 24-h shift in GSE246591. $R$ is shifted/observed range and $S$ is residual shape separation, undefined for constant shifted profiles. The near-zero fraction uses a descriptive, non-calibrated threshold $\delta_{RMS}/\mathrm{range}(f)<0.05$. Denominators are distinct culture-feature strata, not biological replicates.}'+'\n'
           +r'\label{tab:dual-component}\begin{tabular}{llrrrr}\toprule Culture & Stratum & $n$ & Near-zero (\%) & Median $R$ & Median $S$ \\ \midrule'+'\n'
           +'\n'.join(texrows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
    (OUT/'dual_component_table_24h.tex').write_text(latex,encoding='utf-8')
    provenance=dict(dataset='GSE246591',input_sha256=SHA,
       archived_delta_sha256=sha(PRIOR),archived_summary_sha256=sha(PRIOR_SUMMARY),
       transform='log2(1+CPM)',interpolation='piecewise-linear; pre-0h flat',
       projection='signed amplitude + free baseline',
       dynamic_ratio='R=range(f_shifted)/range(f_observed)',
       shape_ratio='S=norm((I-P_[1,f])f_shifted)/norm(f_shifted-mean(f_shifted))',
       shape_ratio_for_flat='undefined; saved as NA, not zero',
       delta_relation='delta_RMS/range(f)=std(f_shifted)/range(f)*S for nonflat f_shifted',
       nearzero_delta_threshold=NEAR,display_only_small_R=SMALL_RATIO,
       display_only_small_S=SMALL_SHAPE,
       dynamic_gate=BASE_GATE,shifted_gate=SHIFT_GATE,
       independent_biological_replicates_available=False,
       onset_ground_truth_available=False,calibrated_error_rates=False,
       selection_bias_direction_claim='not universal',
       no_rerun_of_training=True,source_rows=60676,culture_n=4,times=TIMES.tolist(),
       tau_h=list(TAUS))
    savejson(OUT/'manifest.json',provenance)
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper 7: Dual-component audit</title>
<style>body{font:14px system-ui,sans-serif;color:#1e293b;max-width:1130px;margin:35px auto;padding:0 15px;background:#f8fafc}h1{font-size:26px}section{background:#fff;border:1px solid #dce4ef;border-radius:12px;padding:17px;margin:14px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:11px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#f1f5f9}p{line-height:1.55}.note{border-left:4px solid #be8c39;padding:14px;background:#fff8e6}</style>
<h1>GSE246591 | Dual-component delay geometry</h1>
<p class="note">R measures retained dynamic range. S measures affine shape separation, conditional on a nonflat shifted curve. Flat shifted curves have undefined S. Neither is calibrated biological onset inference. Display-only cutoffs R&lt;0.2 and S&lt;0.2 do not define statistical significance.</p>
<section><h2>Near-zero projection rates by denominator</h2>'''+ (OUT/'figure_selection_denominator.svg').read_text(encoding='utf-8')+'''</section><section><h2>24-hour shift: two components</h2>'''+table_html(subset[cols])+'''</section><section><h2>All 64 summaries</h2>'''+table_html(summaries)+'''</section></html>'''
    (OUT/'index.html').write_text(body,encoding='utf-8')
    # Generate a conservative supplement-ready paragraph with no fabricated interpretation.
    (OUT/'README_interpretation.txt').write_text(
      'R=shifted/original dynamic range, S=normalized affine projection residual.\n'
      'Examine whether near-zero delta among excluded features coincides with small R even when S remains moderate/high.\n'
      'Fixed4 represents genes that passed both range gates for all four candidate shifts in the same culture.\n'
      'Use continuous R and S medians; thresholds 0.2 and 0.05 are descriptive, not calibrated.\n'
      'Do not label small S as a true biological onset ambiguity without shape and noise assumptions.\n'
      'No new independent data, validation, p-value or statistical power statement.\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'p7_dual_component_audit.py')
    say('[SUMMARY: tau24] retained / excluded / fixed4 by culture')
    for c in CULTURES:
        d=subset[subset.culture.eq(c)]
        for r in d.itertuples(index=False):
            say('[R/S]',c,r.stratum,'n=',r.n,'near%=',f'{r.nearzero_pct:.3f}',
                'median_R=',r.median_dynamic_ratio_R,'median_S=',r.median_shape_ratio_S,
                'near_flat=',r.nearzero_flat_shift_n)
    say('[READY]',OUT)
    say('[ZIP]',OUT/'p7_dual_component_audit.zip')
    say('[HTML]',OUT/'index.html')
    (OUT/'console.log').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
    files=[p for p in sorted(OUT.iterdir()) if p.is_file()]
    savejson(OUT/'sha256_manifest.json',{p.name:sha(p) for p in files})
    archive=OUT/'p7_dual_component_audit.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:must(z.testzip() is None,'ZIP integrity failure')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[STOP]',type(e).__name__,str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(1)
