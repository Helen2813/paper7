#!/usr/bin/env python3
"""GSE246591 conditional shift-amplitude projection geometry, no CLI arguments.

OUTPUT IS DESCRIPTIVE, NOT AN ESTIMATOR OF MOLECULAR ONSET.
Primary measured source: verified V103 frozen 8-gene culture trajectories, if present.
Fallback: GEO depositor counts with exact 40 sample-column identity, log2(1+CPM).
Finite-grid f_tau is a MODELLED piecewise-linear curve with a flat pre-0h boundary;
interpolation is not an additional observation, and unknown shape can change delta.
No p-values, biological noise/specificity claims, or arbitrary count-level retuning.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EVIDENCE = PROJECT / 'evidence'
FIGURES = PROJECT / 'figures'
OUTPUT = PROJECT / 'analysis_output'
for p in (EVIDENCE, FIGURES, OUTPUT): p.mkdir(parents=True, exist_ok=True)
GEO_URL = ('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/GSE246591/suppl/'
           'GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz')
COUNTS_NAME = 'GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz'
TIMES = np.array([0.,3.,6.,9.,12.,18.,24.,36.,48.,96.])
CULTURES = ['S169','S170','S506','S508']
TAUS = (3.,6.,12.,24.)
CORE_PANEL = ['PRL','IGFBP1','FOXO1','HAND2','WNT4','PGR','IL11','LEFTY2']
MIN_DYNAMIC_LOG2CPM = 0.5
MIN_SHIFT_DYNAMIC_LOG2CPM = 0.5
HEURISTIC_RELATIVE_CUTOFF = 0.05


def stop(message):
    raise RuntimeError('[NOT EVALUABLE] '+message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def writejson(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def expected_mapping():
    rows=pd.read_csv(EVIDENCE/'GSE246591_verified_sample_manifest.csv')
    if len(rows)!=40 or rows.gsm.nunique()!=40 or rows.duplicated(['culture','time_hours']).any():
        stop('The verified GEO 40-sample manifest is invalid')
    if set(rows.culture)!=set(CULTURES) or set(rows.time_hours)!=set(TIMES):
        stop('The 4 x 10 culture/time design differs')
    return rows


def find_v103():
    roots=[Path.home()/'Desktop'/'paper7',PROJECT.parent,PROJECT]
    sources=[]
    for root in roots:
        sources.extend(root.glob('paper7_measured_identifiability_v103/run_*/input_v67_measured_source.csv'))
        sources.extend(root.glob('paper7_gse403_locked_external_v67/run_*/frozen_v31_source.csv'))
    for src in sorted(set(sources),reverse=True):
        try:
            records=pd.read_csv(src,encoding='utf-8-sig')
            required={'culture_label','gene','time_hours','log2CPM'}
            if not required.issubset(records.columns): continue
            records=records[records.gene.isin(CORE_PANEL)&records.culture_label.isin(CULTURES)].copy()
            if len(records)!=320 or records.duplicated(['culture_label','gene','time_hours']).any():continue
            if not (records.groupby(['culture_label','gene']).size()==10).all():continue
            if not np.isfinite(records.log2CPM.to_numpy(float)).all():continue
            if any(set(records[records.culture_label.eq(c)].time_hours)!=set(TIMES) for c in CULTURES):continue
            manifest=src.with_name('11_manifest.json') if src.name.startswith('input_v67') else src.with_name('08_manifest.json')
            if not manifest.exists():continue
            meta=json.loads(manifest.read_text(encoding='utf-8'))
            expected=str(meta.get('v67_source_sha256',meta.get('source_sha256',''))).lower()
            if len(expected)!=64 or digest(src).lower()!=expected:continue
            return src,records
        except Exception:continue
    return None,None


def find_counts():
    candidates=[PROJECT/'data'/COUNTS_NAME,Path.home()/'Desktop'/'paper7'/'data'/COUNTS_NAME,
                Path.home()/'Downloads'/COUNTS_NAME,PROJECT/COUNTS_NAME]
    for c in candidates:
        if c.is_file():return c
    dest=PROJECT/'data'/COUNTS_NAME
    dest.parent.mkdir(parents=True,exist_ok=True)
    print('[DOWNLOAD]',GEO_URL,flush=True)
    req=urllib.request.Request(GEO_URL,headers={'User-Agent':'Paper7-RECOMB2027-repro/1.0'})
    tmp=dest.with_name('.'+dest.name+'.partial')
    try:
        with urllib.request.urlopen(req,timeout=90) as resp, tmp.open('wb') as f:
            n=0
            while True:
                block=resp.read(1<<18)
                if not block:break
                n+=len(block)
                if n>100_000_000:stop('GEO file unexpectedly large')
                f.write(block)
        with gzip.open(tmp,'rb') as f:
            if not f.read(500):stop('GEO file appears empty')
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print('[SOURCE DOWNLOAD] bytes=',dest.stat().st_size,flush=True)
    return dest


def normalized_name(s):return re.sub('[^a-z0-9]+','',str(s).lower())


def match_count_columns(df,manifest):
    names=[str(c) for c in df.columns]
    matched={}
    unknown=[]
    for row in manifest.itertuples(index=False):
        possibilities=[]
        for col in names:
            canon=normalized_name(col)
            if normalized_name(row.gsm) in canon:
                possibilities.append(col);continue
            # Culture/time convention: S169_0h, S169.0h, S169 0h etc.
            key=normalized_name(f'{row.culture}{int(row.time_hours)}h')
            if canon in (key, normalized_name(f'{row.culture}_{int(row.time_hours)}')):possibilities.append(col)
        possibilities=list(dict.fromkeys(possibilities))
        if len(possibilities)!=1:
            unknown.append({'sample':row.gsm,'culture':row.culture,'time':int(row.time_hours),
                            'possible_columns':possibilities[:5]})
        else:matched[(row.culture,int(row.time_hours))]=possibilities[0]
    diagnostic={'all_columns':names,'unmatched_or_ambiguous':unknown}
    writejson(OUTPUT/'count_column_diagnostic.json',diagnostic)
    if unknown or len(matched)!=40 or len(set(matched.values()))!=40:
        stop('Cannot establish one-to-one sample-column identity from header. See analysis_output/count_column_diagnostic.json')
    return matched


def read_counts_source(manifest):
    src=find_counts()
    with gzip.open(src,'rt',encoding='utf-8-sig',errors='strict') as f:
        header=f.readline()
    sep='\t' if header.count('\t')>=header.count(',') else ','
    df=pd.read_csv(src,sep=sep,compression='gzip',low_memory=False)
    if len(df)<500 or df.shape[1]<41:stop('Unexpected counts matrix dimensions')
    mapping=match_count_columns(df,manifest)
    ordered=[mapping[(culture,int(t))] for culture in CULTURES for t in TIMES]
    counts=df[ordered].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or (counts<0).any() or not np.allclose(counts,np.rint(counts),atol=1e-8,rtol=0):
        stop('GEO count matrix contains invalid or non-integer values')
    lib=counts.sum(axis=0)
    if not (lib>1000).all():stop('Empty or implausibly small count column')
    val=np.log2(1+counts / lib[None,:]*1e6)
    feature_id=df.iloc[:,0].astype(str).to_numpy()
    print('[COUNTS VERIFIED]',len(feature_id),'features x 40 named samples')
    return src,{'mode':'GEO_COUNTS_ALL_ASSAYABLE_GENES','features':feature_id,
                'curves':{c:val[:,i*10:(i+1)*10] for i,c in enumerate(CULTURES)},
                'transform':'log2(1 + counts/library_size * 10^6)',
                'panel_gene_identity':'NOT_VERIFIED_WITHOUT_GENE_ANNOTATION',
                'input_features':int(len(feature_id))}


def read_frozen_source(src,records):
    features=np.array(CORE_PANEL,dtype=object)
    curves={}
    for c in CULTURES:
        x=[]
        for g in CORE_PANEL:
            d=records[(records.culture_label==c)&(records.gene==g)].sort_values('time_hours')
            if len(d)!=10 or not np.array_equal(d.time_hours.to_numpy(float),TIMES):
                stop('Frozen 8-gene culture trajectory has a missing or inconsistent measured time')
            x.append(d.log2CPM.to_numpy(float))
        curves[c]=np.stack(x)
    return {'mode':'V103_EIGHT_GENE_FROZEN_SOURCE','features':features,'curves':curves,
            'transform':'archived log2CPM (not recomputed from counts)',
            'panel_gene_identity':'FROZEN_8_GENE_V103','input_features':8}


def shifted_flat_pre0(F,tau,linear=True):
    """Transform f(t) at the 10 observed knots using a conditional assumed shape.

    Pre-induction signal fixed flat at f(0); between knots linearly interpolate.
    All queried time values are modelled, not extra experimental observations.
    """
    x=np.maximum(TIMES-tau,0.)
    right=np.searchsorted(TIMES,x,side='right')
    right=np.minimum(right,len(TIMES)-1)
    left=np.maximum(right-1,0)
    if linear:
        d=TIMES[right]-TIMES[left]
        w=np.divide(x-TIMES[left],d,out=np.zeros_like(x),where=d>0)
        return F[:,left]*(1-w)[None,:]+F[:,right]*w[None,:]
    return F[:,left]


def profile(F,S):
    x=F-F.mean(axis=1,keepdims=True)
    y=S-S.mean(axis=1,keepdims=True)
    var=(x*x).sum(axis=1)
    slope=np.divide((x*y).sum(axis=1),var,out=np.zeros(F.shape[0]),where=var>1e-12)
    residual=y-x*slope[:,None]
    return np.sqrt(np.mean(residual**2,axis=1)),slope


def geometry(source):
    results=[]; summary=[]
    for culture,F in source['curves'].items():
        rng=np.ptp(F,axis=1)
        for tau in TAUS:
            S=shifted_flat_pre0(F,tau,True)
            S_step=shifted_flat_pre0(F,tau,False)
            shift_rng=np.ptp(S,axis=1)
            delta,slope=profile(F,S)
            delta_step,_=profile(F,S_step)
            rel=np.divide(delta,rng,out=np.full_like(delta,np.nan),where=rng>1e-12)
            good=(rng>=MIN_DYNAMIC_LOG2CPM)&(shift_rng>=MIN_SHIFT_DYNAMIC_LOG2CPM)
            active=np.where(good)[0]
            for idx in range(F.shape[0]):
                results.append(dict(culture=culture,feature_id=str(source['features'][idx]),
                                    tau_h=int(tau),measured_knots=10,range_log2cpm=float(rng[idx]),
                                    shifted_range_log2cpm=float(shift_rng[idx]),
                                    included_in_summary=bool(good[idx]),
                                    delta_rms_log2cpm=float(delta[idx]),
                                    delta_relative_to_range=float(rel[idx]),
                                    step_interpolation_delta_rms=float(delta_step[idx]),
                                    fitted_unconstrained_amplitude=float(slope[idx])))
            if len(active):
                values=rel[active]
                rr=delta[active]
                summary.append(dict(culture=culture,tau_h=int(tau),n_candidates=F.shape[0],
                                    n_included=len(active),
                                    median_delta_rms_log2cpm=float(np.median(rr)),
                                    median_delta_over_range=float(np.median(values)),
                                    q10_delta_over_range=float(np.quantile(values,.1)),
                                    q90_delta_over_range=float(np.quantile(values,.9)),
                                    descriptive_fraction_below_0p05=float(np.mean(values<HEURISTIC_RELATIVE_CUTOFF)),
                                    median_linear_step_delta_absolute_difference=float(np.median(np.abs(rr-delta_step[active])))))
            else:
                summary.append(dict(culture=culture,tau_h=int(tau),n_candidates=F.shape[0],
                                    n_included=0,median_delta_rms_log2cpm=None,
                                    median_delta_over_range=None,q10_delta_over_range=None,
                                    q90_delta_over_range=None,descriptive_fraction_below_0p05=None,
                                    median_linear_step_delta_absolute_difference=None))
            print('[GEOMETRY]',culture,'tau',int(tau),'included',len(active),'of',F.shape[0],flush=True)
    return pd.DataFrame(results),pd.DataFrame(summary)


def build_tex(summary,source):
    if not (summary.n_included>0).all():stop('At least one culture/shift has no valid dynamic features; plot not produced')
    colors=['blue','teal','violet','orange']
    # plot centers and 10-90 bands in a compact vector figure
    lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=0.94\linewidth,height=5.5cm,',
           r'xlabel={Candidate displacement $\tau$ (hours)},ylabel={Median $\delta_{\rm RMS}$/observed range},',
           r'xmin=2,xmax=25,xtick={3,6,12,24},grid=major,grid style={gray!15},',
           r'legend style={font=\scriptsize,at={(0.02,0.98)},anchor=north west},',
           r'legend columns=2,thick,clip=false]']
    for culture,color in zip(CULTURES,colors):
        group=summary[summary.culture.eq(culture)].sort_values('tau_h')
        co=' '.join(f'({int(r.tau_h)},{r.median_delta_over_range:.8f})' for r in group.itertuples())
        lines.append(r'\addplot+[mark=*,line width=1.1pt] coordinates {'+co+r'};')
        lines.append(r'\addlegendentry{'+culture+r'}')
    lines.extend([r'\end{axis}',r'\end{tikzpicture}'])
    (FIGURES/'fig3_gse246591_measured_delta.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tab=[]
    for c in CULTURES:
        group=summary[(summary.culture.eq(c)) & summary.tau_h.isin([6,12])]
        for r in group.itertuples():
            tab.append(f'{c} & {int(r.tau_h)} & {r.n_included}/{r.n_candidates} & '
                       f'{r.median_delta_rms_log2cpm:.3f} & {r.median_delta_over_range:.3f} & '
                       f'{r.descriptive_fraction_below_0p05:.2f} \\\\')
    output=(r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}'+'\n'
            +r'\caption{Observed-grid conditional projection residuals in GSE246591. Values are medians across selected features under piecewise-linear interpolation and a flat pre-induction extension; $\tau$ is an assumed displacement, not a recovered true delay. The final column is an arbitrary descriptive $5\%$-of-range cutoff, not a calibrated error rate.}'+'\n'
            +r'\label{tab:measured-delta}\begin{tabular}{lrrrrr}\toprule Culture & $\tau$ (h) & Dynamic features & $\delta_{\rm RMS}$ & $\delta$/range & $<\!0.05$ \\ \midrule'+'\n'
            +'\n'.join(tab)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
    (EVIDENCE/'gse246591_delta_table.tex').write_text(output,encoding='utf-8')
    # Note: no p, no n-independent gene inference.
    sentence=('Conditional observed-grid projection geometry was computed from '
              + ('the SHA-256-verified historical V103 eight-gene panel' if source['mode'].startswith('V103') else 'the original depositor raw-count matrix with verified sample identities')
              +f'. After excluding trajectories with less than {MIN_DYNAMIC_LOG2CPM:g} log2CPM observed dynamic range or '
              +f'{MIN_SHIFT_DYNAMIC_LOG2CPM:g} log2CPM modelled shifted range, the analysis included '
              + 'different numbers of features by culture and shift (Table~\\ref{tab:measured-delta}). '
              + 'The residuals describe separation only under the chosen interpolation and fixed boundary rule. '
              + 'They do not yield a biological delay estimate, p-value or donor-population sensitivity.')
    (EVIDENCE/'gse246591_delta_paragraph.tex').write_text(sentence+'\n',encoding='utf-8')


def main():
    print('PAPER 7 | GSE246591 | MEASURED-SOURCE CONDITIONAL DELTA AUDIT')
    print('NO SYNTHETIC EXPRESSION | NO DELAY GROUND TRUTH | NO FLAGS')
    manifest=expected_mapping()
    # Pin the all-feature GEO count representation used in the manuscript.
    # Do not silently switch to a local eight-gene V103 archive.
    src,source=read_counts_source(manifest)
    expected='1d340b52b0eac564f73fe892ab28560748a122a7d3d51e21aa3c13e8a667cb08'
    if digest(src) != expected:
        stop('Original GSE246591 source SHA256 differs from the manuscript')
    print('[INPUT]',src,'SHA256',digest(src))
    detailed,summary=geometry(source)
    detailed.to_csv(OUTPUT/'gse246591_delta_gene_level.csv.gz',index=False,compression='gzip')
    summary.to_csv(EVIDENCE/'gse246591_delta_summary.csv',index=False,float_format='%.9g')
    build_tex(summary,source)
    record={'status':'MEASURED_SOURCE_CONDITIONAL_GEOMETRY_NOT_DELAY_INFERENCE',
            'geo_accession':'GSE246591','source_mode':source['mode'],
            'source_file':str(src),'source_sha256':digest(src),'analysis_script_sha256':digest(Path(__file__)),'geo_sample_manifest_sha256':digest(EVIDENCE/'GSE246591_verified_sample_manifest.csv'),'source_n_features':source['input_features'],
            'source_transform':source['transform'],'panel_gene_identity':source['panel_gene_identity'],
            'sample_design':'4 cultures x 10 time points; no within-culture-time replication',
            'grid_hours':TIMES.tolist(),'shifts_hours':TAUS,
            'interpolation':'piecewise linear between measured knots; f(t<0)=f(0)',
            'projection':'unconstrained least-squares onto span{1, f(T)} per gene',
            'definition':'delta_RMS=||f(max(T-tau,0))-(b+a*f(T))||_2/sqrt(10)',
            'dynamic_range_filter_log2cpm':MIN_DYNAMIC_LOG2CPM,
            'modelled_shift_range_filter_log2cpm':MIN_SHIFT_DYNAMIC_LOG2CPM,
            'heuristic_relative_cutoff':HEURISTIC_RELATIVE_CUTOFF,
            'reduced_grid_heldout_validation':False,
            'biological_delay_ground_truth':False,'empirical_noise_sigma':None,
            'claim_limited_to_conditional_design_geometry':True,
            'run_utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
            'generated_files':['evidence/gse246591_delta_summary.csv','evidence/gse246591_delta_table.tex',
                               'evidence/gse246591_delta_paragraph.tex','figures/fig3_gse246591_measured_delta.tex',
                               'analysis_output/gse246591_delta_gene_level.csv.gz']}
    writejson(EVIDENCE/'gse246591_delta_provenance.json',record)
    print('[READY] Conditional measured-source geometry computed')
    print('[WARN] No biological-onset recovery, calibrated error, or external validation claim')
    print('[REBUILD] Upload updated ZIP after regeneration; keep evidence/source SHA in supplement.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[STOP]',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
