#!/usr/bin/env python3
"""Recreate Paper 7 manuscript Figures 3 and 4 from frozen GSE246591 audit summaries.

Requires completed compute, selection and dual-component audits for an
independent regeneration of the summaries. For archival reconstruction,
the two accompanying CSV summaries are used; they are not experimental
replicates or evidence of known biological lag.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
CULTURES = ('S169','S170','S506','S508')
COLORS={'S169':'#177b87','S170':'#c17a47','S506':'#6254a7','S508':'#477e54'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'pdf.fonttype':42,'savefig.bbox':'tight','figure.dpi':180})

# Use regenerated 16-row summary if present; otherwise the deposited
# independently reviewed console summary with the identical 16 records.
regenerated=EVIDENCE/'gse246591_delta_summary.csv'
if regenerated.is_file():
    d=pd.read_csv(regenerated)
    med='median_delta_over_range'; tau='tau_h'
else:
    d=pd.read_csv(EVIDENCE/'GSE246591_16_summary_user_console.csv')
    med='median_delta_over_range'; tau='tau_hours'
assert len(d)==16 and set(d.culture)==set(CULTURES)
assert set(map(int,d[tau]))=={3,6,12,24}
fig,ax=plt.subplots(figsize=(7.15,3.05))
for c in CULTURES:
    v=d[d.culture==c].sort_values(tau)
    ax.plot(v[tau],v[med],'-o',label=c,color=COLORS[c],lw=1.8,ms=5)
ax.set_xlabel('Hypothetical displacement (hours)')
ax.set_ylabel('Median normalized projection residual')
ax.set_xticks([3,6,12,24]);ax.set_xlim(2,25);ax.set_ylim(0.11,0.25)
ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
ax.legend(ncol=4,frameon=False,loc='upper left')
fig.tight_layout()
fig.savefig(FIG/'fig3_empirical_delta.pdf')
fig.savefig(FIG/'fig3_empirical_delta.png')
plt.close(fig)

# Exact aggregate counts supplied in the verified dual-component audit.
# A point represents a culture-level median, NEVER a per-feature scatter.
near=pd.read_csv(EVIDENCE/'dual_component_nearzero_strata_user_console.csv')
expected_groups=[(3,'observed_dynamic'),(24,'excluded_shift'),(24,'retained_shift')]
strata=[]
for tt,group in expected_groups:
    sub=near[(near.tau_h==tt)&(near.stratum==group)].set_index('culture').loc[list(CULTURES)]
    assert len(sub)==4
    strata.append(sub)
fig,(ax0,ax1)=plt.subplots(1,2,figsize=(7.4,3.15),gridspec_kw={'width_ratios':[1.07,1]})
styles=['o','s','^']
labels=['3 h: initial','24 h: excluded','24 h: retained']
for i,(ss,lab) in enumerate(zip(strata,labels)):
    for c in CULTURES:
        row=ss.loc[c]
        ax0.scatter(row['median_R_among_nearzero'],row['median_S_among_nearzero_defined'],
                    color=COLORS[c],marker=styles[i],s=60,edgecolors='white',linewidths=.6,zorder=4)
    ax0.plot([],[],marker=styles[i],color='#53657e',ls='',label=lab)
ax0.set_xlabel('Median retained range (R)')
ax0.set_ylabel('Median shape separation (S)')
ax0.set_xlim(0,1.08);ax0.set_ylim(0,1.12)
ax0.grid(alpha=.18)
ax0.legend(frameon=False,fontsize=9.4,loc='lower left',labelspacing=.1,handletextpad=.3)
ax0.set_title('A  Near-zero groups',loc='left',fontsize=10,weight='bold')

# Panel B pools the 4 culture-feature strata for each group. The pooled
# denominators are dependent features, not independent biological replicates.
pools=[]
for ss in strata:
    n=ss.nearzero_n.to_numpy(int)
    r=ss.nearzero_R_below_0p2_n.to_numpy(int)
    sh=ss.nearzero_S_below_0p2_n.to_numpy(int)
    pools.append((int(n.sum()),int(r.sum()),int(sh.sum())))
x=np.arange(3)
width=.33
ax1.bar(x-width/2,[100*r/n for n,r,_ in pools],width,color='#407b9d',label='Low R (<0.20)')
ax1.bar(x+width/2,[100*sh/n for n,_,sh in pools],width,color='#c27d4b',label='Low S (<0.20)')
ax1.set_xticks(x,['3 h\ninitial','24 h\nexcluded','24 h\nretained'])
ax1.set_ylabel('Share within near-zero records (%)')
ax1.set_ylim(0,106);ax1.grid(axis='y',alpha=.16)
ax1.set_title('B  Two descriptive components',loc='left',fontsize=10,weight='bold')
ax1.legend(fontsize=9.0,frameon=False,loc='upper right')
fig.tight_layout(pad=1.0)
fig.savefig(FIG/'fig4_dual_component.pdf')
fig.savefig(FIG/'fig4_dual_component.png')
plt.close(fig)
print('[FIGURES] Figure 3 and Figure 4 regenerated from verified summaries.')
print('[CAUTION] Descriptive geometry only; no lag truth, p-values, or donor independence.')
