"""Full review -> freeze -> patient analysis with SYNTHETIC GEO-shaped matrix."""
import gzip
import numpy as np
import pandas as pd

from paper7.data import create_lock
from paper7.workflow import run_locked_fit


def test_end_to_end_synthetic_locks_before_case_data(tmp_path):
    root = tmp_path / 'pilot'
    meta_dir = root / 'metadata'
    meta_dir.mkdir(parents=True)
    raw_dir = root / 'raw' / 'GSE243039'
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(1308)
    rows, reviews, expression_cols = [], [], {}
    days = np.array([0.,2.,4.,6.,8.])
    ref = 1/(1+np.exp(-(days-3.5)/1.0))
    for group, donors in [('control',['N1','N2','N3','N4']),('endometriosis',['E1','E2','E3','E4'])]:
        for donor in donors:
            for j, d in enumerate(days):
                gsm = f'SYN_{donor}_D{int(d)}'
                rows.append(dict(gse='GSE243039',gsm=gsm,title=f'{donor} day{d:g}',
                                 donor_local=donor,group=group,time_hours=float(d*24)))
                reviews.append(dict(gse='GSE243039',gsm=gsm,include='1',reviewed='1',donor_id=donor,
                                    group=group,days=str(d),treatment='EPC',exclusion_reason='',
                                    review_provenance='Independent verified synthetic fixture'))
                reference_val = .67 * ref[j] if group == 'endometriosis' else ref[j]
                # 30 genes with background, 3 coherent modules and extra individual noise
                expression_cols[gsm] = 75 + 140 * reference_val * np.r_[np.ones(10),
                                    np.linspace(.6,1.3,10),np.ones(10)] + rng.normal(0,1.2,30)
    metadata = pd.DataFrame(rows)
    meta_path = meta_dir / 'GSE243039_samples.csv'
    metadata.to_csv(meta_path,index=False)
    review_path=meta_dir/'REVIEWED.csv'
    pd.DataFrame(reviews).to_csv(review_path,index=False)
    expr = pd.DataFrame(expression_cols,index=[f'GENE{i}' for i in range(30)])
    expr.index.name='gene_symbol'
    with gzip.open(raw_dir/'GSE243039_normalized_all_counts.csv.gz','wt') as f:
        expr.to_csv(f)
    gmt = root/'programs_CURATED.gmt'
    gmt.write_text('MODULE_A\tSynthetic module; test only\t'+'\t'.join(f'GENE{i}' for i in range(10))+'\n'
                   'MODULE_B\tSynthetic module; test only\t'+'\t'.join(f'GENE{i}' for i in range(10,20))+'\n')
    frozen = root/'locked.json'
    create_lock(review_path,gmt,meta_path,frozen,
                'TEST FIXTURE ONLY: program lists were predefined from synthetic generating process.')
    output=root/'output'
    result=run_locked_fit(root,gmt,review_path,frozen,output,offline=True)
    assert len(result) == 16  # 2 modules x 8 independent donors
    assert result.analysis_status.eq('OK').all()
    assert (output/'tables'/'descriptive_group_summary_NO_PVALUES.csv').exists()
    sensitivity=pd.read_csv(output/'tables'/'case_sensitivity_to_control_reference.csv')
    assert len(sensitivity)== 2*4*4  # modules x cases x reference LODO folds
    assert len(list((output/'figures').glob('*.png'))) == 2

    # A literature-QC exclusion may leave 3 controls. Case fits must still run;
    # control LODO must be explicitly skipped instead of crashing or pooling donors.
    reviewed_three = pd.DataFrame(reviews)
    excluded = reviewed_three.donor_id.eq('N4')
    reviewed_three.loc[excluded, 'include'] = '0'
    reviewed_three.loc[excluded, 'exclusion_reason'] = 'synthetic QC exclusion'
    three_review = meta_dir / 'THREE_CONTROLS.csv'
    reviewed_three.to_csv(three_review, index=False)
    three_lock = root / 'three_controls_lock.json'
    create_lock(three_review,gmt,meta_path,three_lock,
                'TEST FIXTURE ONLY: locked synthetic panel with an excluded control donor.')
    three_out = root / 'output_three_controls'
    three_result = run_locked_fit(root,gmt,three_review,three_lock,three_out,offline=True)
    assert len(three_result.query('group == "endometriosis" and analysis_status == "OK"')) == 8
    assert three_result.query('group == "control"').analysis_status.str.startswith('SKIPPED').all()
    assert pd.read_csv(three_out/'tables'/'case_sensitivity_to_control_reference.csv').empty
