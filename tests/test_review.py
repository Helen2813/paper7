from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from paper7.data import prepare_review, validated_review, create_lock, verify_lock, score_all_samples


def build_review():
    items = []
    for group, ids in [('control', ['C1','C2','C3','C4']),('endometriosis', ['E1','E2','E3','E4'])]:
        for d in ids:
            for day in [0,2,4,6,8]:
                items.append(dict(gse='GSE243039',gsm=f'{d}_D{day}',include='1',reviewed='1',
                                  donor_id=d,group=group,days=str(day),treatment='EPC',
                                  exclusion_reason='',review_provenance='Validated synthetic fixture'))
    return pd.DataFrame(items)


def test_review_template_does_not_auto_verify():
    metadata = pd.DataFrame({'gse':['GSE243039'], 'gsm':['GSM01'],
                             'title':['Endometriosis stroma Day2'], 'group':['endometriosis'],
                             'donor_local':['rep1'], 'time_hours':[48.]})
    r = prepare_review(metadata)
    assert r.group.iloc[0] == '' and r.donor_id.iloc[0] == '' and r.reviewed.iloc[0] == ''
    assert r.suggested_group_UNVERIFIED.iloc[0] == 'endometriosis'


def test_requires_donor_verification_and_all_groups(tmp_path):
    r=build_review(); r.at[0,'reviewed']='0'
    f=tmp_path/'review.csv'; r.to_csv(f,index=False)
    with pytest.raises(ValueError,match='reviewed=1'):
        validated_review(f)
    r=build_review(); r.loc[r.donor_id=='C1','donor_id']='E1'
    r.to_csv(f,index=False)
    with pytest.raises(ValueError,match='overlap'):
        validated_review(f)


def test_curated_freeze_and_integrity(tmp_path):
    meta=pd.DataFrame({'gsm':build_review().gsm,'gse':'GSE243039','title':'SYNTHETIC'})
    m=tmp_path/'meta.csv';meta.to_csv(m,index=False)
    r=tmp_path/'r.csv';build_review().to_csv(r,index=False)
    g=tmp_path/'curated.gmt';g.write_text('A\tExternal curated rationale\tG1\tG2\tG3\n')
    path=tmp_path/'frozen.json'
    create_lock(r,g,m,path,'Synthetic fixture only; independently defined test panel.')
    verify_lock(path,r,g,m)
    g.write_text('A\tchanged\tG1\tG2\tG3\n')
    with pytest.raises(ValueError,match='changed after freeze'):
        verify_lock(path,r,g,m)


def test_control_normalization_ignores_changes_to_cases():
    review = build_review()
    rng=np.random.default_rng(0)
    expr=pd.DataFrame(rng.uniform(20,100,(4,len(review))),
                      index=['G1','G2','G3','G4'], columns=review.gsm)
    programs={'PROG':['G1','G2','G3','G4']}
    first,cov,_=score_all_samples(expr,review,programs)
    expr2=expr.copy()
    expr2.loc[:,review.loc[review.group=='endometriosis','gsm']]*=1000
    second,cov2,_=score_all_samples(expr2,review,programs)
    ctrl=set(review.loc[review.group=='control','gsm'])
    a=first[first.gsm.isin(ctrl)].sort_values('gsm').score.to_numpy()
    b=second[second.gsm.isin(ctrl)].sort_values('gsm').score.to_numpy()
    np.testing.assert_allclose(a,b)
    pd.testing.assert_frame_equal(cov,cov2)


def test_only_verified_controls_allowed_for_training():
    review=build_review()
    expr=pd.DataFrame(np.arange(4*40).reshape(4,40)+1.0,index=['G1','G2','G3','G4'],columns=review.gsm)
    with pytest.raises(ValueError, match='exclusively'):
        score_all_samples(expr,review,{'p':['G1','G2','G3']},train_control_gsms=['E1_D0'])
