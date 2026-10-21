import numpy as np
import pandas as pd
import pytest

from paper7.model import (Reference, make_reference, onset_basis, fit_amplitude,
                          fit_delayed, compare_models, design_identifiability,
                          suggest_sampling_times)
from paper7.simulation import synthetic_reference, run_simulation


def test_reference_requires_day0_and_four_points():
    with pytest.raises(ValueError, match="day-0"):
        Reference(np.array([1.,2.,3.,4.]), np.arange(4.))
    with pytest.raises(ValueError, match="strictly increasing"):
        Reference(np.array([0.,1.,1.,2.]), np.arange(4.))


def test_onset_prevents_extrapolation():
    ref = synthetic_reference()
    with pytest.raises(ValueError, match="outside"):
        onset_basis(ref, [9], 1.)
    with pytest.raises(ValueError, match="delays"):
        onset_basis(ref, [2], -0.1)


def test_amp_only_and_delay_fit_on_noiseless_synthetic():
    ref = synthetic_reference()
    t = ref.days
    y = 0.33 + .61 * onset_basis(ref, t, 0)
    m = fit_amplitude(ref, t, y)
    assert abs(m['amplitude'] - 0.61) < 1e-8
    assert m['sse'] < 1e-8
    mdl = fit_delayed(ref, t, y)
    assert abs(mdl['delay_days']) < 0.15
    y_delay = .22 + 1.0 * onset_basis(ref, t, 1.25)
    mdelay = fit_delayed(ref, t, y_delay)
    assert abs(mdelay['delay_days'] - 1.25) < 0.1
    assert mdelay['sse'] < 1e-5


def test_loto_assesses_timepoint_not_independent_donor():
    ref = synthetic_reference()
    y = .1 + .7 * onset_basis(ref, ref.days, 0.)
    scores = compare_models(ref, ref.days, y)
    assert scores['n_timepoints'] == 5
    assert scores['loto_amp_rmse'] < 1e-8


def test_reference_donor_weighting_not_replicate_weighting():
    records = []
    for i in range(3):
        for t in [0., 1., 2., 3., 4.]:
            records.append(dict(donor_id=str(i), days=t, score=t+i))
    records.append(dict(donor_id='0', days=2., score=2.))  # technical duplicate
    ref = make_reference(pd.DataFrame(records))
    np.testing.assert_allclose(ref.values, [1.,2.,3.,4.,5.])


def test_monotonic_approx_exponential_has_collinear_jacobian():
    ref = Reference(np.array([0.,1.,2.,3.,4.,5.]),
                    np.exp(np.arange(6.)/2))
    metrics = design_identifiability(ref, ref.days)
    assert np.isfinite(metrics['scaled_condition'])
    assert metrics['min_singular'] >= 0


def test_straight_flat_reference_flags_problem():
    ref = Reference(np.array([0.,1.,2.,3.,4.]), np.zeros(5))
    report = design_identifiability(ref, ref.days)
    assert not report['rank_two']
    assert report['scaled_condition'] == float('inf')


def test_designs_contain_endpoints():
    ref = synthetic_reference()
    design = suggest_sampling_times(ref, np.arange(0,8.01,1.0), n_samples=5, top_k=4)
    assert len(design) == 4
    assert all(row.startswith('0,') and row.endswith(',8') for row in design.days)


def test_synthetic_experiment_is_reproducible(tmp_path):
    first = run_simulation(tmp_path/'a', reps=2, seed=1)
    second = run_simulation(tmp_path/'b', reps=2, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path/'a'/'SYNTHETIC_NOT_GEO.txt').exists()
