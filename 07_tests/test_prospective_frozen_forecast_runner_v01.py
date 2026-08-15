import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '02_src' / 'modeling' / 'prospective_frozen_forecast_runner_v01.py'
if not SRC.exists():
    SRC = Path(__file__).with_name('prospective_frozen_forecast_runner_v01.py')
spec = importlib.util.spec_from_file_location('ff', SRC)
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)


def test_soft_expected_no_threshold():
    p = np.array([0.0, 0.2, 0.9])
    q = np.array([10.0, 100.0, 10.0])
    got = ff.soft_expected_demand(p, q)
    assert np.allclose(got, [0.0, 20.0, 9.0])


def test_soft_expected_clips_probability_and_quantity():
    got = ff.soft_expected_demand(np.array([-1, 2, np.nan]), np.array([-5, 3, 4]))
    assert np.allclose(got, [0.0, 3.0, 0.0])


def test_soft_expected_shape_guard():
    with pytest.raises(ValueError):
        ff.soft_expected_demand([0.2, 0.3], [1.0])


def test_validate_pair_panel_cutoff_pass():
    d = pd.DataFrame({'month': ['2026-05-01', '2026-06-01']})
    assert ff.validate_pair_panel_cutoff(d, '2026-06-01') == pd.Timestamp('2026-06-01')


def test_validate_pair_panel_cutoff_rejects_future_month():
    d = pd.DataFrame({'month': ['2026-06-01', '2026-07-01']})
    with pytest.raises(ValueError):
        ff.validate_pair_panel_cutoff(d, '2026-06-01')


def _synthetic_supervised():
    rows=[]
    for h in [1,2,3]:
        for m in pd.date_range('2024-01-01','2026-06-01',freq='MS'):
            rows.append({
                'horizon':h, 'target_month':m, 'target_available':True,
                'known_pair_asof_origin':True, 'current_production_forecast_mask':True,
                ff.PAIR_TARGET: 1.0,
            })
    return pd.DataFrame(rows)


def test_split_fit_calibration_boundaries():
    d = _synthetic_supervised()
    fit, cal, audit = ff.split_fit_calibration(d, 1, '2026-06-01', 3, True)
    assert fit['target_month'].max() == pd.Timestamp('2026-03-01')
    assert cal['target_month'].min() == pd.Timestamp('2026-04-01')
    assert cal['target_month'].max() == pd.Timestamp('2026-06-01')
    assert audit['calibration_start_target_month'] == '2026-04-01'


def test_split_calibration_current_active_filter():
    d = _synthetic_supervised()
    d.loc[(d['horizon']==2) & (d['target_month']==pd.Timestamp('2026-05-01')), 'current_production_forecast_mask'] = False
    _, cal, _ = ff.split_fit_calibration(d, 2, '2026-06-01', 3, True)
    assert not ((cal['target_month']==pd.Timestamp('2026-05-01')) & (~cal['current_production_forecast_mask'])).any()


def test_forecast_summary_preserves_same_pair_universe():
    rows=[]
    for h, tm in [(1,'2026-07-01'),(2,'2026-08-01'),(3,'2026-09-01')]:
        for sku in ['A','B']:
            rows.append({'horizon':h,'target_month':pd.Timestamp(tm),'base_sku':sku,'branch_code':'X','forecast_m2':10.0,'p_positive':0.5,'pred_positive_quantity':20.0})
    s = ff.forecast_summary(pd.DataFrame(rows))
    assert s['n_pairs'].tolist() == [2,2,2]
    assert s['forecast_sum_m2'].tolist() == [20.0,20.0,20.0]
