import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNNER = ROOT / "02_src" / "modeling" / "underforecast_diagnosis_runner_v01.py"
assert RUNNER.exists(), f"Missing diagnosis runner: {RUNNER}"
SPEC = importlib.util.spec_from_file_location("diag", RUNNER)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)

CFG = {
    "event_rules": {
        "medium_gap_months": 3,
        "long_gap_months": 6,
        "moderate_spike_ratio": 2.0,
        "extreme_spike_ratio": 3.0,
        "min_spike_actual_m2": 1.0,
    }
}

def base_df():
    return pd.DataFrame({
        "base_sku": ["A"]*8,
        "branch_code": ["B"]*8,
        "forecast_origin": pd.to_datetime(["2025-01-01"]*8),
        "target_month": pd.to_datetime(["2025-02-01"]*8),
        "horizon": [1]*8,
        "target_actual_gross_m2": [0,10,20,30,40,50,300,100],
        "pred_lightgbm_tweedie": [0.2,2,5,10,20,25,100,90],
        "behavior_segment": ["regular"]*8,
        "pair_positive_count_to_origin": [5,0,3,3,3,3,3,3],
        "months_since_last_positive": [0,np.nan,7,4,2,0,0,0],
        "pair_mean_positive_m2_to_origin": [10,10,10,10,10,10,100,70],
    })

def test_event_classification_is_mutually_exclusive():
    got = diag.classify_demand_event(base_df(), CFG).tolist()
    assert got == [
        "actual_zero", "first_positive_known_pair", "reactivation_gap_6plus",
        "reactivation_gap_3_5", "reactivation_gap_1_2", "spike_3x_plus",
        "spike_3x_plus", "ongoing_positive"
    ]

def test_round_half_up_nonnegative():
    s = pd.Series([0.1, 0.49, 0.5, 1.49, 1.5])
    assert diag._round_nearest_1(s).tolist() == [0,0,1,1,2]

def test_ceil_never_reduces_nonnegative_prediction():
    s = pd.Series([0,0.01,0.9,1.0,1.2])
    c = diag._ceil_1(s)
    assert np.all(c >= s)

def test_rounding_does_not_touch_actual():
    d = base_df()
    actual_before = d[diag.TARGET].copy()
    _ = diag.compare_pair_month_rounding(d)
    pd.testing.assert_series_equal(d[diag.TARGET], actual_before)

def test_complete_3m_requires_all_horizons():
    d = pd.DataFrame({
        "base_sku":["A"]*5,
        "branch_code":["B"]*5,
        "forecast_origin":pd.to_datetime(["2025-01-01"]*3+["2025-02-01"]*2),
        "horizon":[1,2,3,1,2],
        "target_actual_gross_m2":[1,2,3,1,2],
        "pred_lightgbm_tweedie":[1,2,3,1,2],
    })
    got = diag._complete_3m(d)
    assert len(got) == 3
    assert got["forecast_origin"].nunique() == 1

def test_diagnosis_shares_sum_sensibly():
    d = base_df()
    out = diag.build_underforecast_diagnosis(d, CFG)
    events = out[out.dimension.eq("DEMAND_EVENT")]
    assert abs(events["actual_sum_m2"].sum() - d[diag.TARGET].sum()) < 1e-9
    assert events["share_of_total_underforecast"].sum() > 0.99

def test_granularity_detects_integer_share():
    d = pd.DataFrame({diag.TARGET:[1.0,2.0,1.5,2.25,0.0]})
    summary, freq = diag.actual_m2_granularity(d)
    assert summary.loc[0,"share_exact_integer_m2"] == 0.5
    assert summary.loc[0,"share_on_0_01_m2_grid"] == 1.0
    assert not freq.empty
