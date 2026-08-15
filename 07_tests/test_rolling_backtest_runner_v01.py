import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK9 = HERE.parent
for candidate in [HERE, WORK9 / "02_src" / "modeling", WORK9 / "02_src" / "features"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rolling_backtest_runner_v01 import (
    precompute_intermittent_states,
    rolling_origins,
    split_origin_honest,
    select_gate_from_calibration,
    model_stability_summary,
)


def _rolling_contract():
    return {
        "horizons": [1, 2, 3],
        "origin_window": {
            "start": "2025-01-01", "end": "2025-12-01",
            "frozen_test_start": "2026-04-01", "min_origins": 12,
        },
        "calibration": {
            "target_months": 3, "current_active_only": True,
            "min_fit_rows_per_horizon": 1, "min_calibration_rows_per_horizon": 1,
        },
        "behavior": {
            "adi_threshold": 1.32, "cv2_threshold": 0.49,
            "very_sparse_max_positive_count": 1, "min_segment_calibration_rows": 2,
            "regular_experts": ["lightgbm_tweedie", "moving_average_3"],
            "intermittent_experts": ["lightgbm_tweedie", "moving_average_3"],
            "very_sparse_experts": ["lightgbm_tweedie", "moving_average_3"],
        },
    }


def test_rolling_origins_stop_before_frozen_test():
    origins = rolling_origins(_rolling_contract())
    assert len(origins) == 12
    assert origins[0] == pd.Timestamp("2025-01-01")
    assert origins[-1] == pd.Timestamp("2025-12-01")
    assert origins[-1] + pd.offsets.DateOffset(months=3) < pd.Timestamp("2026-04-01")


def test_split_origin_is_label_honest_and_train_keeps_inactive():
    rows = []
    for tm in pd.date_range("2024-06-01", "2025-03-01", freq="MS"):
        for h in [1, 2, 3]:
            fo = tm - pd.offsets.DateOffset(months=h)
            rows.append({
                "forecast_origin": fo, "target_month": tm, "horizon": h,
                "target_available": True, "current_production_forecast_mask": tm.month % 2 == 0,
                "known_pair_asof_origin": True, "target_actual_gross_m2": 10.0,
            })
    df = pd.DataFrame(rows)
    fit, cal, ev, audit = split_origin_honest(df, pd.Timestamp("2025-01-01"), _rolling_contract())
    assert fit["target_month"].max() < pd.Timestamp("2024-11-01")
    assert cal["target_month"].min() >= pd.Timestamp("2024-11-01")
    assert cal["current_production_forecast_mask"].all()
    # Fit deliberately keeps historical rows regardless of current-active mask.
    assert (~fit["current_production_forecast_mask"]).any()
    assert (fit["target_month"] <= pd.Timestamp("2025-01-01")).all()
    assert (cal["target_month"] <= pd.Timestamp("2025-01-01")).all()


def test_precomputed_croston_tsb_match_reference_semantics():
    panel = pd.DataFrame({
        "base_sku": ["A"] * 7,
        "branch_code": ["X"] * 7,
        "month": pd.date_range("2025-01-01", periods=7, freq="MS"),
        "target_available": [True, True, True, True, True, True, True],
        "actual_gross_m2": [0, 0, 10, 0, 0, 20, 0],
    })
    out = precompute_intermittent_states(panel, alpha=0.1, beta=0.1)
    # Direct reference formulas copied here to make test independent of Drive modules.
    y = np.array([0, 0, 10, 0, 0, 20, 0], dtype=float)
    nz = np.flatnonzero(y > 0)
    first = int(nz[0]); z = y[first]; p = first + 1.0; interval = 1.0
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            z = 0.1 * y[t] + 0.9 * z; p = 0.1 * interval + 0.9 * p; interval = 1.0
        else:
            interval += 1.0
    croston = (1 - 0.1 / 2) * z / p
    zt = y[first]; prob = 1 / (first + 1.0)
    for t in range(first + 1, len(y)):
        occ = float(y[t] > 0); prob = prob + 0.1 * (occ - prob)
        if occ: zt = zt + 0.1 * (y[t] - zt)
    tsb = prob * zt
    assert np.isclose(out.iloc[-1]["pred_croston_sba"], croston)
    assert np.isclose(out.iloc[-1]["pred_tsb"], tsb)


def test_unavailable_month_does_not_advance_intermittent_state():
    panel = pd.DataFrame({
        "base_sku": ["A"] * 4, "branch_code": ["X"] * 4,
        "month": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "target_available": [True, True, False, True],
        "actual_gross_m2": [0, 10, 0, 0],
    })
    out = precompute_intermittent_states(panel)
    assert out.iloc[2]["pred_tsb"] == out.iloc[1]["pred_tsb"]


def _fake_deps():
    def behavior_segment(df, contract):
        return pd.Series(["regular"] * len(df), index=df.index)
    def wape(y, p):
        y = np.asarray(y, float); p = np.asarray(p, float)
        return np.abs(y-p).sum() / np.abs(y).sum()
    return {"behavior_segment": behavior_segment, "wape": wape}


def test_gate_uses_calibration_not_evaluation_actuals():
    cal = pd.DataFrame({
        "horizon": [1, 1, 1], "target_actual_gross_m2": [10, 10, 10],
        "pred_lightgbm_tweedie": [10, 10, 10], "pred_moving_average_3": [100, 100, 100],
    })
    ev = pd.DataFrame({
        "horizon": [1, 1], "target_actual_gross_m2": [10, 10],
        "pred_lightgbm_tweedie": [100, 100], "pred_moving_average_3": [10, 10],
    })
    gate, final = select_gate_from_calibration(
        cal, ev,
        {"lightgbm_tweedie": "pred_lightgbm_tweedie", "moving_average_3": "pred_moving_average_3"},
        _rolling_contract(), pd.Timestamp("2025-01-01"), _fake_deps()
    )
    row = gate[(gate.horizon == 1) & (gate.behavior_segment == "regular")].iloc[0]
    assert row.chosen_expert == "lightgbm_tweedie"
    assert np.all(final.to_numpy() == 100)


def test_small_calibration_segment_falls_back_to_calibration_horizon():
    dep = _fake_deps()
    dep["behavior_segment"] = lambda df, c: pd.Series(
        ["regular"] + ["intermittent"] * (len(df)-1), index=df.index
    )
    cal = pd.DataFrame({
        "horizon": [1, 1, 1], "target_actual_gross_m2": [10, 10, 10],
        "pred_lightgbm_tweedie": [100, 100, 100], "pred_moving_average_3": [10, 10, 10],
    })
    ev = cal.iloc[:1].copy()
    gate, _ = select_gate_from_calibration(
        cal, ev,
        {"lightgbm_tweedie": "pred_lightgbm_tweedie", "moving_average_3": "pred_moving_average_3"},
        _rolling_contract(), pd.Timestamp("2025-01-01"), dep
    )
    row = gate[(gate.horizon == 1) & (gate.behavior_segment == "regular")].iloc[0]
    assert row.selection_scope == "CALIBRATION_HORIZON_FALLBACK"
    assert row.chosen_expert == "moving_average_3"


def test_model_stability_summary_counts_origin_wins():
    by = pd.DataFrame({
        "forecast_origin": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"]),
        "model": ["a", "b", "a", "b"], "horizon": [np.nan]*4, "segment": [np.nan]*4,
        "wape": [0.2, 0.3, 0.4, 0.1], "bias_ratio": [-0.1, 0.2, -0.2, 0.1],
    })
    cum = pd.DataFrame({
        "model": ["a", "a", "b", "b"], "level": ["PAIR", "BASE_SKU", "PAIR", "BASE_SKU"],
        "wape_3m": [0.2, 0.1, 0.3, 0.2],
    })
    s = model_stability_summary(by, cum, {"a":"x","b":"y"})
    assert int(s.loc[s.model.eq("a"), "origin_wins"].iloc[0]) == 1
    assert int(s.loc[s.model.eq("b"), "origin_wins"].iloc[0]) == 1
