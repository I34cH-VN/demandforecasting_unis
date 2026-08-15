import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "02_src" / "modeling"
assert MODEL_DIR.exists(), f"Missing modeling source dir: {MODEL_DIR}"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runner_v02 import (
    behavior_segment, build_complete_3m_pair_windows, calibrate_hurdle_threshold,
    catboost_target_scale, croston_sba_forecast, cumulative_3m_scoreboard,
    forecast_revision_scoreboard, split_universes, tsb_forecast,
    validate_feature_safety, wape, zero_false_positive_rate, _select_gate,
)


def _contract():
    return {
        "feature_safety": {
            "forbidden_exact": ["target_actual_gross_m2", "target_available"],
            "forbidden_contains": ["current_active", "current_status"],
        },
        "catboost_target_scaling": {"enabled": True, "method": "max_train_to_one", "min_scale": 1.0},
        "hurdle": {
            "threshold_calibration": {
                "enabled": True,
                "grid": [0.0, 0.5, 0.8],
                "wape_tie_tolerance": 1e-9,
            }
        },
        "behavior": {
            "adi_threshold": 1.32,
            "cv2_threshold": 0.49,
            "very_sparse_max_positive_count": 1,
            "min_segment_validation_rows": 3,
            "regular_experts": ["lightgbm_tweedie", "naive_1"],
            "intermittent_experts": ["lightgbm_tweedie", "naive_1"],
            "very_sparse_experts": ["lightgbm_tweedie", "naive_1"],
        },
    }


def test_wape_zero_error():
    assert wape([1, 2, 3], [1, 2, 3]) == 0.0


def test_zero_false_positive_rate():
    assert zero_false_positive_rate([0, 0, 2], [1, 0, 2]) == 0.5


def test_primary_mask_excludes_inactive_or_unseen():
    df = pd.DataFrame({
        "historical_train_mask": [1, 0, 0, 0],
        "official_validation_mask": [0, 1, 1, 1],
        "target_available": [1, 1, 1, 1],
        "current_production_forecast_mask": [0, 1, 0, 1],
        "known_pair_asof_origin": [1, 1, 1, 0],
        "horizon": [1, 1, 1, 1],
        "target_actual_gross_m2": [10, 20, 30, 40],
    })
    train, primary, secondary = split_universes(df)
    assert len(train) == 1
    assert len(secondary) == 3
    assert len(primary) == 1
    assert primary.iloc[0]["target_actual_gross_m2"] == 20


def test_train_does_not_filter_current_status():
    df = pd.DataFrame({
        "historical_train_mask": [1, 1, 0],
        "official_validation_mask": [0, 0, 1],
        "target_available": [1, 1, 1],
        "current_production_forecast_mask": [0, 1, 1],
        "known_pair_asof_origin": [1, 1, 1],
        "horizon": [1, 1, 1],
        "target_actual_gross_m2": [10, 20, 30],
    })
    train, primary, _ = split_universes(df)
    assert len(train) == 2
    assert len(primary) == 1


def test_forbidden_current_status_feature_rejected():
    try:
        validate_feature_safety(["pair_lag_1", "base_current_active"], _contract())
        assert False
    except ValueError:
        assert True


def test_behavior_regular_requires_both_low():
    df = pd.DataFrame({
        "pair_adi_target_available_to_origin": [1.0],
        "pair_cv2_positive_to_origin": [0.2],
        "pair_positive_count_to_origin": [10],
    })
    assert behavior_segment(df, _contract()).iloc[0] == "regular"


def test_behavior_high_adi_is_intermittent():
    df = pd.DataFrame({
        "pair_adi_target_available_to_origin": [2.0],
        "pair_cv2_positive_to_origin": [0.2],
        "pair_positive_count_to_origin": [4],
    })
    assert behavior_segment(df, _contract()).iloc[0] == "intermittent"


def test_behavior_high_cv2_is_intermittent():
    df = pd.DataFrame({
        "pair_adi_target_available_to_origin": [1.0],
        "pair_cv2_positive_to_origin": [0.8],
        "pair_positive_count_to_origin": [4],
    })
    assert behavior_segment(df, _contract()).iloc[0] == "intermittent"


def test_behavior_very_sparse_when_state_undefined_or_low_count():
    df = pd.DataFrame({
        "pair_adi_target_available_to_origin": [2.0, 1.0],
        "pair_cv2_positive_to_origin": [0.2, np.nan],
        "pair_positive_count_to_origin": [1, 5],
    })
    out = behavior_segment(df, _contract()).tolist()
    assert out == ["very_sparse", "very_sparse"]


def test_catboost_scale_maps_max_train_to_one():
    scale = catboost_target_scale([0, 50, 200], _contract())
    assert scale == 200.0
    assert max(np.asarray([0, 50, 200], dtype=float) / scale) == 1.0


def test_hurdle_threshold_calibration_can_emit_zero_forecast():
    y = np.array([0.0, 100.0])
    p = np.array([0.2, 0.9])
    pos = np.array([100.0, 100.0])
    threshold, pred, report = calibrate_hurdle_threshold(y, p, pos, _contract())
    assert threshold >= 0.5
    assert pred[0] == 0.0
    assert pred[1] > 0
    assert len(report) == 3


def test_small_segment_gate_uses_horizon_best_allowed_expert():
    # Segment has only 1 row (< min=3), while whole horizon clearly favors naive_1.
    df = pd.DataFrame({
        "horizon": [1, 1, 1, 1],
        "target_actual_gross_m2": [10, 10, 10, 10],
        "pair_adi_target_available_to_origin": [2.0, 1.0, 1.0, 1.0],
        "pair_cv2_positive_to_origin": [0.2, 0.2, 0.2, 0.2],
        "pair_positive_count_to_origin": [4, 4, 4, 4],
        "pred_lightgbm_tweedie": [100, 100, 100, 100],
        "pred_naive_1": [10, 10, 10, 10],
    })
    gate, final = _select_gate(df, {"lightgbm_tweedie": "pred_lightgbm_tweedie", "naive_1": "pred_naive_1"}, _contract())
    row = gate.loc[gate["behavior_segment"].eq("intermittent")].iloc[0]
    assert row["selection_scope"] == "HORIZON_PRIMARY_FALLBACK"
    assert row["chosen_expert"] == "naive_1"
    assert final.iloc[0] == 10


def test_complete_3m_requires_h1_h2_h3():
    df = pd.DataFrame({
        "base_sku": ["A", "A", "A", "B", "B"],
        "branch_code": ["X", "X", "X", "X", "X"],
        "forecast_origin": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-01"] * 2),
        "horizon": [1, 2, 3, 1, 2],
        "target_actual_gross_m2": [10, 20, 30, 5, 6],
        "pred_m": [11, 19, 31, 5, 7],
    })
    pair3, coverage = build_complete_3m_pair_windows(df, {"m": "pred_m"})
    assert len(pair3) == 1
    assert pair3.iloc[0]["actual_3m_m2"] == 60
    assert pair3.iloc[0]["pred_m_3m"] == 61
    assert coverage["pair_origin_complete_h1_h2_h3"] == 1
    assert coverage["pair_origin_incomplete"] == 1


def test_cumulative_3m_scores_pair_and_sku():
    pair3 = pd.DataFrame({
        "base_sku": ["A", "A"],
        "branch_code": ["X", "Y"],
        "forecast_origin": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "actual_3m_m2": [60.0, 40.0],
        "pred_m_3m": [61.0, 39.0],
    })
    score = cumulative_3m_scoreboard(pair3, {"m": "pred_m"})
    assert set(score["level"]) == {"PAIR", "BASE_SKU", "BRANCH", "PORTFOLIO"}
    sku = score[(score.model == "m") & (score.level == "BASE_SKU")].iloc[0]
    assert sku["wape_3m"] == 0.0


def test_forecast_revision_h2_to_h1():
    df = pd.DataFrame({
        "base_sku": ["A", "A"],
        "branch_code": ["X", "X"],
        "forecast_origin": pd.to_datetime(["2026-03-01", "2026-04-01"]),
        "target_month": pd.to_datetime(["2026-05-01", "2026-05-01"]),
        "horizon": [2, 1],
        "target_actual_gross_m2": [100, 100],
        "pred_m": [120, 105],
    })
    score, detail = forecast_revision_scoreboard(df, {"m": "pred_m"})
    pair = score[(score.model == "m") & (score.level == "PAIR") & (score.transition == "H2_TO_H1")].iloc[0]
    assert pair["revision_mae_m2"] == 15.0
    assert pair["signed_revision_m2"] == -15.0
    assert len(detail) == 1



def test_forecast_revision_empty_has_stable_schema():
    df = pd.DataFrame({
        "base_sku": ["A", "A"],
        "branch_code": ["X", "X"],
        "forecast_origin": pd.to_datetime(["2026-03-01", "2026-03-01"]),
        "target_month": pd.to_datetime(["2026-04-01", "2026-05-01"]),
        "horizon": [1, 2],
        "target_actual_gross_m2": [100, 100],
        "pred_m": [90, 95],
    })
    score, detail = forecast_revision_scoreboard(df, {"m": "pred_m"})
    assert score.empty
    assert detail.empty
    assert list(score.columns) == [
        "model", "level", "transition", "n_units", "revision_mae_m2",
        "revision_ratio_vs_old_forecast", "signed_revision_m2",
    ]
    assert "transition" in detail.columns

def test_croston_sba_nonnegative():
    pred = croston_sba_forecast([0, 0, 10, 0, 0, 20, 0], alpha=0.1)
    assert np.isfinite(pred) and pred >= 0


def test_tsb_decays_after_zeros():
    before = tsb_forecast([0, 10], alpha=0.1, beta=0.1)
    after = tsb_forecast([0, 10, 0, 0, 0, 0], alpha=0.1, beta=0.1)
    assert after < before
