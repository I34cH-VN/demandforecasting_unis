import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "02_src" / "features"
if not MODULE_DIR.exists():
    MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
import feature_builder_v013 as f


def _pair_fixture():
    months = pd.date_range("2025-08-01", "2026-03-01", freq="MS")
    rows = []
    # Two branches for the same Base SKU to test cross-branch aggregation.
    vals = {
        "001": [10.0, np.nan, np.nan, 0.0, 20.0, 30.0, np.nan, 40.0],
        "002": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    }
    neg_only_month = pd.Timestamp("2025-10-01")
    for branch, seq in vals.items():
        first_obs = pd.Timestamp("2025-08-01")
        seen_pos = None
        last_pos = None
        obs_count = 0
        pos_count = 0
        for m, v in zip(months, seq):
            actual_observed = not pd.isna(v)
            actual_negative_only = branch == "001" and m == neg_only_month
            if actual_negative_only:
                actual_observed = True
                actual_gross = np.nan
                target_available = False
                actual_positive = False
                zero_semantics = "NEGATIVE_ONLY_GROSS_ZERO"
            elif actual_observed:
                actual_gross = float(v)
                target_available = True
                actual_positive = v > 0
                zero_semantics = "OBSERVED_ZERO" if v == 0 else "OBSERVED_NONZERO_OR_MIXED"
            else:
                actual_gross = np.nan
                target_available = False
                actual_positive = False
                zero_semantics = "MISSING_UNKNOWN"
            if actual_observed:
                obs_count += 1
            if actual_positive:
                pos_count += 1
                if seen_pos is None:
                    seen_pos = m
                last_pos = m
            rows.append({
                "base_sku": "01.L1.3060.X",
                "branch_code": branch,
                "month": m,
                "actual_gross_m2": actual_gross,
                "actual_observed": actual_observed,
                "actual_positive": actual_positive,
                "actual_negative_only": actual_negative_only,
                "target_available": target_available,
                "zero_semantics": zero_semantics,
                "first_observed_month": first_obs,
                "first_positive_month_to_date": seen_pos,
                "months_since_first_positive": ((m.year-seen_pos.year)*12 + m.month-seen_pos.month) if seen_pos is not None else np.nan,
                "months_since_last_positive": ((m.year-last_pos.year)*12 + m.month-last_pos.month) if last_pos is not None else np.nan,
                "observed_month_count_to_date": obs_count,
                "positive_month_count_to_date": pos_count,
                "tile_size_code": "3060",
                "base_group1": "01", "base_group2": "L1", "base_group3": "3060", "base_group4": "X",
                "brand": "UNIS", "product_group": "GACH", "price_group": "P1", "factory_code": "F1", "pull_source": "P",
                "region": "R", "branch_brand": "UNIS",
                "base_current_active": True, "branch_current_active": True,
            })
    return pd.DataFrame(rows)


def _branch_fixture():
    months = pd.date_range("2025-08-01", "2026-03-01", freq="MS")
    vals = [15.0, 6.0, 7.0, 8.0, 29.0, 40.0, np.nan, 52.0]
    rows = []
    obs_count = 0
    pos_count = 0
    for m, v in zip(months, vals):
        obs = not pd.isna(v)
        pos = obs and v > 0
        if obs: obs_count += 1
        if pos: pos_count += 1
        rows.append({
            "branch_code": "001", "month": m,
            "branch_gross_m2": v, "branch_observed": obs, "branch_positive": pos,
            "branch_first_observed_month": pd.Timestamp("2025-08-01"),
            "branch_observed_month_count_to_date": obs_count,
            "branch_positive_month_count_to_date": pos_count,
            "observed_pair_count": 2 if obs else np.nan,
            "positive_pair_count": 2 if pos else np.nan,
            "explicit_zero_pair_count": 0 if obs else np.nan,
            "negative_only_pair_count": 0 if obs else np.nan,
            "new_positive_pair_count": 0 if obs else np.nan,
            "reactivated_pair_count": 0 if obs else np.nan,
            "pair_top1_share": 0.7 if pos else np.nan,
            "pair_top5_share": 1.0 if pos else np.nan,
            "pair_top10_share": 1.0 if pos else np.nan,
            "pair_hhi": 0.58 if pos else np.nan,
            "region": "R", "branch_brand": "UNIS", "branch_current_active": True,
        })
    return pd.DataFrame(rows)


def test_pair_lag_1_is_origin_value_not_future_target():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    dec = origin[(origin.branch_code == "001") & (origin.month == pd.Timestamp("2025-12-01"))].iloc[0]
    assert dec.pair_lag_1 == 20.0
    dev = f.expand_pair_development_panel(origin, pair)
    jan_h1 = dev[(dev.branch_code == "001") & (dev.forecast_origin == pd.Timestamp("2025-12-01")) & (dev.horizon == 1)].iloc[0]
    assert jan_h1.pair_lag_1 == 20.0
    assert jan_h1.target_actual_gross_m2 == 30.0


def test_missing_unknown_not_filled_with_zero_in_lag():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    sep = origin[(origin.branch_code == "001") & (origin.month == pd.Timestamp("2025-09-01"))].iloc[0]
    assert pd.isna(sep.pair_lag_1)



def test_pair_peak_is_carried_forward_after_positive():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    # Branch 001: Aug=10 positive, Sep is MISSING_UNKNOWN. The historical peak
    # known at the Sep origin must remain 10 rather than becoming NaN.
    sep = origin[(origin.branch_code == "001") & (origin.month == pd.Timestamp("2025-09-01"))].iloc[0]
    assert sep.pair_positive_count_to_origin == 1
    assert sep.pair_peak_positive_m2_to_origin == 10.0
    assert sep.pair_peak_share_to_origin == 1.0

def test_negative_only_target_is_not_trainable():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    dev = f.expand_pair_development_panel(origin, pair)
    row = dev[(dev.branch_code == "001") & (dev.target_month == pd.Timestamp("2025-10-01")) & (dev.horizon == 1)].iloc[0]
    assert bool(row.target_actual_negative_only)
    assert pd.isna(row.target_actual_gross_m2)
    assert not bool(row.historical_train_mask)


def test_official_validation_is_locked_and_frozen_test_absent():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    dev = f.expand_pair_development_panel(origin, pair)
    val = dev[dev.split_role == "OFFICIAL_VALIDATION"]
    assert set(val.horizon.unique()) == {1, 2, 3}
    assert val.forecast_origin.eq(pd.Timestamp("2025-12-01")).all()
    assert dev.target_month.max() == pd.Timestamp("2026-03-01")
    assert not dev.target_month.ge(pd.Timestamp("2026-04-01")).any()


def test_cross_branch_features_are_origin_safe():
    pair = _pair_fixture()
    origin = f.build_pair_origin_features(pair)
    aug = origin[(origin.branch_code == "001") & (origin.month == pd.Timestamp("2025-08-01"))].iloc[0]
    assert aug.sku_known_branch_count_to_origin == 2
    assert aug.sku_global_lag_1 == 15.0
    assert aug.cross_branch_penetration_to_origin == 1.0


def test_branch_feature_panel_split_and_lag():
    branch = _branch_fixture()
    origin = f.build_branch_origin_features(branch)
    dec = origin[origin.month == pd.Timestamp("2025-12-01")].iloc[0]
    assert dec.branch_lag_1 == 29.0
    dev = f.expand_branch_development_panel(origin, branch)
    val = dev[dev.split_role == "OFFICIAL_VALIDATION"]
    assert set(val.horizon.unique()) == {1, 2, 3}
    assert dev.target_month.max() == pd.Timestamp("2026-03-01")


def test_feature_inventory_blocks_snapshot_metadata():
    pair = _pair_fixture()
    branch = _branch_fixture()
    po = f.build_pair_origin_features(pair)
    bo = f.build_branch_origin_features(branch)
    pdv = f.expand_pair_development_panel(po, pair)
    bdv = f.expand_branch_development_panel(bo, branch)
    inv = f.build_feature_inventory(pdv, bdv)
    blocked = inv[inv.feature_name.isin(["brand_snapshot", "region_snapshot", "current_production_forecast_mask"])]
    assert not blocked.initial_model_candidate.any()


def test_stage_validation_passes():
    pair = _pair_fixture()
    branch = _branch_fixture()
    po = f.build_pair_origin_features(pair)
    bo = f.build_branch_origin_features(branch)
    pdv = f.expand_pair_development_panel(po, pair)
    bdv = f.expand_branch_development_panel(bo, branch)
    report = f.validate_feature_stage(pdv, bdv)
    assert report["status"] == "PASS", report
    assert report["frozen_test_touched"] is False


def test_tet_calendar_flags_move_between_january_and_february():
    x = pd.DataFrame({
        "target_month": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-03-01",
            "2025-01-01", "2025-02-01",
            "2026-01-01", "2026-02-01", "2026-03-01",
        ]),
        "horizon": [1]*8,
    })
    y = f._add_target_calendar(x)
    by_month = y.set_index(y["target_month"].dt.strftime("%Y-%m"))
    assert bool(by_month.loc["2024-02", "target_is_tet_month"])
    assert bool(by_month.loc["2025-01", "target_is_tet_month"])
    assert bool(by_month.loc["2026-02", "target_is_tet_month"])
    assert bool(by_month.loc["2026-01", "target_is_pre_tet_month"])
    assert bool(by_month.loc["2026-03", "target_is_post_tet_month"])
    assert not bool(by_month.loc["2025-02", "target_is_tet_month"])


def test_validation_tet_sequence_is_origin_safe_known_future_calendar():
    pair = _pair_fixture()
    branch = _branch_fixture()
    po = f.build_pair_origin_features(pair)
    bo = f.build_branch_origin_features(branch)
    pdv = f.expand_pair_development_panel(po, pair)
    bdv = f.expand_branch_development_panel(bo, branch)
    pv = pdv[pdv.split_role.eq("OFFICIAL_VALIDATION")]
    bv = bdv[bdv.split_role.eq("OFFICIAL_VALIDATION")]
    for v in (pv, bv):
        assert v.loc[v.target_month.eq(pd.Timestamp("2026-01-01")), "target_is_pre_tet_month"].all()
        assert v.loc[v.target_month.eq(pd.Timestamp("2026-02-01")), "target_is_tet_month"].all()
        assert v.loc[v.target_month.eq(pd.Timestamp("2026-03-01")), "target_is_post_tet_month"].all()


def test_vietnam_lunar_conversion_matches_official_anchor_dates():
    assert f.solar_to_lunar_vn("2024-02-10")[:3] == (1, 1, 2024)
    assert f.solar_to_lunar_vn("2025-01-29")[:3] == (1, 1, 2025)
    assert f.solar_to_lunar_vn("2026-02-17")[:3] == (1, 1, 2026)
    assert f.solar_to_lunar_vn("2024-04-18")[:3] == (10, 3, 2024)
    assert f.solar_to_lunar_vn("2025-04-07")[:3] == (10, 3, 2025)
    assert f.solar_to_lunar_vn("2026-04-26")[:3] == (10, 3, 2026)


def test_calendar_v013_tet_day_position_and_lunar_month_features():
    cal = f.build_target_calendar_table(pd.to_datetime([
        "2024-02-01", "2025-01-01", "2026-02-01",
    ])).set_index("target_month")
    assert cal.loc[pd.Timestamp("2024-02-01"), "target_tet_day_of_month"] == 10
    assert cal.loc[pd.Timestamp("2025-01-01"), "target_tet_day_of_month"] == 29
    assert cal.loc[pd.Timestamp("2026-02-01"), "target_tet_day_of_month"] == 17
    assert cal["target_lunar_month_mid"].between(1, 12).all()
    assert np.isfinite(cal["target_lunar_month_sin"]).all()
    assert np.isfinite(cal["target_lunar_month_cos"]).all()


def test_calendar_v013_public_holiday_nominal_days_and_working_proxy():
    cal = f.build_target_calendar_table(pd.to_datetime([
        "2024-04-01",  # Hùng Kings + 30/4
        "2025-01-01",  # New Year + Tết
        "2025-09-01",  # National Day nominal 2 days
        "2026-05-01",  # 1/5
    ])).set_index("target_month")

    apr24 = cal.loc[pd.Timestamp("2024-04-01")]
    assert apr24.target_public_holiday_event_count == 2
    assert apr24.target_statutory_holiday_nominal_days == 2
    assert apr24.target_weekday_count == 22
    assert apr24.target_working_days_proxy == 20

    jan25 = cal.loc[pd.Timestamp("2025-01-01")]
    assert jan25.target_public_holiday_event_count == 2
    assert jan25.target_statutory_holiday_nominal_days == 6
    assert jan25.target_weekday_count == 23
    assert jan25.target_working_days_proxy == 17

    sep25 = cal.loc[pd.Timestamp("2025-09-01")]
    assert sep25.target_statutory_holiday_nominal_days == 2
    assert sep25.target_working_days_proxy == 20

    may26 = cal.loc[pd.Timestamp("2026-05-01")]
    assert may26.target_statutory_holiday_nominal_days == 1
    assert may26.target_working_days_proxy == 20


def test_calendar_v013_tet_distance_is_known_future_and_nonnegative():
    cal = f.build_target_calendar_table(pd.to_datetime([
        "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01",
    ])).set_index("target_month")
    assert (cal["target_days_to_next_tet_from_midmonth"] >= 0).all()
    assert (cal["target_days_since_prev_tet_from_midmonth"] >= 0).all()
    # 15-Feb-2026 is two days before mùng 1 Tết (17-Feb-2026).
    assert cal.loc[pd.Timestamp("2026-02-01"), "target_days_to_next_tet_from_midmonth"] == 2


def test_calendar_v013_features_are_model_candidates_for_both_tracks():
    pair = _pair_fixture()
    branch = _branch_fixture()
    po = f.build_pair_origin_features(pair)
    bo = f.build_branch_origin_features(branch)
    pdv = f.expand_pair_development_panel(po, pair)
    bdv = f.expand_branch_development_panel(bo, branch)
    inv = f.build_feature_inventory(pdv, bdv)
    required = {
        "target_weekday_count",
        "target_public_holiday_event_count",
        "target_statutory_holiday_nominal_days",
        "target_working_days_proxy",
        "target_lunar_month_mid",
        "target_lunar_month_mid_is_leap",
        "target_lunar_month_sin",
        "target_lunar_month_cos",
        "target_tet_day_of_month",
        "target_days_to_next_tet_from_midmonth",
        "target_days_since_prev_tet_from_midmonth",
    }
    for track in ("PAIR", "BRANCH"):
        rows = inv[(inv.track == track) & (inv.feature_name.isin(required))]
        assert set(rows.feature_name) == required
        assert rows.initial_model_candidate.all()
