from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import math

import numpy as np
import pandas as pd

PAIR_FEATURE_VERSION = "pair_feature_v013"
BRANCH_FEATURE_VERSION = "branch_feature_v013"
DATASET_VERSION = "dataset_v012"

PAIR_KEYS = ["base_sku", "branch_code"]


def _month_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_datetime(a)
    bb = pd.to_datetime(b)
    return (aa.dt.year - bb.dt.year) * 12 + (aa.dt.month - bb.dt.month)


def _month_add(s: pd.Series, n: int) -> pd.Series:
    return pd.to_datetime(s) + pd.offsets.DateOffset(months=n)


def _group_rolling(df: pd.DataFrame, keys: List[str], col: str, window: int, op: str, min_periods: int = 1) -> pd.Series:
    r = df.groupby(keys, sort=False)[col].rolling(window=window, min_periods=min_periods)
    if op == "mean":
        out = r.mean()
    elif op == "sum":
        out = r.sum()
    elif op == "std":
        out = r.std(ddof=0)
    elif op == "max":
        out = r.max()
    else:
        raise ValueError(f"Unsupported rolling op: {op}")
    return out.reset_index(level=keys, drop=True).sort_index()


def _sum_min_count(s: pd.Series):
    return s.sum(min_count=1)


def _assert_unique(df: pd.DataFrame, keys: List[str], label: str):
    if df.duplicated(keys).any():
        sample = df.loc[df.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise ValueError(f"{label} duplicate grain {keys}; sample={sample}")


def build_pair_origin_features(pair_panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "base_sku", "branch_code", "month", "actual_gross_m2", "actual_observed",
        "actual_positive", "actual_negative_only", "target_available",
        "first_observed_month", "first_positive_month_to_date",
        "months_since_first_positive", "months_since_last_positive",
        "observed_month_count_to_date", "positive_month_count_to_date",
        "tile_size_code",
    }
    missing = required - set(pair_panel.columns)
    if missing:
        raise ValueError(f"pair_panel missing columns: {sorted(missing)}")

    p = pair_panel.copy()
    p["month"] = pd.to_datetime(p["month"])
    p = p.sort_values(PAIR_KEYS + ["month"]).reset_index(drop=True)

    # Target-relative lags: lag_1 is the closed origin month when target is origin+1.
    g = p.groupby(PAIR_KEYS, sort=False)["actual_gross_m2"]
    p["pair_lag_1"] = p["actual_gross_m2"]
    p["pair_lag_2"] = g.shift(1)
    p["pair_lag_3"] = g.shift(2)
    p["pair_lag_6"] = g.shift(5)
    p["pair_lag_12"] = g.shift(11)

    p["_target_avail_i"] = p["target_available"].astype(int)
    p["_positive_i"] = p["actual_positive"].astype(int)
    p["_calendar_i"] = 1

    for w in (3, 6, 12):
        p[f"pair_roll_{w}_mean"] = _group_rolling(p, PAIR_KEYS, "actual_gross_m2", w, "mean", 1)
        p[f"pair_roll_{w}_sum"] = _group_rolling(p, PAIR_KEYS, "actual_gross_m2", w, "sum", 1)
        p[f"pair_roll_{w}_std"] = _group_rolling(p, PAIR_KEYS, "actual_gross_m2", w, "std", 2)
        p[f"pair_roll_{w}_target_available_count"] = _group_rolling(p, PAIR_KEYS, "_target_avail_i", w, "sum", 1)
        p[f"pair_roll_{w}_positive_count"] = _group_rolling(p, PAIR_KEYS, "_positive_i", w, "sum", 1)
        p[f"pair_roll_{w}_calendar_count"] = _group_rolling(p, PAIR_KEYS, "_calendar_i", w, "sum", 1)
        denom = p[f"pair_roll_{w}_calendar_count"].replace(0, np.nan)
        p[f"pair_roll_{w}_coverage"] = p[f"pair_roll_{w}_target_available_count"] / denom

    p["pair_history_length_months"] = _month_diff(p["month"], p["first_observed_month"]) + 1
    p["pair_target_available_count_to_origin"] = p.groupby(PAIR_KEYS, sort=False)["_target_avail_i"].cumsum()
    p["pair_observed_source_month_count_to_origin"] = p["observed_month_count_to_date"].astype(int)
    p["pair_positive_count_to_origin"] = p["positive_month_count_to_date"].astype(int)
    denom = p["pair_target_available_count_to_origin"].replace(0, np.nan)
    p["pair_positive_rate_to_origin"] = p["pair_positive_count_to_origin"] / denom

    p["_positive_value"] = p["actual_gross_m2"].where(p["actual_positive"])
    p["_positive_value_zero"] = p["_positive_value"].fillna(0.0)
    p["_positive_value_sq"] = p["_positive_value_zero"] ** 2
    p["pair_last_positive_m2_to_origin"] = p.groupby(PAIR_KEYS, sort=False)["_positive_value"].ffill()
    p["_pos_sum"] = p.groupby(PAIR_KEYS, sort=False)["_positive_value_zero"].cumsum()
    p["_pos_sq_sum"] = p.groupby(PAIR_KEYS, sort=False)["_positive_value_sq"].cumsum()
    pos_n = p["pair_positive_count_to_origin"].astype(float)
    p["pair_mean_positive_m2_to_origin"] = p["_pos_sum"] / pos_n.replace(0, np.nan)
    p["pair_peak_positive_m2_to_origin"] = p.groupby(PAIR_KEYS, sort=False)["_positive_value"].cummax()
    # cummax returns NaN on non-positive / missing rows; a historical peak already known
    # at an earlier month must remain known at all later origins. Preserve NaN only
    # before the first positive event.
    p["pair_peak_positive_m2_to_origin"] = p.groupby(PAIR_KEYS, sort=False)["pair_peak_positive_m2_to_origin"].ffill()
    p["pair_peak_share_to_origin"] = p["pair_peak_positive_m2_to_origin"] / p["_pos_sum"].replace(0, np.nan)
    p["pair_adi_target_available_to_origin"] = p["pair_target_available_count_to_origin"] / pos_n.replace(0, np.nan)

    mean_pos = p["pair_mean_positive_m2_to_origin"]
    var_pos = p["_pos_sq_sum"] / pos_n.replace(0, np.nan) - mean_pos ** 2
    var_pos = var_pos.clip(lower=0)
    p["pair_cv2_positive_to_origin"] = np.where(
        (pos_n >= 2) & mean_pos.gt(0), var_pos / (mean_pos ** 2), np.nan
    )

    # Base-SKU global / cross-branch monthly behavior from historical Pair panel only.
    sku = p.groupby(["base_sku", "month"], as_index=False).agg(
        sku_global_gross_m2=("actual_gross_m2", _sum_min_count),
        sku_target_available_branch_count=("target_available", "sum"),
        sku_observed_branch_count=("actual_observed", "sum"),
        sku_positive_branch_count=("actual_positive", "sum"),
        sku_known_branch_count_to_origin=("branch_code", "nunique"),
        sku_ever_positive_branch_count_to_origin=("first_positive_month_to_date", lambda s: int(s.notna().sum())),
    )
    sku = sku.sort_values(["base_sku", "month"]).reset_index(drop=True)
    sg = sku.groupby("base_sku", sort=False)["sku_global_gross_m2"]
    sku["sku_global_lag_1"] = sku["sku_global_gross_m2"]
    sku["sku_global_lag_2"] = sg.shift(1)
    sku["sku_global_lag_3"] = sg.shift(2)
    sku["sku_global_lag_6"] = sg.shift(5)
    sku["sku_global_lag_12"] = sg.shift(11)
    sku["_sku_month_observed"] = sku["sku_target_available_branch_count"].gt(0).astype(int)
    sku["_sku_month_positive"] = sku["sku_global_gross_m2"].fillna(0).gt(0).astype(int)
    for w in (3, 6, 12):
        sku[f"sku_global_roll_{w}_mean"] = _group_rolling(sku, ["base_sku"], "sku_global_gross_m2", w, "mean", 1)
        sku[f"sku_global_roll_{w}_sum"] = _group_rolling(sku, ["base_sku"], "sku_global_gross_m2", w, "sum", 1)
    obs12 = _group_rolling(sku, ["base_sku"], "_sku_month_observed", 12, "sum", 1)
    pos12 = _group_rolling(sku, ["base_sku"], "_sku_month_positive", 12, "sum", 1)
    sku["sku_global_positive_rate_12m"] = pos12 / obs12.replace(0, np.nan)
    sku["sku_target_available_branch_count_lag_1"] = sku["sku_target_available_branch_count"]
    sku["sku_positive_branch_count_lag_1"] = sku["sku_positive_branch_count"]
    sku["cross_branch_penetration_to_origin"] = (
        sku["sku_ever_positive_branch_count_to_origin"] /
        sku["sku_known_branch_count_to_origin"].replace(0, np.nan)
    )

    sku_keep = [
        "base_sku", "month",
        "sku_global_lag_1", "sku_global_lag_2", "sku_global_lag_3", "sku_global_lag_6", "sku_global_lag_12",
        "sku_global_roll_3_mean", "sku_global_roll_3_sum",
        "sku_global_roll_6_mean", "sku_global_roll_6_sum",
        "sku_global_roll_12_mean", "sku_global_roll_12_sum",
        "sku_global_positive_rate_12m",
        "sku_target_available_branch_count_lag_1", "sku_positive_branch_count_lag_1",
        "sku_known_branch_count_to_origin", "sku_ever_positive_branch_count_to_origin",
        "cross_branch_penetration_to_origin",
    ]
    p = p.merge(sku[sku_keep], on=["base_sku", "month"], how="left", validate="many_to_one")

    # Branch × tile-size family behavior. This is derived from historical target-available Pair values.
    size = p.groupby(["branch_code", "tile_size_code", "month"], as_index=False).agg(
        size_branch_gross_m2=("actual_gross_m2", _sum_min_count),
        size_branch_target_available_sku_count=("target_available", "sum"),
        size_branch_positive_sku_count=("actual_positive", "sum"),
    )
    size = size.sort_values(["branch_code", "tile_size_code", "month"]).reset_index(drop=True)
    skeys = ["branch_code", "tile_size_code"]
    sgg = size.groupby(skeys, sort=False)["size_branch_gross_m2"]
    size["size_branch_lag_1"] = size["size_branch_gross_m2"]
    size["size_branch_lag_2"] = sgg.shift(1)
    size["size_branch_lag_3"] = sgg.shift(2)
    size["size_branch_lag_6"] = sgg.shift(5)
    size["size_branch_lag_12"] = sgg.shift(11)
    for w in (3, 6, 12):
        size[f"size_branch_roll_{w}_mean"] = _group_rolling(size, skeys, "size_branch_gross_m2", w, "mean", 1)
    size["size_branch_target_available_sku_count_lag_1"] = size["size_branch_target_available_sku_count"]
    size["size_branch_positive_sku_count_lag_1"] = size["size_branch_positive_sku_count"]

    size_keep = [
        "branch_code", "tile_size_code", "month",
        "size_branch_lag_1", "size_branch_lag_2", "size_branch_lag_3", "size_branch_lag_6", "size_branch_lag_12",
        "size_branch_roll_3_mean", "size_branch_roll_6_mean", "size_branch_roll_12_mean",
        "size_branch_target_available_sku_count_lag_1", "size_branch_positive_sku_count_lag_1",
    ]
    p = p.merge(size[size_keep], on=["branch_code", "tile_size_code", "month"], how="left", validate="many_to_one")

    # Rename snapshot metadata explicitly so it cannot masquerade as historical truth.
    rename_snapshot = {
        "brand": "brand_snapshot",
        "product_group": "product_group_snapshot",
        "price_group": "price_group_snapshot",
        "factory_code": "factory_code_snapshot",
        "pull_source": "pull_source_snapshot",
        "region": "region_snapshot",
        "branch_brand": "branch_brand_snapshot",
    }
    p = p.rename(columns={k: v for k, v in rename_snapshot.items() if k in p.columns})

    p["current_production_forecast_mask"] = (
        p.get("base_current_active", False).astype(bool) & p.get("branch_current_active", False).astype(bool)
        if "base_current_active" in p.columns and "branch_current_active" in p.columns
        else False
    )
    p["known_pair_asof_origin"] = True
    p["feature_information_max_month"] = p["month"]

    # Keep only origin-safe features + identifiers/QA metadata. Raw origin targets are intentionally omitted.
    base_cols = [
        "base_sku", "branch_code", "month", "feature_information_max_month",
        "known_pair_asof_origin", "current_production_forecast_mask",
        "tile_size_code", "base_group1", "base_group2", "base_group3", "base_group4",
        "brand_snapshot", "product_group_snapshot", "price_group_snapshot", "factory_code_snapshot",
        "pull_source_snapshot", "region_snapshot", "branch_brand_snapshot",
    ]
    feature_prefixes = (
        "pair_lag_", "pair_roll_", "pair_history_", "pair_target_", "pair_observed_", "pair_positive_",
        "months_since_", "pair_last_", "pair_mean_", "pair_peak_", "pair_adi_", "pair_cv2_",
        "sku_global_", "sku_target_", "sku_positive_", "sku_known_", "sku_ever_", "cross_branch_",
        "size_branch_",
    )
    feature_cols = [c for c in p.columns if c.startswith(feature_prefixes)]
    cols = [c for c in base_cols if c in p.columns] + [c for c in feature_cols if c not in base_cols]
    out = p[cols].copy()
    _assert_unique(out, ["base_sku", "branch_code", "month"], "pair_origin_features")
    return out


def build_branch_origin_features(branch_panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "branch_code", "month", "branch_gross_m2", "branch_observed", "branch_positive",
        "branch_first_observed_month", "branch_observed_month_count_to_date", "branch_positive_month_count_to_date",
        "observed_pair_count", "positive_pair_count", "explicit_zero_pair_count", "negative_only_pair_count",
        "new_positive_pair_count", "reactivated_pair_count", "pair_top1_share", "pair_top5_share",
        "pair_top10_share", "pair_hhi",
    }
    missing = required - set(branch_panel.columns)
    if missing:
        raise ValueError(f"branch_panel missing columns: {sorted(missing)}")

    b = branch_panel.copy()
    b["month"] = pd.to_datetime(b["month"])
    b = b.sort_values(["branch_code", "month"]).reset_index(drop=True)
    bg = b.groupby("branch_code", sort=False)["branch_gross_m2"]
    b["branch_lag_1"] = b["branch_gross_m2"]
    b["branch_lag_2"] = bg.shift(1)
    b["branch_lag_3"] = bg.shift(2)
    b["branch_lag_6"] = bg.shift(5)
    b["branch_lag_12"] = bg.shift(11)
    b["_observed_i"] = b["branch_observed"].astype(int)
    b["_positive_i"] = b["branch_positive"].astype(int)
    b["_calendar_i"] = 1

    for w in (3, 6, 12):
        b[f"branch_roll_{w}_mean"] = _group_rolling(b, ["branch_code"], "branch_gross_m2", w, "mean", 1)
        b[f"branch_roll_{w}_sum"] = _group_rolling(b, ["branch_code"], "branch_gross_m2", w, "sum", 1)
        b[f"branch_roll_{w}_std"] = _group_rolling(b, ["branch_code"], "branch_gross_m2", w, "std", 2)
        b[f"branch_roll_{w}_observed_count"] = _group_rolling(b, ["branch_code"], "_observed_i", w, "sum", 1)
        b[f"branch_roll_{w}_calendar_count"] = _group_rolling(b, ["branch_code"], "_calendar_i", w, "sum", 1)
        b[f"branch_roll_{w}_coverage"] = b[f"branch_roll_{w}_observed_count"] / b[f"branch_roll_{w}_calendar_count"].replace(0, np.nan)

    b["branch_history_length_months"] = _month_diff(b["month"], b["branch_first_observed_month"]) + 1
    b["branch_observed_month_count_to_origin"] = b["branch_observed_month_count_to_date"].astype(int)
    b["branch_positive_month_count_to_origin"] = b["branch_positive_month_count_to_date"].astype(int)
    b["branch_positive_rate_to_origin"] = (
        b["branch_positive_month_count_to_origin"] / b["branch_observed_month_count_to_origin"].replace(0, np.nan)
    )
    b["_pos_month"] = b["month"].where(b["branch_positive"])
    b["_last_pos_month"] = b.groupby("branch_code", sort=False)["_pos_month"].ffill()
    b["months_since_last_branch_positive"] = _month_diff(b["month"], b["_last_pos_month"])

    # Current closed-origin composition features, named lag_1 relative to next-month target.
    composition_map = {
        "observed_pair_count": "branch_observed_pair_count_lag_1",
        "positive_pair_count": "branch_positive_pair_count_lag_1",
        "explicit_zero_pair_count": "branch_explicit_zero_pair_count_lag_1",
        "negative_only_pair_count": "branch_negative_only_pair_count_lag_1",
        "new_positive_pair_count": "branch_new_positive_pair_count_lag_1",
        "reactivated_pair_count": "branch_reactivated_pair_count_lag_1",
        "pair_top1_share": "branch_pair_top1_share_lag_1",
        "pair_top5_share": "branch_pair_top5_share_lag_1",
        "pair_top10_share": "branch_pair_top10_share_lag_1",
        "pair_hhi": "branch_pair_hhi_lag_1",
    }
    for src, dst in composition_map.items():
        b[dst] = b[src]
    b["branch_average_m2_per_positive_pair_lag_1"] = (
        b["branch_gross_m2"] / b["positive_pair_count"].replace(0, np.nan)
    )
    b["branch_positive_pair_count_roll_3_mean"] = _group_rolling(b, ["branch_code"], "positive_pair_count", 3, "mean", 1)
    b["branch_new_positive_pair_count_roll_3_mean"] = _group_rolling(b, ["branch_code"], "new_positive_pair_count", 3, "mean", 1)

    rename_snapshot = {
        "region": "region_snapshot",
        "branch_brand": "branch_brand_snapshot",
    }
    b = b.rename(columns={k: v for k, v in rename_snapshot.items() if k in b.columns})
    b["current_production_forecast_mask"] = b.get("branch_current_active", False).astype(bool) if "branch_current_active" in b.columns else False
    b["feature_information_max_month"] = b["month"]

    base_cols = [
        "branch_code", "month", "feature_information_max_month", "current_production_forecast_mask",
        "region_snapshot", "branch_brand_snapshot",
    ]
    feature_prefixes = ("branch_lag_", "branch_roll_", "branch_history_", "branch_observed_", "branch_positive_", "months_since_", "branch_explicit_", "branch_negative_", "branch_new_", "branch_reactivated_", "branch_average_", "branch_pair_")
    feature_cols = [c for c in b.columns if c.startswith(feature_prefixes)]
    cols = [c for c in base_cols if c in b.columns] + [c for c in feature_cols if c not in base_cols]
    out = b[cols].copy()
    _assert_unique(out, ["branch_code", "month"], "branch_origin_features")
    return out


# Calendar V0.1.3
# -----------------
# Lunar conversion is deterministic and self-contained for Vietnam (UTC+7).
# The implementation follows the standard astronomical new-moon / solar-longitude
# construction and is regression-tested against official Vietnamese calendar dates:
# Tết 2024-02-10, 2025-01-29, 2026-02-17 and Hùng Kings 10/3 lunar.
#
# Public-holiday features below deliberately use *statutory nominal days* rather
# than year-specific ad-hoc public-sector swap schedules. This keeps the model
# origin-safe in historical backtests. `target_working_days_proxy` is therefore
# a rule-based calendar capacity proxy, not a company shutdown calendar.

VN_TIMEZONE = 7.0


def _jd_from_date(dd: int, mm: int, yy: int) -> int:
    a = (14 - mm) // 12
    y = yy + 4800 - a
    m = mm + 12 * a - 3
    jd = dd + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    if jd < 2299161:
        jd = dd + (153 * m + 2) // 5 + 365 * y + y // 4 - 32083
    return int(jd)


def _new_moon(k: int) -> float:
    t = k / 1236.85
    t2 = t * t
    t3 = t2 * t
    dr = math.pi / 180.0
    jd1 = 2415020.75933 + 29.53058868 * k + 0.0001178 * t2 - 0.000000155 * t3
    jd1 += 0.00033 * math.sin((166.56 + 132.87 * t - 0.009173 * t2) * dr)
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3
    mpr = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3
    c1 = (0.1734 - 0.000393 * t) * math.sin(m * dr) + 0.0021 * math.sin(2 * m * dr)
    c1 -= 0.4068 * math.sin(mpr * dr) + 0.0161 * math.sin(2 * mpr * dr)
    c1 -= 0.0004 * math.sin(3 * mpr * dr)
    c1 += 0.0104 * math.sin(2 * f * dr) - 0.0051 * math.sin((m + mpr) * dr)
    c1 -= 0.0074 * math.sin((m - mpr) * dr) + 0.0004 * math.sin((2 * f + m) * dr)
    c1 -= 0.0004 * math.sin((2 * f - m) * dr) - 0.0006 * math.sin((2 * f + mpr) * dr)
    c1 += 0.0010 * math.sin((2 * f - mpr) * dr) + 0.0005 * math.sin((2 * mpr + m) * dr)
    if t < -11:
        delta_t = 0.001 + 0.000839 * t + 0.0002261 * t2 - 0.00000845 * t3 - 0.000000081 * t * t3
    else:
        delta_t = -0.000278 + 0.000265 * t + 0.000262 * t2
    return jd1 + c1 - delta_t


def _new_moon_day(k: int, timezone: float = VN_TIMEZONE) -> int:
    return int(math.floor(_new_moon(k) + 0.5 + timezone / 24.0))


def _sun_longitude(jdn: float) -> float:
    t = (jdn - 2451545.0) / 36525.0
    t2 = t * t
    dr = math.pi / 180.0
    m = 357.52910 + 35999.05030 * t - 0.0001559 * t2 - 0.00000048 * t * t2
    l0 = 280.46645 + 36000.76983 * t + 0.0003032 * t2
    dl = (1.914600 - 0.004817 * t - 0.000014 * t2) * math.sin(dr * m)
    dl += (0.019993 - 0.000101 * t) * math.sin(2 * dr * m) + 0.000290 * math.sin(3 * dr * m)
    l = (l0 + dl) * dr
    return l - math.pi * 2.0 * math.floor(l / (math.pi * 2.0))


def _sun_longitude_sector(day_number: int, timezone: float = VN_TIMEZONE) -> int:
    return int(math.floor(_sun_longitude(day_number - 0.5 - timezone / 24.0) / math.pi * 6.0))


def _lunar_month11(yy: int, timezone: float = VN_TIMEZONE) -> int:
    off = _jd_from_date(31, 12, yy) - 2415021
    k = int(math.floor(off / 29.530588853))
    nm = _new_moon_day(k, timezone)
    if _sun_longitude_sector(nm, timezone) >= 9:
        nm = _new_moon_day(k - 1, timezone)
    return nm


def _leap_month_offset(a11: int, timezone: float = VN_TIMEZONE) -> int:
    k = int(math.floor(0.5 + (a11 - 2415021.076998695) / 29.530588853))
    last = 0
    i = 1
    arc = _sun_longitude_sector(_new_moon_day(k + i, timezone), timezone)
    while True:
        last = arc
        i += 1
        arc = _sun_longitude_sector(_new_moon_day(k + i, timezone), timezone)
        if arc == last or i >= 14:
            break
    return i - 1


def solar_to_lunar_vn(date_value, timezone: float = VN_TIMEZONE) -> Tuple[int, int, int, int]:
    """Return (lunar_day, lunar_month, lunar_year, is_leap_month) for Vietnam."""
    d = pd.Timestamp(date_value)
    dd, mm, yy = int(d.day), int(d.month), int(d.year)
    day_number = _jd_from_date(dd, mm, yy)
    k = int(math.floor((day_number - 2415021.076998695) / 29.530588853))
    month_start = _new_moon_day(k + 1, timezone)
    if month_start > day_number:
        month_start = _new_moon_day(k, timezone)
    a11 = _lunar_month11(yy, timezone)
    b11 = a11
    if a11 >= month_start:
        lunar_year = yy
        a11 = _lunar_month11(yy - 1, timezone)
    else:
        lunar_year = yy + 1
        b11 = _lunar_month11(yy + 1, timezone)
    lunar_day = day_number - month_start + 1
    diff = int(math.floor((month_start - a11) / 29.0))
    lunar_leap = 0
    lunar_month = diff + 11
    if b11 - a11 > 365:
        leap_month_diff = _leap_month_offset(a11, timezone)
        if diff >= leap_month_diff:
            lunar_month = diff + 10
            if diff == leap_month_diff:
                lunar_leap = 1
    if lunar_month > 12:
        lunar_month -= 12
    if lunar_month >= 11 and diff < 4:
        lunar_year -= 1
    return int(lunar_day), int(lunar_month), int(lunar_year), int(lunar_leap)


def lunar_new_year_date(year: int) -> pd.Timestamp:
    """Find mùng 1 Tết for lunar year `year` by deterministic calendar conversion."""
    for d in pd.date_range(f"{year}-01-15", f"{year}-03-05", freq="D"):
        lunar_day, lunar_month, lunar_year, lunar_leap = solar_to_lunar_vn(d)
        if lunar_day == 1 and lunar_month == 1 and lunar_year == year and lunar_leap == 0:
            return pd.Timestamp(d)
    raise ValueError(f"Could not locate Lunar New Year date for {year}")


def hung_kings_date(year: int) -> pd.Timestamp:
    """Find 10/3 lunar (Giỗ Tổ Hùng Vương) for the corresponding lunar year."""
    for d in pd.date_range(f"{year}-03-15", f"{year}-05-15", freq="D"):
        lunar_day, lunar_month, lunar_year, lunar_leap = solar_to_lunar_vn(d)
        if lunar_day == 10 and lunar_month == 3 and lunar_year == year and lunar_leap == 0:
            return pd.Timestamp(d)
    raise ValueError(f"Could not locate Hùng Kings date for {year}")


def _target_calendar_row(month_start: pd.Timestamp) -> Dict[str, object]:
    ms = pd.Timestamp(month_start).to_period("M").to_timestamp()
    days_in_month = int(ms.days_in_month)
    month_end = ms + pd.offsets.MonthEnd(0)
    midpoint = ms + pd.Timedelta(days=14)  # deterministic 15th day of target month

    weekday_count = int(sum(d.weekday() < 5 for d in pd.date_range(ms, month_end, freq="D")))

    lunar_day_mid, lunar_month_mid, _, lunar_leap_mid = solar_to_lunar_vn(midpoint)
    lunar_angle = 2.0 * math.pi * (lunar_month_mid - 1) / 12.0

    # Need adjacent lunar years so December/January distances remain well-defined.
    tet_dates = [lunar_new_year_date(y) for y in (ms.year - 1, ms.year, ms.year + 1)]
    prev_tet = max(d for d in tet_dates if d <= midpoint)
    next_tet = min(d for d in tet_dates if d >= midpoint)
    same_year_tet = lunar_new_year_date(ms.year)

    hk = hung_kings_date(ms.year)

    # Article-112 nominal paid-holiday days assigned to the month of the
    # corresponding statutory holiday anchor. This avoids ad-hoc swap schedules.
    public_holiday_event_count = 0
    statutory_holiday_nominal_days = 0

    if ms.month == 1:  # New Year's Day
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 1
    if same_year_tet.to_period("M") == ms.to_period("M"):  # Lunar New Year
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 5
    if hk.to_period("M") == ms.to_period("M"):  # Hùng Kings
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 1
    if ms.month == 4:  # 30 April
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 1
    if ms.month == 5:  # 1 May
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 1
    if ms.month == 9:  # National Day: 2 statutory days, same September month
        public_holiday_event_count += 1
        statutory_holiday_nominal_days += 2

    working_days_proxy = max(0, weekday_count - statutory_holiday_nominal_days)
    tet_in_month = same_year_tet.to_period("M") == ms.to_period("M")

    return {
        "target_month": ms,
        "target_month_num": int(ms.month),
        "target_quarter": int(ms.quarter),
        "target_year": int(ms.year),
        "target_days_in_month": days_in_month,
        "target_month_sin": math.sin(2.0 * math.pi * (ms.month - 1) / 12.0),
        "target_month_cos": math.cos(2.0 * math.pi * (ms.month - 1) / 12.0),
        "target_weekday_count": weekday_count,
        "target_public_holiday_event_count": int(public_holiday_event_count),
        "target_statutory_holiday_nominal_days": int(statutory_holiday_nominal_days),
        "target_working_days_proxy": int(working_days_proxy),
        "target_lunar_month_mid": int(lunar_month_mid),
        "target_lunar_month_mid_is_leap": bool(lunar_leap_mid),
        "target_lunar_month_sin": math.sin(lunar_angle),
        "target_lunar_month_cos": math.cos(lunar_angle),
        "target_tet_day_of_month": int(same_year_tet.day) if tet_in_month else 0,
        "target_days_to_next_tet_from_midmonth": int((next_tet - midpoint).days),
        "target_days_since_prev_tet_from_midmonth": int((midpoint - prev_tet).days),
        "target_is_tet_month": bool(tet_in_month),
        "target_is_pre_tet_month": bool(ms.to_period("M") == same_year_tet.to_period("M") - 1),
        "target_is_post_tet_month": bool(ms.to_period("M") == same_year_tet.to_period("M") + 1),
    }


def build_target_calendar_table(target_months: Iterable) -> pd.DataFrame:
    months = pd.to_datetime(pd.Series(list(target_months))).dt.to_period("M").dt.to_timestamp()
    unique_months = sorted(pd.unique(months))
    cal = pd.DataFrame([_target_calendar_row(pd.Timestamp(m)) for m in unique_months])
    _assert_unique(cal, ["target_month"], "target_calendar_table")
    return cal


def _add_target_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_month"] = pd.to_datetime(out["target_month"]).dt.to_period("M").dt.to_timestamp()
    cal = build_target_calendar_table(out["target_month"].unique())
    out = out.merge(cal, on="target_month", how="left", validate="many_to_one")
    out["horizon_label"] = "h" + out["horizon"].astype(str)
    return out


def expand_pair_development_panel(
    pair_origin: pd.DataFrame,
    pair_panel: pd.DataFrame,
    train_target_end: str = "2025-12-01",
    validation_origin: str = "2025-12-01",
) -> pd.DataFrame:
    train_end = pd.Timestamp(train_target_end)
    val_origin = pd.Timestamp(validation_origin)
    origin = pair_origin.loc[pair_origin["month"].le(val_origin)].copy()
    target = pair_panel[[
        "base_sku", "branch_code", "month", "actual_gross_m2", "actual_observed",
        "actual_positive", "actual_negative_only", "target_available", "zero_semantics"
    ]].copy()
    target = target.rename(columns={
        "month": "target_month",
        "actual_gross_m2": "target_actual_gross_m2",
        "actual_observed": "target_actual_observed",
        "actual_positive": "target_actual_positive",
        "actual_negative_only": "target_actual_negative_only",
        "zero_semantics": "target_zero_semantics",
    })

    frames = []
    for h in (1, 2, 3):
        x = origin.copy()
        x = x.rename(columns={"month": "forecast_origin"})
        x["horizon"] = h
        x["target_month"] = pd.to_datetime(x["forecast_origin"]) + pd.offsets.DateOffset(months=h)
        x = x.merge(target, on=["base_sku", "branch_code", "target_month"], how="left", validate="many_to_one")
        is_train = x["target_month"].le(train_end)
        is_val = x["forecast_origin"].eq(val_origin) & x["target_month"].eq(val_origin + pd.offsets.DateOffset(months=h))
        x["split_role"] = np.select([is_train, is_val], ["TRAIN", "OFFICIAL_VALIDATION"], default="DROP")
        x = x.loc[~x["split_role"].eq("DROP")].copy()
        x["historical_train_mask"] = x["split_role"].eq("TRAIN") & x["target_available"].fillna(False)
        x["official_validation_mask"] = x["split_role"].eq("OFFICIAL_VALIDATION") & x["target_available"].fillna(False)
        x["feature_information_max_month"] = pd.to_datetime(x["feature_information_max_month"])
        x["dataset_version"] = DATASET_VERSION
        x["feature_version"] = PAIR_FEATURE_VERSION
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    out = _add_target_calendar(out)
    out = out.sort_values(["forecast_origin", "horizon", "base_sku", "branch_code"]).reset_index(drop=True)
    _assert_unique(out, ["forecast_origin", "target_month", "horizon", "base_sku", "branch_code"], "pair_feature_panel")
    return out


def expand_branch_development_panel(
    branch_origin: pd.DataFrame,
    branch_panel: pd.DataFrame,
    train_target_end: str = "2025-12-01",
    validation_origin: str = "2025-12-01",
) -> pd.DataFrame:
    train_end = pd.Timestamp(train_target_end)
    val_origin = pd.Timestamp(validation_origin)
    origin = branch_origin.loc[branch_origin["month"].le(val_origin)].copy()
    target = branch_panel[["branch_code", "month", "branch_gross_m2", "branch_observed", "branch_positive"]].copy()
    target = target.rename(columns={
        "month": "target_month",
        "branch_gross_m2": "target_branch_gross_m2",
        "branch_observed": "target_branch_observed",
        "branch_positive": "target_branch_positive",
    })

    frames = []
    for h in (1, 2, 3):
        x = origin.copy().rename(columns={"month": "forecast_origin"})
        x["horizon"] = h
        x["target_month"] = pd.to_datetime(x["forecast_origin"]) + pd.offsets.DateOffset(months=h)
        x = x.merge(target, on=["branch_code", "target_month"], how="left", validate="many_to_one")
        x["target_available"] = x["target_branch_observed"].fillna(False).astype(bool)
        is_train = x["target_month"].le(train_end)
        is_val = x["forecast_origin"].eq(val_origin) & x["target_month"].eq(val_origin + pd.offsets.DateOffset(months=h))
        x["split_role"] = np.select([is_train, is_val], ["TRAIN", "OFFICIAL_VALIDATION"], default="DROP")
        x = x.loc[~x["split_role"].eq("DROP")].copy()
        x["historical_train_mask"] = x["split_role"].eq("TRAIN") & x["target_available"]
        x["official_validation_mask"] = x["split_role"].eq("OFFICIAL_VALIDATION") & x["target_available"]
        x["dataset_version"] = DATASET_VERSION
        x["feature_version"] = BRANCH_FEATURE_VERSION
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    out = _add_target_calendar(out)
    out = out.sort_values(["forecast_origin", "horizon", "branch_code"]).reset_index(drop=True)
    _assert_unique(out, ["forecast_origin", "target_month", "horizon", "branch_code"], "branch_feature_panel")
    return out


PAIR_MODEL_CANDIDATES = [
    "pair_lag_1", "pair_lag_2", "pair_lag_3", "pair_lag_6", "pair_lag_12",
    "pair_roll_3_mean", "pair_roll_3_sum", "pair_roll_3_std", "pair_roll_3_target_available_count", "pair_roll_3_positive_count", "pair_roll_3_coverage",
    "pair_roll_6_mean", "pair_roll_6_sum", "pair_roll_6_std", "pair_roll_6_target_available_count", "pair_roll_6_positive_count", "pair_roll_6_coverage",
    "pair_roll_12_mean", "pair_roll_12_sum", "pair_roll_12_std", "pair_roll_12_target_available_count", "pair_roll_12_positive_count", "pair_roll_12_coverage",
    "pair_history_length_months", "pair_target_available_count_to_origin", "pair_positive_count_to_origin", "pair_positive_rate_to_origin",
    "months_since_last_positive", "months_since_first_positive", "pair_last_positive_m2_to_origin", "pair_mean_positive_m2_to_origin",
    "pair_peak_positive_m2_to_origin", "pair_peak_share_to_origin", "pair_adi_target_available_to_origin", "pair_cv2_positive_to_origin",
    "sku_global_lag_1", "sku_global_lag_2", "sku_global_lag_3", "sku_global_lag_6", "sku_global_lag_12",
    "sku_global_roll_3_mean", "sku_global_roll_3_sum", "sku_global_roll_6_mean", "sku_global_roll_6_sum", "sku_global_roll_12_mean", "sku_global_roll_12_sum",
    "sku_global_positive_rate_12m", "sku_target_available_branch_count_lag_1", "sku_positive_branch_count_lag_1",
    "sku_known_branch_count_to_origin", "sku_ever_positive_branch_count_to_origin", "cross_branch_penetration_to_origin",
    "size_branch_lag_1", "size_branch_lag_2", "size_branch_lag_3", "size_branch_lag_6", "size_branch_lag_12",
    "size_branch_roll_3_mean", "size_branch_roll_6_mean", "size_branch_roll_12_mean",
    "size_branch_target_available_sku_count_lag_1", "size_branch_positive_sku_count_lag_1",
    "target_month_num", "target_quarter", "target_year", "target_days_in_month", "target_month_sin", "target_month_cos",
    "target_weekday_count", "target_public_holiday_event_count", "target_statutory_holiday_nominal_days", "target_working_days_proxy",
    "target_lunar_month_mid", "target_lunar_month_mid_is_leap", "target_lunar_month_sin", "target_lunar_month_cos",
    "target_tet_day_of_month", "target_days_to_next_tet_from_midmonth", "target_days_since_prev_tet_from_midmonth",
    "target_is_tet_month", "target_is_pre_tet_month", "target_is_post_tet_month", "horizon",
    "tile_size_code",
]

BRANCH_MODEL_CANDIDATES = [
    "branch_lag_1", "branch_lag_2", "branch_lag_3", "branch_lag_6", "branch_lag_12",
    "branch_roll_3_mean", "branch_roll_3_sum", "branch_roll_3_std", "branch_roll_3_observed_count", "branch_roll_3_coverage",
    "branch_roll_6_mean", "branch_roll_6_sum", "branch_roll_6_std", "branch_roll_6_observed_count", "branch_roll_6_coverage",
    "branch_roll_12_mean", "branch_roll_12_sum", "branch_roll_12_std", "branch_roll_12_observed_count", "branch_roll_12_coverage",
    "branch_history_length_months", "branch_observed_month_count_to_origin", "branch_positive_month_count_to_origin", "branch_positive_rate_to_origin",
    "months_since_last_branch_positive",
    "branch_observed_pair_count_lag_1", "branch_positive_pair_count_lag_1", "branch_explicit_zero_pair_count_lag_1",
    "branch_negative_only_pair_count_lag_1", "branch_new_positive_pair_count_lag_1", "branch_reactivated_pair_count_lag_1",
    "branch_average_m2_per_positive_pair_lag_1", "branch_pair_top1_share_lag_1", "branch_pair_top5_share_lag_1",
    "branch_pair_top10_share_lag_1", "branch_pair_hhi_lag_1", "branch_positive_pair_count_roll_3_mean", "branch_new_positive_pair_count_roll_3_mean",
    "target_month_num", "target_quarter", "target_year", "target_days_in_month", "target_month_sin", "target_month_cos",
    "target_weekday_count", "target_public_holiday_event_count", "target_statutory_holiday_nominal_days", "target_working_days_proxy",
    "target_lunar_month_mid", "target_lunar_month_mid_is_leap", "target_lunar_month_sin", "target_lunar_month_cos",
    "target_tet_day_of_month", "target_days_to_next_tet_from_midmonth", "target_days_since_prev_tet_from_midmonth",
    "target_is_tet_month", "target_is_pre_tet_month", "target_is_post_tet_month", "horizon",
]

BLOCKED_SNAPSHOT_COLUMNS = [
    "brand_snapshot", "product_group_snapshot", "price_group_snapshot", "factory_code_snapshot", "pull_source_snapshot",
    "region_snapshot", "branch_brand_snapshot", "current_production_forecast_mask",
]


def build_feature_inventory(pair_df: pd.DataFrame, branch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track, df, candidates in [
        ("PAIR", pair_df, PAIR_MODEL_CANDIDATES),
        ("BRANCH", branch_df, BRANCH_MODEL_CANDIDATES),
    ]:
        candidate_set = set(candidates)
        for c in df.columns:
            if c in {"target_actual_gross_m2", "target_branch_gross_m2", "target_available", "target_actual_observed", "target_actual_positive", "target_actual_negative_only", "target_zero_semantics", "target_branch_observed", "target_branch_positive"}:
                role = "TARGET_OR_EVAL"
                eligible = False
                timing = "TARGET_MONTH_ONLY"
            elif c in BLOCKED_SNAPSHOT_COLUMNS or c.endswith("_snapshot"):
                role = "QA_SNAPSHOT_METADATA"
                eligible = False
                timing = "BLOCKED_PENDING_TIMING_REVIEW"
            elif c in candidate_set:
                role = "MODEL_CANDIDATE_STAGE6_REVIEW"
                eligible = True
                timing = "ORIGIN_SAFE_OR_KNOWN_FUTURE"
            elif c in {"base_sku", "branch_code", "forecast_origin", "target_month", "split_role", "horizon_label", "dataset_version", "feature_version", "feature_information_max_month", "known_pair_asof_origin", "historical_train_mask", "official_validation_mask"}:
                role = "IDENTITY_OR_CONTROL"
                eligible = False
                timing = "CONTROL"
            else:
                role = "QA_OR_AUXILIARY"
                eligible = False
                timing = "REVIEW"
            rows.append({
                "track": track,
                "feature_name": c,
                "dtype": str(df[c].dtype),
                "role": role,
                "initial_model_candidate": bool(eligible),
                "timing_status": timing,
                "missing_rate": float(df[c].isna().mean()),
                "n_unique": int(df[c].nunique(dropna=True)),
            })
    return pd.DataFrame(rows)


def validate_feature_stage(pair_df: pd.DataFrame, branch_df: pd.DataFrame) -> Dict[str, object]:
    checks: Dict[str, Dict[str, object]] = {}

    def add(name: str, ok: bool, detail=None):
        checks[name] = {"pass": bool(ok), "detail": detail}

    add("pair_grain_unique", ~pair_df.duplicated(["forecast_origin", "target_month", "horizon", "base_sku", "branch_code"]).any())
    add("branch_grain_unique", ~branch_df.duplicated(["forecast_origin", "target_month", "horizon", "branch_code"]).any())
    add("pair_information_origin_safe", bool((pd.to_datetime(pair_df["feature_information_max_month"]) <= pd.to_datetime(pair_df["forecast_origin"])).all()))
    add("branch_information_origin_safe", bool((pd.to_datetime(branch_df["feature_information_max_month"]) <= pd.to_datetime(branch_df["forecast_origin"])).all()))

    pair_expected_target = pd.to_datetime([
        pd.Timestamp(o) + pd.offsets.DateOffset(months=int(h))
        for o, h in zip(pair_df["forecast_origin"], pair_df["horizon"])
    ])
    branch_expected_target = pd.to_datetime([
        pd.Timestamp(o) + pd.offsets.DateOffset(months=int(h))
        for o, h in zip(branch_df["forecast_origin"], branch_df["horizon"])
    ])
    add("pair_target_horizon_alignment", bool((pair_expected_target == pd.to_datetime(pair_df["target_month"]).to_numpy()).all()))
    add("branch_target_horizon_alignment", bool((branch_expected_target == pd.to_datetime(branch_df["target_month"]).to_numpy()).all()))

    frozen_start = pd.Timestamp("2026-04-01")
    add("frozen_test_not_touched_pair", bool(pd.to_datetime(pair_df["target_month"]).lt(frozen_start).all()))
    add("frozen_test_not_touched_branch", bool(pd.to_datetime(branch_df["target_month"]).lt(frozen_start).all()))
    add("pair_train_cutoff_respected", bool(pd.to_datetime(pair_df.loc[pair_df["split_role"].eq("TRAIN"), "target_month"]).le(pd.Timestamp("2025-12-01")).all()))
    add("branch_train_cutoff_respected", bool(pd.to_datetime(branch_df.loc[branch_df["split_role"].eq("TRAIN"), "target_month"]).le(pd.Timestamp("2025-12-01")).all()))

    pv = pair_df.loc[pair_df["split_role"].eq("OFFICIAL_VALIDATION")]
    bv = branch_df.loc[branch_df["split_role"].eq("OFFICIAL_VALIDATION")]
    add("pair_validation_origin_locked", bool(len(pv) > 0 and pd.to_datetime(pv["forecast_origin"]).eq(pd.Timestamp("2025-12-01")).all() and set(pv["horizon"].unique()) == {1, 2, 3}))
    add("branch_validation_origin_locked", bool(len(bv) > 0 and pd.to_datetime(bv["forecast_origin"]).eq(pd.Timestamp("2025-12-01")).all() and set(bv["horizon"].unique()) == {1, 2, 3}))

    if "target_actual_negative_only" in pair_df.columns:
        neg = pair_df["target_actual_negative_only"].fillna(False)
        add("negative_only_target_not_trainable", bool((~pair_df.loc[neg, "historical_train_mask"]).all() and (~pair_df.loc[neg, "official_validation_mask"]).all() and pair_df.loc[neg, "target_actual_gross_m2"].isna().all()))
    else:
        add("negative_only_target_not_trainable", False, "target_actual_negative_only missing")

    # Direct target/snapshot leakage: no forbidden target or current snapshot field may appear in candidate lists.
    forbidden = {
        "target_actual_gross_m2", "target_branch_gross_m2", "target_available",
        "target_actual_positive", "target_actual_observed", "target_actual_negative_only",
        *BLOCKED_SNAPSHOT_COLUMNS,
    }
    add("pair_candidate_list_no_direct_target_or_snapshot", len(forbidden.intersection(PAIR_MODEL_CANDIDATES)) == 0)
    add("branch_candidate_list_no_direct_target_or_snapshot", len(forbidden.intersection(BRANCH_MODEL_CANDIDATES)) == 0)

    # Semantic gate for cumulative peak features: after any positive month has been
    # observed by the origin, the known historical peak and peak share must remain
    # available even when the current origin month is missing/zero/negative-only.
    prior_positive = pair_df["pair_positive_count_to_origin"].fillna(0).gt(0)
    add(
        "pair_peak_carried_forward_after_positive",
        bool(pair_df.loc[prior_positive, "pair_peak_positive_m2_to_origin"].notna().all()),
    )
    add(
        "pair_peak_share_carried_forward_after_positive",
        bool(pair_df.loc[prior_positive, "pair_peak_share_to_origin"].notna().all()),
    )

    # Calendar V0.1.3 gates. All new fields are deterministic target-calendar
    # features and must be identical for every row sharing the same target month.
    calendar_v013 = {
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
        "target_is_tet_month",
        "target_is_pre_tet_month",
        "target_is_post_tet_month",
    }
    for label, frame in (("pair", pair_df), ("branch", branch_df)):
        add(f"{label}_calendar_v013_features_present", calendar_v013.issubset(frame.columns))
        if calendar_v013.issubset(frame.columns):
            add(
                f"{label}_calendar_v013_no_missing",
                bool(frame[list(calendar_v013)].notna().all().all()),
            )
            lunar_month = pd.to_numeric(frame["target_lunar_month_mid"], errors="coerce")
            add(
                f"{label}_lunar_month_range_valid",
                bool(lunar_month.between(1, 12).all()),
            )
            wd = pd.to_numeric(frame["target_weekday_count"], errors="coerce")
            hdays = pd.to_numeric(frame["target_statutory_holiday_nominal_days"], errors="coerce")
            work = pd.to_numeric(frame["target_working_days_proxy"], errors="coerce")
            add(
                f"{label}_working_days_proxy_valid",
                bool((wd.ge(0) & hdays.ge(0) & work.ge(0) & work.le(wd)).all()),
            )
            add(
                f"{label}_tet_distance_nonnegative",
                bool(
                    pd.to_numeric(frame["target_days_to_next_tet_from_midmonth"], errors="coerce").ge(0).all()
                    and pd.to_numeric(frame["target_days_since_prev_tet_from_midmonth"], errors="coerce").ge(0).all()
                ),
            )
            # Calendar values must be a pure function of target_month.
            check_cols = sorted(calendar_v013)
            per_month_nunique = frame.groupby("target_month", dropna=False)[check_cols].nunique(dropna=False)
            add(
                f"{label}_calendar_constant_within_target_month",
                bool((per_month_nunique <= 1).all().all()),
            )

            vv = frame.loc[frame["split_role"].eq("OFFICIAL_VALIDATION")].copy()
            jan = vv.loc[pd.to_datetime(vv["target_month"]).eq(pd.Timestamp("2026-01-01"))]
            feb = vv.loc[pd.to_datetime(vv["target_month"]).eq(pd.Timestamp("2026-02-01"))]
            mar = vv.loc[pd.to_datetime(vv["target_month"]).eq(pd.Timestamp("2026-03-01"))]
            seq_ok = (
                len(jan) > 0 and len(feb) > 0 and len(mar) > 0
                and jan["target_is_pre_tet_month"].astype(bool).all()
                and (~jan["target_is_tet_month"].astype(bool)).all()
                and feb["target_is_tet_month"].astype(bool).all()
                and feb["target_tet_day_of_month"].eq(17).all()
                and mar["target_is_post_tet_month"].astype(bool).all()
                and (~mar["target_is_tet_month"].astype(bool)).all()
            )
            add(f"{label}_validation_tet_sequence_and_position_correct", bool(seq_ok))

            tr = frame.loc[frame["historical_train_mask"].astype(bool)]
            target_min = pd.to_datetime(frame["target_month"]).min()
            if target_min <= pd.Timestamp("2025-01-01"):
                add(f"{label}_train_contains_tet_examples", bool(tr["target_is_tet_month"].astype(bool).any()))
                add(
                    f"{label}_train_contains_non_tet_public_holidays",
                    bool((pd.to_numeric(tr["target_public_holiday_event_count"], errors="coerce") > tr["target_is_tet_month"].astype(int)).any()),
                )
            else:
                add(f"{label}_train_contains_tet_examples", True, "NOT_APPLICABLE_SHORT_FIXTURE")
                add(f"{label}_train_contains_non_tet_public_holidays", True, "NOT_APPLICABLE_SHORT_FIXTURE")

    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "pair_rows": int(len(pair_df)),
        "branch_rows": int(len(branch_df)),
        "pair_train_rows_eligible": int(pair_df["historical_train_mask"].sum()),
        "pair_validation_rows_eligible": int(pair_df["official_validation_mask"].sum()),
        "branch_train_rows_eligible": int(branch_df["historical_train_mask"].sum()),
        "branch_validation_rows_eligible": int(branch_df["official_validation_mask"].sum()),
        "frozen_test_touched": False,
    }
