from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

DIAG_VERSION = "underforecast_diagnosis_v01"
TARGET = "target_actual_gross_m2"
CORE_MODEL = "lightgbm_tweedie"
CORE_PRED = "pred_lightgbm_tweedie"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wape(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    den = float(np.abs(y).sum())
    return float(np.abs(y - p).sum() / den) if den > 0 else float("nan")


def mae(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def bias(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float((p - y).sum())


def bias_ratio(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    den = float(np.abs(y).sum())
    return float(bias(y, y_pred) / den) if den > 0 else float("nan")


def _metric_row(df: pd.DataFrame, pred_col: str, **labels) -> dict:
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(dtype=float)
    under = np.maximum(y - p, 0.0)
    over = np.maximum(p - y, 0.0)
    actual_sum = float(y.sum())
    return {
        **labels,
        "n_rows": int(len(df)),
        "actual_sum_m2": actual_sum,
        "forecast_sum_m2": float(p.sum()),
        "wape": wape(y, p),
        "mae": mae(y, p),
        "bias_m2": bias(y, p),
        "bias_ratio": bias_ratio(y, p),
        "underforecast_m2": float(under.sum()),
        "overforecast_m2": float(over.sum()),
    }


def classify_demand_event(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Mutually-exclusive diagnostic event class.

    This is NOT a model label. It is computed after rolling evaluation only to explain
    where error volume comes from.
    """
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)
    pos_count = pd.to_numeric(df.get("pair_positive_count_to_origin"), errors="coerce").fillna(0)
    gap = pd.to_numeric(df.get("months_since_last_positive"), errors="coerce")
    mean_pos = pd.to_numeric(df.get("pair_mean_positive_m2_to_origin"), errors="coerce")

    medium_gap = int(cfg["event_rules"].get("medium_gap_months", 3))
    long_gap = int(cfg["event_rules"].get("long_gap_months", 6))
    spike2 = float(cfg["event_rules"].get("moderate_spike_ratio", 2.0))
    spike3 = float(cfg["event_rules"].get("extreme_spike_ratio", 3.0))
    min_spike_m2 = float(cfg["event_rules"].get("min_spike_actual_m2", 1.0))

    out = pd.Series("ongoing_positive", index=df.index, dtype="object")
    out.loc[y.le(0)] = "actual_zero"

    positive = y.gt(0)
    first_pos = positive & pos_count.le(0)
    out.loc[first_pos] = "first_positive_known_pair"

    remaining = positive & ~first_pos
    long_react = remaining & gap.ge(long_gap)
    out.loc[long_react] = "reactivation_gap_6plus"

    med_react = remaining & ~long_react & gap.ge(medium_gap)
    out.loc[med_react] = "reactivation_gap_3_5"

    short_react = remaining & ~long_react & ~med_react & gap.ge(1)
    out.loc[short_react] = "reactivation_gap_1_2"

    not_react = remaining & ~long_react & ~med_react & ~short_react
    ratio = y / mean_pos.replace(0, np.nan)
    extreme = not_react & y.ge(min_spike_m2) & ratio.ge(spike3)
    out.loc[extreme] = "spike_3x_plus"
    moderate = not_react & ~extreme & y.ge(min_spike_m2) & ratio.ge(spike2)
    out.loc[moderate] = "spike_2x_3x"
    return out


def build_underforecast_diagnosis(df: pd.DataFrame, cfg: dict, pred_col: str = CORE_PRED) -> pd.DataFrame:
    d = df.copy()
    d["demand_event"] = classify_demand_event(d, cfg)
    rows = []

    # Overall, horizon, behavior, event, and event x horizon.
    rows.append(_metric_row(d, pred_col, dimension="OVERALL", bucket="ALL", horizon="ALL"))
    for h, g in d.groupby("horizon", dropna=False):
        rows.append(_metric_row(g, pred_col, dimension="HORIZON", bucket=f"H{int(h)}", horizon=int(h)))
    if "behavior_segment" in d.columns:
        for seg, g in d.groupby("behavior_segment", dropna=False):
            rows.append(_metric_row(g, pred_col, dimension="BEHAVIOR", bucket=str(seg), horizon="ALL"))
    for event, g in d.groupby("demand_event", dropna=False):
        rows.append(_metric_row(g, pred_col, dimension="DEMAND_EVENT", bucket=str(event), horizon="ALL"))
        for h, gh in g.groupby("horizon", dropna=False):
            rows.append(_metric_row(gh, pred_col, dimension="DEMAND_EVENT_X_HORIZON", bucket=str(event), horizon=int(h)))

    out = pd.DataFrame(rows)
    total_actual = float(pd.to_numeric(d[TARGET], errors="coerce").fillna(0).sum())
    total_under = float(np.maximum(
        pd.to_numeric(d[TARGET], errors="coerce").fillna(0).to_numpy(dtype=float)
        - pd.to_numeric(d[pred_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float), 0
    ).sum())
    out["share_of_total_actual"] = out["actual_sum_m2"] / total_actual if total_actual > 0 else np.nan
    out["share_of_total_underforecast"] = out["underforecast_m2"] / total_under if total_under > 0 else np.nan
    return out


def build_origin_event_diagnosis(df: pd.DataFrame, cfg: dict, pred_col: str = CORE_PRED) -> pd.DataFrame:
    d = df.copy()
    d["forecast_origin"] = pd.to_datetime(d["forecast_origin"])
    d["demand_event"] = classify_demand_event(d, cfg)
    rows = []
    for (origin, event), g in d.groupby(["forecast_origin", "demand_event"], dropna=False):
        rows.append(_metric_row(g, pred_col, forecast_origin=origin, demand_event=event))
    return pd.DataFrame(rows)


def _round_nearest_1(x: pd.Series) -> pd.Series:
    # deterministic half-up for non-negative demand forecasts
    v = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(lower=0.0)
    return np.floor(v + 0.5)


def _ceil_1(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(lower=0.0)
    return np.ceil(v)


def _round_policy_series(x: pd.Series, policy: str) -> pd.Series:
    if policy == "raw":
        return pd.to_numeric(x, errors="coerce").fillna(0.0).clip(lower=0.0)
    if policy == "round_nearest_1m2":
        return pd.Series(_round_nearest_1(x), index=x.index)
    if policy == "ceil_1m2":
        return pd.Series(_ceil_1(x), index=x.index)
    raise ValueError(f"Unsupported rounding policy: {policy}")


def compare_pair_month_rounding(df: pd.DataFrame, pred_col: str = CORE_PRED) -> pd.DataFrame:
    rows = []
    for policy in ["raw", "round_nearest_1m2", "ceil_1m2"]:
        d = df.copy()
        d["_pred"] = _round_policy_series(d[pred_col], policy)
        r = _metric_row(d.rename(columns={"_pred": "_tmp_pred"}), "_tmp_pred", level="PAIR_MONTH", policy=policy)
        raw = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).clip(lower=0)
        transformed = d["_pred"]
        r["added_m2_vs_raw"] = float((transformed - raw).sum())
        r["changed_row_rate"] = float(np.mean(np.abs(transformed - raw) > 1e-12))
        r["raw_pred_between_0_and_1_rate"] = float(np.mean((raw > 0) & (raw < 1)))
        rows.append(r)
    return pd.DataFrame(rows)


def _complete_3m(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["base_sku", "branch_code", "forecast_origin"]
    d = df.copy()
    d["horizon"] = pd.to_numeric(d["horizon"], errors="coerce").astype("Int64")
    stats = d.groupby(keys, dropna=False)["horizon"].agg(["count", "nunique", "min", "max", "sum"]).reset_index()
    keep = stats.loc[
        stats["count"].eq(3) & stats["nunique"].eq(3) & stats["min"].eq(1) & stats["max"].eq(3) & stats["sum"].eq(6), keys
    ]
    return d.merge(keep, on=keys, how="inner")


def compare_3m_rounding(df: pd.DataFrame, pred_col: str = CORE_PRED) -> pd.DataFrame:
    complete = _complete_3m(df)
    if complete.empty:
        return pd.DataFrame()
    levels = {
        "PAIR_3M": ["base_sku", "branch_code", "forecast_origin"],
        "BASE_SKU_3M": ["base_sku", "forecast_origin"],
        "BRANCH_3M": ["branch_code", "forecast_origin"],
        "PORTFOLIO_3M": ["forecast_origin"],
    }
    rows = []
    for policy in ["raw", "round_nearest_1m2", "ceil_1m2"]:
        d = complete.copy()
        d["_pred_row"] = _round_policy_series(d[pred_col], policy)
        for level, keys in levels.items():
            agg = d.groupby(keys, as_index=False, dropna=False)[[TARGET, "_pred_row"]].sum()
            r = _metric_row(agg.rename(columns={"_pred_row": "_tmp_pred"}), "_tmp_pred", level=level, policy=f"{policy}_before_aggregate")
            rows.append(r)

    # Preferred display-only alternatives: aggregate raw prediction first, then round one business total.
    for agg_policy in ["round_nearest_1m2", "ceil_1m2"]:
        for level, keys in levels.items():
            agg = complete.groupby(keys, as_index=False, dropna=False)[[TARGET, pred_col]].sum()
            agg["_tmp_pred"] = _round_policy_series(agg[pred_col], agg_policy)
            rows.append(_metric_row(agg, "_tmp_pred", level=level, policy=f"{agg_policy}_after_aggregate"))
    return pd.DataFrame(rows)


def actual_m2_granularity(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    pos = y.loc[y > 0]
    if pos.empty:
        return pd.DataFrame(), pd.DataFrame()

    def share_on_grid(step: float) -> float:
        scaled = pos / step
        return float(np.mean(np.isclose(scaled, np.round(scaled), atol=1e-8)))

    summary = pd.DataFrame([{
        "n_positive_actual_rows": int(len(pos)),
        "actual_sum_m2": float(pos.sum()),
        "share_exact_integer_m2": share_on_grid(1.0),
        "share_on_0_1_m2_grid": share_on_grid(0.1),
        "share_on_0_01_m2_grid": share_on_grid(0.01),
        "share_on_0_001_m2_grid": share_on_grid(0.001),
        "min_positive_m2": float(pos.min()),
        "p01_positive_m2": float(pos.quantile(0.01)),
        "p05_positive_m2": float(pos.quantile(0.05)),
        "median_positive_m2": float(pos.median()),
        "p95_positive_m2": float(pos.quantile(0.95)),
    }])

    frac = np.mod(pos.to_numpy(dtype=float), 1.0)
    frac = np.where(np.isclose(frac, 1.0, atol=1e-9), 0.0, frac)
    frac3 = pd.Series(np.round(frac, 3), name="fractional_part_3dp")
    freq = frac3.value_counts(dropna=False).rename_axis("fractional_part_3dp").reset_index(name="n_rows")
    freq["row_rate"] = freq["n_rows"] / len(frac3)
    return summary, freq.head(50)


def run_diagnosis(rolling_prediction_path: str, rolling_pointer: dict, contract_path: str, report_dir: str) -> dict:
    pred_path = Path(rolling_prediction_path)
    contract_file = Path(contract_path)
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(contract_file.read_text(encoding="utf-8"))

    df = pd.read_parquet(pred_path)
    required = {
        "base_sku", "branch_code", "forecast_origin", "target_month", "horizon", TARGET,
        CORE_PRED, "behavior_segment", "pair_positive_count_to_origin",
        "months_since_last_positive", "pair_mean_positive_m2_to_origin",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Rolling predictions missing diagnosis columns: {missing}")

    if rolling_pointer.get("status") != "PASS":
        raise ValueError("Rolling pointer must be PASS")
    if rolling_pointer.get("backtest_version") != cfg["lineage"]["required_backtest_version"]:
        raise ValueError("Rolling backtest version mismatch")

    under = build_underforecast_diagnosis(df, cfg)
    origin_event = build_origin_event_diagnosis(df, cfg)
    round_month = compare_pair_month_rounding(df)
    round_3m = compare_3m_rounding(df)
    gran_summary, gran_freq = actual_m2_granularity(df)

    under.to_csv(out_dir / "underforecast_diagnosis.csv", index=False)
    origin_event.to_csv(out_dir / "underforecast_by_origin_event.csv", index=False)
    round_month.to_csv(out_dir / "rounding_pair_month_comparison.csv", index=False)
    round_3m.to_csv(out_dir / "rounding_3m_comparison.csv", index=False)
    gran_summary.to_csv(out_dir / "actual_m2_granularity_summary.csv", index=False)
    gran_freq.to_csv(out_dir / "actual_m2_fraction_frequency.csv", index=False)

    overall = under.loc[(under["dimension"] == "OVERALL") & (under["bucket"] == "ALL")].iloc[0].to_dict()
    event_only = under.loc[under["dimension"] == "DEMAND_EVENT"].sort_values("underforecast_m2", ascending=False)
    top_events = event_only.head(10)[[
        "bucket", "n_rows", "actual_sum_m2", "bias_ratio", "underforecast_m2",
        "share_of_total_underforecast"
    ]].to_dict("records")

    manifest = {
        "run_type": "UNDERFORECAST_DIAGNOSIS_V01",
        "status": "PASS",
        "diagnosis_version": DIAG_VERSION,
        "source_rolling_run_id": rolling_pointer.get("run_id"),
        "source_rolling_backtest_version": rolling_pointer.get("backtest_version"),
        "evaluation_rows": int(len(df)),
        "core_model": CORE_MODEL,
        "overall": overall,
        "top_underforecast_events": top_events,
        "rounding_policy": {
            "training_target_rounded": False,
            "actual_evaluation_rounded": False,
            "forecast_rounding_is_diagnostic_only": True,
            "policies_tested": ["raw", "round_nearest_1m2", "ceil_1m2"],
            "preferred_business_display_rule": "round_after_aggregation_if_integer_M2_display_is_required",
        },
        "safety": {
            "supabase_accessed": False,
            "frozen_test_touched": False,
            "model_retrained": False,
            "model_freeze_run": False,
            "production_published": False,
        },
        "input_sha256": {
            "rolling_predictions": sha256_file(pred_path),
            "diagnosis_contract": sha256_file(contract_file),
        },
    }
    (out_dir / "diagnosis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest
