from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

MODEL_VERSION = "pair_modeling_v02"
PAIR_TARGET = "target_actual_gross_m2"

REVISION_SUMMARY_COLUMNS = [
    "model", "level", "transition", "n_units", "revision_mae_m2",
    "revision_ratio_vs_old_forecast", "signed_revision_m2",
]
REVISION_DETAIL_COLUMNS = [
    "base_sku", "branch_code", "target_month", "forecast_origin_old",
    "forecast_origin_new", "transition", "model", "old_forecast_m2",
    "new_forecast_m2", "revision_m2",
]


@dataclass
class DeviceResult:
    library: str
    requested: str
    actual: str
    fallback_reason: Optional[str] = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float((y_pred - y_true).sum())


def bias_ratio(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    denom = np.abs(y_true).sum()
    return float(bias(y_true, y_pred) / denom) if denom > 0 else float("nan")


def zero_false_positive_rate(y_true, y_pred, threshold: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true <= threshold
    return float(np.mean(y_pred[mask] > threshold)) if mask.any() else float("nan")


def positive_wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return wape(y_true[mask], y_pred[mask]) if mask.any() else float("nan")


def metric_row(name: str, y_true, y_pred, horizon=None, segment=None, universe="PRIMARY") -> dict:
    return {
        "model": name,
        "universe": universe,
        "horizon": horizon,
        "segment": segment,
        "n_rows": int(len(y_true)),
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "bias_ratio": bias_ratio(y_true, y_pred),
        "zero_false_positive_rate": zero_false_positive_rate(y_true, y_pred),
        "positive_wape": positive_wape(y_true, y_pred),
    }


def load_selected_features(path: Path) -> List[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    feats = data.get("selected_features", [])
    if not feats:
        raise ValueError("selected_features is empty")
    return list(dict.fromkeys(feats))


def validate_feature_safety(features: Sequence[str], contract: dict) -> List[str]:
    exact = set(contract["feature_safety"].get("forbidden_exact", []))
    contains = tuple(contract["feature_safety"].get("forbidden_contains", []))
    bad = [f for f in features if f in exact or any(x in f for x in contains)]
    if bad:
        raise ValueError(f"Forbidden/leaky features selected: {bad}")
    return list(features)


def split_universes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "historical_train_mask", "official_validation_mask", "target_available",
        "current_production_forecast_mask", "known_pair_asof_origin", "horizon", PAIR_TARGET,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train = df.loc[df["historical_train_mask"].fillna(False).astype(bool) & df["target_available"].fillna(False).astype(bool)].copy()
    val_all = df.loc[df["official_validation_mask"].fillna(False).astype(bool) & df["target_available"].fillna(False).astype(bool)].copy()
    primary_mask = (
        val_all["current_production_forecast_mask"].fillna(False).astype(bool)
        & val_all["known_pair_asof_origin"].fillna(False).astype(bool)
    )
    val_primary = val_all.loc[primary_mask].copy()
    if train.empty or val_primary.empty:
        raise ValueError("Train or primary validation is empty")
    return train, val_primary, val_all


def nvidia_gpu_available() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def _prepare_frame(df: pd.DataFrame, features: Sequence[str], direct_horizon: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feats = [f for f in features if f in df.columns]
    if direct_horizon and "horizon" in feats:
        feats.remove("horizon")
    if not feats:
        raise ValueError("No usable model features found in panel")
    X = df[feats].copy()
    categorical = []
    for c in feats:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")
            categorical.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, feats, categorical


def _lgbm_params(contract: dict, device: str, binary: bool = False) -> dict:
    base = dict(contract["lightgbm_tweedie"])
    early = base.pop("early_stopping_rounds", 120)
    max_bin_gpu = base.pop("max_bin_gpu", 63)
    if binary:
        base.pop("tweedie_variance_power", None)
        base["objective"] = "binary"
    base.update({"random_state": contract.get("seed", 42), "verbosity": -1, "device_type": device})
    if device in ("gpu", "cuda"):
        base["max_bin"] = max_bin_gpu
    base["_early_stopping_rounds"] = early
    return base


def fit_lightgbm_direct(X_train, y_train, X_val, y_val, contract: dict, binary: bool = False):
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

    requested = contract["gpu"].get("lightgbm_device_order", ["cuda", "gpu", "cpu"])
    if not contract["gpu"].get("prefer_gpu", True) or not nvidia_gpu_available():
        requested = ["cpu"]
    errors = []
    for device in requested:
        params = _lgbm_params(contract, device, binary=binary)
        es = params.pop("_early_stopping_rounds")
        try:
            cls = LGBMClassifier if binary else LGBMRegressor
            model = cls(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(es, verbose=False)])
            return model, DeviceResult("lightgbm", requested[0], device, None if not errors else " | ".join(errors))
        except Exception as e:
            errors.append(f"{device}:{type(e).__name__}:{str(e)[:180]}")
    raise RuntimeError("LightGBM failed on all devices: " + " | ".join(errors))


def _catboost_params(contract: dict, task_type: str) -> dict:
    p = dict(contract["catboost_tweedie"])
    es = p.pop("early_stopping_rounds", 120)
    verbose = p.pop("verbose", 100)
    p.update({"random_seed": contract.get("seed", 42), "task_type": task_type, "allow_writing_files": False})
    if task_type == "GPU":
        p["devices"] = str(contract["gpu"].get("gpu_device_id", 0))
    p["_early_stopping_rounds"] = es
    p["_verbose"] = verbose
    return p


def catboost_target_scale(y_train, contract: dict) -> float:
    cfg = contract.get("catboost_target_scaling", {})
    if not cfg.get("enabled", True):
        return 1.0
    y = np.asarray(y_train, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 1.0
    method = str(cfg.get("method", "max_train_to_one"))
    if method != "max_train_to_one":
        raise ValueError(f"Unsupported CatBoost target scaling method: {method}")
    scale = float(np.max(np.maximum(finite, 0.0)))
    return max(scale, float(cfg.get("min_scale", 1.0)))


def fit_catboost_direct(X_train, y_train, X_val, y_val, cat_features: Sequence[str], contract: dict):
    from catboost import CatBoostRegressor

    requested = contract["gpu"].get("catboost_device_order", ["GPU", "CPU"])
    if not contract["gpu"].get("prefer_gpu", True) or not nvidia_gpu_available():
        requested = ["CPU"]
    errors = []
    target_scale = catboost_target_scale(y_train, contract)
    y_train_scaled = np.asarray(y_train, dtype=float) / target_scale
    y_val_scaled = np.asarray(y_val, dtype=float) / target_scale
    for task_type in requested:
        p = _catboost_params(contract, task_type)
        es, verbose = p.pop("_early_stopping_rounds"), p.pop("_verbose")
        try:
            Xt, Xv = X_train.copy(), X_val.copy()
            cat_idx = []
            for c in cat_features:
                if c in Xt.columns:
                    Xt[c] = Xt[c].astype(str).fillna("__NA__")
                    Xv[c] = Xv[c].astype(str).fillna("__NA__")
                    cat_idx.append(Xt.columns.get_loc(c))
            model = CatBoostRegressor(**p)
            model.fit(
                Xt, y_train_scaled,
                eval_set=(Xv, y_val_scaled),
                cat_features=cat_idx,
                early_stopping_rounds=es,
                verbose=verbose,
            )
            return (
                model,
                DeviceResult("catboost", requested[0], task_type, None if not errors else " | ".join(errors)),
                Xt,
                Xv,
                target_scale,
            )
        except Exception as e:
            errors.append(f"{task_type}:{type(e).__name__}:{str(e)[:180]}")
    raise RuntimeError("CatBoost failed on all devices: " + " | ".join(errors))


def predict_nonnegative(model, X) -> np.ndarray:
    return np.maximum(np.asarray(model.predict(X), dtype=float), 0.0)


def baseline_predictions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    lag1 = pd.to_numeric(df.get("pair_lag_1"), errors="coerce").fillna(0.0).to_numpy()
    lag2 = pd.to_numeric(df.get("pair_lag_2"), errors="coerce").fillna(0.0).to_numpy()
    lag3 = pd.to_numeric(df.get("pair_lag_3"), errors="coerce").fillna(0.0).to_numpy()
    lag12 = pd.to_numeric(df.get("pair_lag_12"), errors="coerce").fillna(0.0).to_numpy()
    out["naive_1"] = np.maximum(lag1, 0.0)
    out["seasonal_naive_12"] = np.maximum(lag12, 0.0)
    out["moving_average_3"] = np.maximum(np.nanmean(np.column_stack([lag1, lag2, lag3]), axis=1), 0.0)

    return out


def croston_sba_forecast(series: Sequence[float], alpha: float = 0.1) -> float:
    y = np.maximum(np.asarray(series, dtype=float), 0.0)
    if y.size == 0 or not np.any(y > 0):
        return 0.0
    nz = np.flatnonzero(y > 0)
    first = int(nz[0])
    z = float(y[first])
    p = float(first + 1)
    interval = 1.0
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            z = alpha * float(y[t]) + (1.0 - alpha) * z
            p = alpha * interval + (1.0 - alpha) * p
            interval = 1.0
        else:
            interval += 1.0
    return max((1.0 - alpha / 2.0) * z / max(p, 1e-12), 0.0)


def tsb_forecast(series: Sequence[float], alpha: float = 0.1, beta: float = 0.1) -> float:
    y = np.maximum(np.asarray(series, dtype=float), 0.0)
    if y.size == 0 or not np.any(y > 0):
        return 0.0
    nz = np.flatnonzero(y > 0)
    first = int(nz[0])
    z = float(y[first])
    prob = 1.0 / float(first + 1)
    for t in range(first + 1, len(y)):
        occurrence = 1.0 if y[t] > 0 else 0.0
        prob = prob + beta * (occurrence - prob)
        if occurrence > 0:
            z = z + alpha * (float(y[t]) - z)
    return max(prob * z, 0.0)


def intermittent_baselines_from_panel(validation_df: pd.DataFrame, pair_panel_path: str, alpha: float = 0.1, beta: float = 0.1) -> pd.DataFrame:
    required_keys = ["base_sku", "branch_code", "forecast_origin"]
    missing = [c for c in required_keys if c not in validation_df.columns]
    if missing:
        raise ValueError(f"Validation data missing keys for intermittent baselines: {missing}")
    panel = pd.read_parquet(pair_panel_path, columns=["base_sku", "branch_code", "month", "target_available", "actual_gross_m2"])
    panel["month"] = pd.to_datetime(panel["month"])
    panel = panel.loc[panel["target_available"].fillna(False).astype(bool)].copy()
    panel["actual_gross_m2"] = pd.to_numeric(panel["actual_gross_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = {k: g.sort_values("month") for k, g in panel.groupby(["base_sku", "branch_code"], sort=False)}
    req = validation_df[required_keys].drop_duplicates().copy()
    req["forecast_origin"] = pd.to_datetime(req["forecast_origin"])
    rows = []
    for r in req.itertuples(index=False):
        g = grouped.get((r.base_sku, r.branch_code))
        if g is None:
            values = np.asarray([], dtype=float)
        else:
            values = g.loc[g["month"] <= r.forecast_origin, "actual_gross_m2"].to_numpy(dtype=float)
        rows.append({
            "base_sku": r.base_sku,
            "branch_code": r.branch_code,
            "forecast_origin": r.forecast_origin,
            "pred_croston_sba": croston_sba_forecast(values, alpha=alpha),
            "pred_tsb": tsb_forecast(values, alpha=alpha, beta=beta),
        })
    states = pd.DataFrame(rows)
    left = validation_df[required_keys].copy()
    left["forecast_origin"] = pd.to_datetime(left["forecast_origin"])
    merged = left.merge(states, on=required_keys, how="left")
    return merged[["pred_croston_sba", "pred_tsb"]].fillna(0.0)


def behavior_segment(df: pd.DataFrame, contract: dict) -> pd.Series:
    """Three operational buckets using both ADI and CV².

    - very_sparse: insufficient positive history or undefined demand-state statistics.
    - regular: sufficient history AND ADI < threshold AND CV² < threshold.
    - intermittent: sufficient history but either ADI or CV² is high. This bucket intentionally
      combines the classic intermittent/erratic/lumpy non-regular states for expert routing.
    """
    cfg = contract["behavior"]
    adi = pd.to_numeric(df.get("pair_adi_target_available_to_origin"), errors="coerce")
    cv2 = pd.to_numeric(df.get("pair_cv2_positive_to_origin"), errors="coerce")
    pos_count = pd.to_numeric(df.get("pair_positive_count_to_origin"), errors="coerce").fillna(0)
    sparse_max = int(cfg.get("very_sparse_max_positive_count", 1))
    adi_thr = float(cfg.get("adi_threshold", 1.32))
    cv2_thr = float(cfg.get("cv2_threshold", 0.49))

    finite_adi = adi.notna() & np.isfinite(adi)
    finite_cv2 = cv2.notna() & np.isfinite(cv2)
    sufficient = pos_count > sparse_max

    seg = pd.Series("very_sparse", index=df.index, dtype="object")
    regular = sufficient & finite_adi & finite_cv2 & adi.lt(adi_thr) & cv2.lt(cv2_thr)
    irregular = sufficient & finite_adi & finite_cv2 & ~regular
    seg.loc[regular] = "regular"
    seg.loc[irregular] = "intermittent"
    return seg


def calibrate_hurdle_threshold(y_true, p_pos, pos_pred, contract: dict) -> Tuple[float, np.ndarray, pd.DataFrame]:
    cfg = contract.get("hurdle", {}).get("threshold_calibration", {})
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pos, dtype=float), 0.0, 1.0)
    positive = np.maximum(np.asarray(pos_pred, dtype=float), 0.0)
    expected = p * positive
    if not cfg.get("enabled", True):
        pred = expected
        return 0.0, pred, pd.DataFrame([{
            "threshold": 0.0,
            "wape": wape(y, pred),
            "zero_false_positive_rate": zero_false_positive_rate(y, pred),
        }])

    grid = cfg.get("grid", [round(x * 0.05, 2) for x in range(0, 20)])
    rows = []
    for t in sorted({float(x) for x in grid}):
        pred = np.where(p < t, 0.0, expected)
        rows.append({
            "threshold": t,
            "wape": wape(y, pred),
            "zero_false_positive_rate": zero_false_positive_rate(y, pred),
            "mae": mae(y, pred),
            "bias_ratio": bias_ratio(y, pred),
        })
    report = pd.DataFrame(rows)
    finite = report.loc[np.isfinite(report["wape"])].copy()
    if finite.empty:
        return 0.0, expected, report
    # Primary objective WAPE; zero-demand false-positive rate is an explicit tie-break.
    best_wape = float(finite["wape"].min())
    tol = float(cfg.get("wape_tie_tolerance", 1e-9))
    near = finite.loc[finite["wape"] <= best_wape + tol].copy()
    near["_zero_sort"] = near["zero_false_positive_rate"].fillna(np.inf)
    chosen = near.sort_values(["_zero_sort", "threshold"], ascending=[True, False]).iloc[0]
    threshold = float(chosen["threshold"])
    pred = np.where(p < threshold, 0.0, expected)
    return threshold, pred, report


def _branch_aggregate_metric(pred_df: pd.DataFrame, pred_col: str) -> dict:
    keys = [c for c in ["branch_code", "target_month"] if c in pred_df.columns]
    if len(keys) < 2:
        return {"branch_aggregate_wape": float("nan")}
    agg = pred_df.groupby(keys, dropna=False)[[PAIR_TARGET, pred_col]].sum().reset_index()
    return {"branch_aggregate_wape": wape(agg[PAIR_TARGET], agg[pred_col])}


def _select_gate(pred_df: pd.DataFrame, expert_cols: Dict[str, str], contract: dict) -> Tuple[pd.DataFrame, pd.Series]:
    df = pred_df.copy()
    df["behavior_segment"] = behavior_segment(df, contract)
    min_rows = int(contract["behavior"].get("min_segment_validation_rows", 50))
    gate_rows = []
    final = pd.Series(index=df.index, dtype=float)
    allowed_key = {
        "regular": "regular_experts",
        "intermittent": "intermittent_experts",
        "very_sparse": "very_sparse_experts",
    }
    for h in sorted(df["horizon"].dropna().astype(int).unique()):
        hmask = df["horizon"].astype(int) == h
        hsub = df.loc[hmask]
        for seg in ["regular", "intermittent", "very_sparse"]:
            mask = hmask & (df["behavior_segment"] == seg)
            sub = df.loc[mask]
            allowed = contract["behavior"].get(allowed_key[seg], list(expert_cols))
            candidates = [e for e in allowed if e in expert_cols and expert_cols[e] in df.columns]
            if not candidates:
                raise ValueError(f"No eligible experts for behavior segment={seg}, horizon={h}")

            if len(sub) >= min_rows:
                selection_scope = "SEGMENT_PRIMARY"
                selection_df = sub
            else:
                # Small segments cannot support stable segment-specific ranking. Choose the best
                # allowed expert on the whole Primary horizon instead of an arbitrary fixed model.
                selection_scope = "HORIZON_PRIMARY_FALLBACK"
                selection_df = hsub

            scores = [(e, wape(selection_df[PAIR_TARGET], selection_df[expert_cols[e]])) for e in candidates]
            finite = [(e, s) for e, s in scores if np.isfinite(s)]
            if not finite:
                raise ValueError(f"No finite gate score for segment={seg}, horizon={h}")
            chosen, chosen_selection_wape = min(finite, key=lambda x: x[1])
            final.loc[mask] = df.loc[mask, expert_cols[chosen]].to_numpy()
            segment_wape = wape(sub[PAIR_TARGET], sub[expert_cols[chosen]]) if len(sub) else float("nan")
            gate_rows.append({
                "horizon": h,
                "behavior_segment": seg,
                "n_rows": int(mask.sum()),
                "selection_scope": selection_scope,
                "chosen_expert": chosen,
                "selection_wape": chosen_selection_wape,
                "segment_wape": segment_wape,
            })
    return pd.DataFrame(gate_rows), final.fillna(0.0)


def build_complete_3m_pair_windows(pred_df: pd.DataFrame, pred_cols: Dict[str, str]) -> Tuple[pd.DataFrame, dict]:
    keys = ["base_sku", "branch_code", "forecast_origin"]
    required = keys + ["horizon", PAIR_TARGET]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"3M cumulative evaluation missing columns: {missing}")
    dup_keys = keys + ["horizon"]
    if pred_df.duplicated(dup_keys).any():
        raise ValueError("Duplicate Pair × forecast_origin × horizon rows in 3M evaluation")

    d = pred_df.copy()
    d["horizon"] = d["horizon"].astype(int)
    stats = d.groupby(keys, dropna=False)["horizon"].agg(["count", "nunique", "min", "max", "sum"]).reset_index()
    complete_keys = stats.loc[
        stats["count"].eq(3) & stats["nunique"].eq(3) & stats["min"].eq(1) & stats["max"].eq(3) & stats["sum"].eq(6),
        keys,
    ]
    complete = d.merge(complete_keys, on=keys, how="inner")
    value_cols = [PAIR_TARGET] + [c for c in pred_cols.values() if c in complete.columns]
    pair3 = complete.groupby(keys, as_index=False, dropna=False)[value_cols].sum()
    pair3 = pair3.rename(columns={PAIR_TARGET: "actual_3m_m2", **{c: f"{c}_3m" for c in pred_cols.values() if c in pair3.columns}})
    coverage = {
        "pair_origin_total": int(len(stats)),
        "pair_origin_complete_h1_h2_h3": int(len(pair3)),
        "pair_origin_incomplete": int(len(stats) - len(pair3)),
        "complete_rate": float(len(pair3) / len(stats)) if len(stats) else float("nan"),
    }
    return pair3, coverage


def cumulative_3m_scoreboard(pair3: pd.DataFrame, pred_cols: Dict[str, str]) -> pd.DataFrame:
    if pair3.empty:
        return pd.DataFrame(columns=["model", "level", "n_units", "wape_3m", "mae_3m", "bias_3m", "bias_ratio_3m", "actual_sum_m2", "forecast_sum_m2"])
    level_keys = {
        "PAIR": ["base_sku", "branch_code", "forecast_origin"],
        "BASE_SKU": ["base_sku", "forecast_origin"],
        "BRANCH": ["branch_code", "forecast_origin"],
        "PORTFOLIO": ["forecast_origin"],
    }
    rows = []
    for level, keys in level_keys.items():
        value_cols = ["actual_3m_m2"] + [f"{c}_3m" for c in pred_cols.values() if f"{c}_3m" in pair3.columns]
        if level == "PAIR":
            agg = pair3[keys + value_cols].copy()
        else:
            agg = pair3.groupby(keys, as_index=False, dropna=False)[value_cols].sum()
        y = agg["actual_3m_m2"].to_numpy(dtype=float)
        for model, pred_col in pred_cols.items():
            col3 = f"{pred_col}_3m"
            if col3 not in agg.columns:
                continue
            p = agg[col3].to_numpy(dtype=float)
            rows.append({
                "model": model,
                "level": level,
                "n_units": int(len(agg)),
                "wape_3m": wape(y, p),
                "mae_3m": mae(y, p),
                "bias_3m": bias(y, p),
                "bias_ratio_3m": bias_ratio(y, p),
                "actual_sum_m2": float(np.sum(y)),
                "forecast_sum_m2": float(np.sum(np.maximum(p, 0.0))),
            })
    return pd.DataFrame(rows)


def forecast_revision_scoreboard(pred_df: pd.DataFrame, pred_cols: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["base_sku", "branch_code", "forecast_origin", "target_month", "horizon"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Forecast revision evaluation missing columns: {missing}")
    d = pred_df.copy()
    d["forecast_origin"] = pd.to_datetime(d["forecast_origin"])
    d["target_month"] = pd.to_datetime(d["target_month"])
    details = []
    summary = []

    for old_h, new_h, label in [(2, 1, "H2_TO_H1"), (3, 2, "H3_TO_H2")]:
        old = d.loc[d["horizon"].astype(int).eq(old_h)].copy()
        new = d.loc[d["horizon"].astype(int).eq(new_h)].copy()
        keys = ["base_sku", "branch_code", "target_month"]
        keep_old = keys + ["forecast_origin"] + [c for c in pred_cols.values() if c in old.columns]
        keep_new = keys + ["forecast_origin"] + [c for c in pred_cols.values() if c in new.columns]
        m = new[keep_new].merge(old[keep_old], on=keys, suffixes=("_new", "_old"), how="inner")
        if not m.empty:
            month_delta = (
                (m["forecast_origin_new"].dt.year - m["forecast_origin_old"].dt.year) * 12
                + (m["forecast_origin_new"].dt.month - m["forecast_origin_old"].dt.month)
            )
            m = m.loc[month_delta.eq(1)].copy()
        if m.empty:
            continue
        m["transition"] = label
        for model, col in pred_cols.items():
            new_col, old_col = f"{col}_new", f"{col}_old"
            if new_col not in m.columns or old_col not in m.columns:
                continue
            newp = pd.to_numeric(m[new_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            oldp = pd.to_numeric(m[old_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            rev = newp - oldp
            denom = float(np.abs(oldp).sum())
            summary.append({
                "model": model,
                "level": "PAIR",
                "transition": label,
                "n_units": int(len(m)),
                "revision_mae_m2": float(np.mean(np.abs(rev))),
                "revision_ratio_vs_old_forecast": float(np.abs(rev).sum() / denom) if denom > 0 else float("nan"),
                "signed_revision_m2": float(rev.sum()),
            })
            part = m[keys + ["forecast_origin_old", "forecast_origin_new", "transition"]].copy()
            part["model"] = model
            part["old_forecast_m2"] = oldp
            part["new_forecast_m2"] = newp
            part["revision_m2"] = rev
            details.append(part)

        # Base-SKU revision after summing branches for the same target and origin pair.
        for model, col in pred_cols.items():
            new_col, old_col = f"{col}_new", f"{col}_old"
            if new_col not in m.columns or old_col not in m.columns:
                continue
            sku = m.groupby(["base_sku", "target_month", "forecast_origin_old", "forecast_origin_new"], as_index=False)[[new_col, old_col]].sum()
            newp = sku[new_col].to_numpy(dtype=float)
            oldp = sku[old_col].to_numpy(dtype=float)
            rev = newp - oldp
            denom = float(np.abs(oldp).sum())
            summary.append({
                "model": model,
                "level": "BASE_SKU",
                "transition": label,
                "n_units": int(len(sku)),
                "revision_mae_m2": float(np.mean(np.abs(rev))),
                "revision_ratio_vs_old_forecast": float(np.abs(rev).sum() / denom) if denom > 0 else float("nan"),
                "signed_revision_m2": float(rev.sum()),
            })
    detail_df = (
        pd.concat(details, ignore_index=True)
        if details
        else pd.DataFrame(columns=REVISION_DETAIL_COLUMNS)
    )
    summary_df = (
        pd.DataFrame(summary)
        if summary
        else pd.DataFrame(columns=REVISION_SUMMARY_COLUMNS)
    )
    return summary_df, detail_df


def run_pair_model_stage(
    pair_feature_path: str,
    pair_panel_path: str,
    selected_feature_path: str,
    contract_path: str,
    output_dir: str,
    run_id: str,
    save_models: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    selected = validate_feature_safety(load_selected_features(Path(selected_feature_path)), contract)
    df = pd.read_parquet(pair_feature_path)
    train, val_primary, val_all = split_universes(df)

    predictions = val_primary[[c for c in [
        "base_sku", "branch_code", "forecast_origin", "target_month", "horizon", PAIR_TARGET,
        "pair_adi_target_available_to_origin", "pair_cv2_positive_to_origin", "pair_positive_count_to_origin",
    ] if c in val_primary.columns]].copy()
    base_preds = baseline_predictions(val_primary)
    for name, values in base_preds.items():
        predictions[f"pred_{name}"] = values

    ib_cfg = contract.get("intermittent_baselines", {})
    ib = intermittent_baselines_from_panel(
        val_primary, pair_panel_path,
        alpha=float(ib_cfg.get("alpha", 0.1)),
        beta=float(ib_cfg.get("beta", 0.1)),
    )
    for col in ["pred_croston_sba", "pred_tsb"]:
        predictions[col] = ib[col].to_numpy()
    base_preds["croston_sba"] = predictions["pred_croston_sba"].to_numpy()
    base_preds["tsb"] = predictions["pred_tsb"].to_numpy()

    scoreboard = []
    devices = []
    catboost_scales = []
    hurdle_thresholds = []
    threshold_reports = []
    expert_cols = {k: f"pred_{k}" for k in base_preds}

    for name, values in base_preds.items():
        scoreboard.append(metric_row(name, predictions[PAIR_TARGET], values, universe="PRIMARY"))
        for h in [1, 2, 3]:
            m = predictions["horizon"].astype(int) == h
            scoreboard.append(metric_row(name, predictions.loc[m, PAIR_TARGET], predictions.loc[m, f"pred_{name}"], horizon=h, universe="PRIMARY"))

    for h in [1, 2, 3]:
        tr = train.loc[train["horizon"].astype(int) == h].copy()
        va = val_primary.loc[val_primary["horizon"].astype(int) == h].copy()
        if tr.empty or va.empty:
            raise ValueError(f"Empty direct horizon split H{h}")
        Xtr, feats, cats = _prepare_frame(tr, selected, direct_horizon=True)
        Xv, _, _ = _prepare_frame(va, feats, direct_horizon=True)
        ytr = tr[PAIR_TARGET].astype(float).to_numpy()
        yv = va[PAIR_TARGET].astype(float).to_numpy()

        lgbm, dev = fit_lightgbm_direct(Xtr, ytr, Xv, yv, contract, binary=False)
        devices.append(asdict(dev) | {"horizon": h, "candidate": "lightgbm_tweedie"})
        pred_lgbm = predict_nonnegative(lgbm, Xv)
        predictions.loc[va.index, "pred_lightgbm_tweedie"] = pred_lgbm
        if save_models:
            lgbm.booster_.save_model(str(model_dir / f"lightgbm_tweedie_h{h}.txt"))

        cat, cdev, Xtr_cat, Xv_cat, target_scale = fit_catboost_direct(Xtr, ytr, Xv, yv, cats, contract)
        devices.append(asdict(cdev) | {"horizon": h, "candidate": "catboost_tweedie"})
        pred_cat = np.maximum(np.asarray(cat.predict(Xv_cat), dtype=float) * target_scale, 0.0)
        predictions.loc[va.index, "pred_catboost_tweedie"] = pred_cat
        catboost_scales.append({"horizon": h, "target_scale": float(target_scale)})
        if save_models:
            cat.save_model(str(model_dir / f"catboost_tweedie_h{h}.cbm"))
            (model_dir / f"catboost_tweedie_h{h}_meta.json").write_text(
                json.dumps({"target_scale": float(target_scale), "scaling_method": "max_train_to_one"}, indent=2),
                encoding="utf-8",
            )

        # Hurdle: occurrence classifier + positive-only Tweedie regressor + calibrated zero threshold.
        clf, hdev = fit_lightgbm_direct(Xtr, (ytr > 0).astype(int), Xv, (yv > 0).astype(int), contract, binary=True)
        devices.append(asdict(hdev) | {"horizon": h, "candidate": "hurdle_occurrence"})
        p_pos = np.asarray(clf.predict_proba(Xv)[:, 1], dtype=float)
        predictions.loc[va.index, "hurdle_p_positive"] = p_pos
        pos_mask = ytr > 0
        if int(pos_mask.sum()) >= int(contract["hurdle"].get("min_positive_train_rows", 500)):
            val_pos_mask = yv > 0
            Xv_positive_eval = Xv.loc[val_pos_mask] if np.any(val_pos_mask) else Xv
            yv_positive_eval = yv[val_pos_mask] if np.any(val_pos_mask) else yv
            preg, pdev = fit_lightgbm_direct(
                Xtr.loc[pos_mask], ytr[pos_mask], Xv_positive_eval, yv_positive_eval, contract, binary=False
            )
            devices.append(asdict(pdev) | {"horizon": h, "candidate": "hurdle_positive"})
            pos_pred = predict_nonnegative(preg, Xv)
            threshold, hurdle_pred, threshold_report = calibrate_hurdle_threshold(yv, p_pos, pos_pred, contract)
            threshold_report["horizon"] = h
            threshold_reports.append(threshold_report)
            hurdle_thresholds.append({
                "horizon": h,
                "threshold": float(threshold),
                "validation_wape": wape(yv, hurdle_pred),
                "zero_false_positive_rate": zero_false_positive_rate(yv, hurdle_pred),
            })
            if save_models:
                clf.booster_.save_model(str(model_dir / f"hurdle_occurrence_h{h}.txt"))
                preg.booster_.save_model(str(model_dir / f"hurdle_positive_h{h}.txt"))
                (model_dir / f"hurdle_h{h}_meta.json").write_text(
                    json.dumps({"probability_threshold": float(threshold)}, indent=2), encoding="utf-8"
                )
        else:
            hurdle_pred = pred_lgbm
            hurdle_thresholds.append({"horizon": h, "threshold": None, "fallback": "lightgbm_insufficient_positive_train"})
        predictions.loc[va.index, "pred_hurdle"] = hurdle_pred

    expert_cols.update({
        "lightgbm_tweedie": "pred_lightgbm_tweedie",
        "catboost_tweedie": "pred_catboost_tweedie",
        "hurdle": "pred_hurdle",
    })

    gate_df, gated_pred = _select_gate(predictions, expert_cols, contract)
    predictions["pred_behavior_gated"] = gated_pred
    predictions["behavior_segment"] = behavior_segment(predictions, contract)

    all_experts = {**expert_cols, "behavior_gated": "pred_behavior_gated"}
    for expert, col in all_experts.items():
        if col not in predictions.columns:
            continue
        scoreboard.append(metric_row(expert, predictions[PAIR_TARGET], predictions[col], universe="PRIMARY"))
        for h in [1, 2, 3]:
            m = predictions["horizon"].astype(int) == h
            scoreboard.append(metric_row(expert, predictions.loc[m, PAIR_TARGET], predictions.loc[m, col], horizon=h, universe="PRIMARY"))
        for seg in ["regular", "intermittent", "very_sparse"]:
            m = predictions["behavior_segment"] == seg
            scoreboard.append(metric_row(expert, predictions.loc[m, PAIR_TARGET], predictions.loc[m, col], segment=seg, universe="PRIMARY"))

    score_df = pd.DataFrame(scoreboard)
    score_path = out_dir / "model_scoreboard_primary.csv"
    gate_path = out_dir / "behavior_gate.csv"
    pred_path = out_dir / "primary_validation_predictions.parquet"
    score_df.to_csv(score_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    predictions.to_parquet(pred_path, index=False)

    if threshold_reports:
        threshold_df = pd.concat(threshold_reports, ignore_index=True)
    else:
        threshold_df = pd.DataFrame()
    threshold_path = out_dir / "hurdle_threshold_calibration.csv"
    threshold_df.to_csv(threshold_path, index=False)

    pair3, cumulative_coverage = build_complete_3m_pair_windows(predictions, all_experts)
    pair3_path = out_dir / "cumulative_3m_pair_predictions.parquet"
    pair3.to_parquet(pair3_path, index=False)
    cumulative_df = cumulative_3m_scoreboard(pair3, all_experts)
    cumulative_path = out_dir / "cumulative_3m_scoreboard.csv"
    cumulative_df.to_csv(cumulative_path, index=False)

    revision_df, revision_detail = forecast_revision_scoreboard(predictions, all_experts)
    revision_path = out_dir / "forecast_revision_scoreboard.csv"
    revision_detail_path = out_dir / "forecast_revision_detail.parquet"
    revision_df.to_csv(revision_path, index=False)
    revision_detail.to_parquet(revision_detail_path, index=False)

    branch_diag = _branch_aggregate_metric(predictions, "pred_behavior_gated")
    gated_3m = cumulative_df.loc[cumulative_df["model"].eq("behavior_gated")].to_dict("records") if not cumulative_df.empty else []
    manifest = {
        "run_id": run_id,
        "run_type": "PAIR_MODEL_CANDIDATE_V02",
        "status": "PASS",
        "model_version": MODEL_VERSION,
        "dataset_version": str(df.get("dataset_version", pd.Series([None])).dropna().iloc[0]) if "dataset_version" in df.columns and df["dataset_version"].notna().any() else None,
        "feature_version": str(df.get("feature_version", pd.Series([None])).dropna().iloc[0]) if "feature_version" in df.columns and df["feature_version"].notna().any() else None,
        "row_counts": {"train": int(len(train)), "validation_primary": int(len(val_primary)), "validation_secondary": int(len(val_all))},
        "selected_feature_count": int(len(selected)),
        "selected_features": selected,
        "devices": devices,
        "catboost_target_scaling": catboost_scales,
        "hurdle_thresholds": hurdle_thresholds,
        "behavior_rule": {
            "regular": "positive_count>very_sparse_max AND ADI<adi_threshold AND CV2<cv2_threshold",
            "intermittent": "sufficient_history AND finite ADI/CV2 AND NOT regular",
            "very_sparse": "insufficient positive history OR undefined ADI/CV2",
            "small_segment_gate_fallback": "best allowed expert on whole Primary horizon",
        },
        "primary_behavior_gated": metric_row("behavior_gated", predictions[PAIR_TARGET], predictions["pred_behavior_gated"]),
        "branch_diagnostic": branch_diag,
        "cumulative_3m": {
            "coverage": cumulative_coverage,
            "behavior_gated_metrics": gated_3m,
            "overlapping_windows_note": "Rolling 3M windows overlap across consecutive forecast origins and are not independent samples.",
        },
        "forecast_revision": {
            "transitions": ["H2_TO_H1", "H3_TO_H2"],
            "evaluable": bool(not revision_df.empty),
            "not_evaluable_reason": (
                None if not revision_df.empty
                else "NO_CONSECUTIVE_ORIGIN_TARGET_OVERLAP_IN_PRIMARY_VALIDATION"
            ),
            "rows": revision_df.to_dict("records") if not revision_df.empty else [],
        },
        "safety": {
            "supabase_accessed": False,
            "frozen_test_touched": False,
            "reconciliation_run": False,
            "production_published": False,
            "current_status_used_as_predictor": False,
        },
        "input_sha256": {
            "pair_feature_panel": sha256_file(Path(pair_feature_path)),
            "pair_panel": sha256_file(Path(pair_panel_path)),
            "selected_feature_list": sha256_file(Path(selected_feature_path)),
            "model_contract": sha256_file(Path(contract_path)),
        },
        "output_sha256": {
            "model_scoreboard_primary": sha256_file(score_path),
            "behavior_gate": sha256_file(gate_path),
            "primary_validation_predictions": sha256_file(pred_path),
            "hurdle_threshold_calibration": sha256_file(threshold_path),
            "cumulative_3m_pair_predictions": sha256_file(pair3_path),
            "cumulative_3m_scoreboard": sha256_file(cumulative_path),
            "forecast_revision_scoreboard": sha256_file(revision_path),
            "forecast_revision_detail": sha256_file(revision_detail_path),
        },
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest
