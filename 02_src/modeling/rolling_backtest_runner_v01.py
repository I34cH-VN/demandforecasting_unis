from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

ROLLING_BACKTEST_VERSION = "rolling_backtest_v01"
PAIR_TARGET = "target_actual_gross_m2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dependencies(work9_root: Path):
    import sys
    modeling = work9_root / "02_src" / "modeling"
    features = work9_root / "02_src" / "features"
    for p in [modeling, features]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from feature_builder_v013 import build_pair_origin_features, build_target_calendar_table
    from model_runner_v02 import (
        _prepare_frame,
        _lgbm_params,
        nvidia_gpu_available,
        baseline_predictions,
        behavior_segment,
        bias,
        bias_ratio,
        build_complete_3m_pair_windows,
        calibrate_hurdle_threshold,
        croston_sba_forecast,
        cumulative_3m_scoreboard,
        fit_lightgbm_direct,
        forecast_revision_scoreboard,
        load_selected_features,
        mae,
        metric_row,
        predict_nonnegative,
        tsb_forecast,
        validate_feature_safety,
        wape,
        zero_false_positive_rate,
    )
    return {
        "build_pair_origin_features": build_pair_origin_features,
        "build_target_calendar_table": build_target_calendar_table,
        "_prepare_frame": _prepare_frame,
        "_lgbm_params": _lgbm_params,
        "nvidia_gpu_available": nvidia_gpu_available,
        "baseline_predictions": baseline_predictions,
        "behavior_segment": behavior_segment,
        "bias": bias,
        "bias_ratio": bias_ratio,
        "build_complete_3m_pair_windows": build_complete_3m_pair_windows,
        "calibrate_hurdle_threshold": calibrate_hurdle_threshold,
        "croston_sba_forecast": croston_sba_forecast,
        "cumulative_3m_scoreboard": cumulative_3m_scoreboard,
        "fit_lightgbm_direct": fit_lightgbm_direct,
        "forecast_revision_scoreboard": forecast_revision_scoreboard,
        "load_selected_features": load_selected_features,
        "mae": mae,
        "metric_row": metric_row,
        "predict_nonnegative": predict_nonnegative,
        "tsb_forecast": tsb_forecast,
        "validate_feature_safety": validate_feature_safety,
        "wape": wape,
        "zero_false_positive_rate": zero_false_positive_rate,
    }


def _primary_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["target_available"].fillna(False).astype(bool)
        & df["current_production_forecast_mask"].fillna(False).astype(bool)
        & df["known_pair_asof_origin"].fillna(False).astype(bool)
    )


def precompute_intermittent_states(pair_panel: pd.DataFrame, alpha: float = 0.1, beta: float = 0.1) -> pd.DataFrame:
    """Exact monthly Croston-SBA / TSB states using only target-available history <= origin.

    Missing/unavailable months do not advance the intermittent-demand state, matching the V0.2
    baseline implementation that first filters history to target_available rows.
    """
    required = {"base_sku", "branch_code", "month", "target_available", "actual_gross_m2"}
    missing = required - set(pair_panel.columns)
    if missing:
        raise ValueError(f"pair_panel missing intermittent-state columns: {sorted(missing)}")

    d = pair_panel[["base_sku", "branch_code", "month", "target_available", "actual_gross_m2"]].copy()
    d["month"] = pd.to_datetime(d["month"])
    d = d.sort_values(["base_sku", "branch_code", "month"]).reset_index(drop=True)

    out_rows = []
    for (sku, branch), g in d.groupby(["base_sku", "branch_code"], sort=False):
        initialized = False
        avail_count = 0
        z_c = p_c = interval = 0.0
        z_t = prob_t = 0.0
        for r in g.itertuples(index=False):
            available = bool(r.target_available) if pd.notna(r.target_available) else False
            if available:
                y = 0.0 if pd.isna(r.actual_gross_m2) else max(float(r.actual_gross_m2), 0.0)
                if not initialized:
                    avail_count += 1
                    if y > 0:
                        initialized = True
                        z_c = y
                        p_c = float(avail_count)
                        interval = 1.0
                        z_t = y
                        prob_t = 1.0 / float(avail_count)
                else:
                    if y > 0:
                        z_c = alpha * y + (1.0 - alpha) * z_c
                        p_c = alpha * interval + (1.0 - alpha) * p_c
                        interval = 1.0
                    else:
                        interval += 1.0
                    occurrence = 1.0 if y > 0 else 0.0
                    prob_t = prob_t + beta * (occurrence - prob_t)
                    if occurrence > 0:
                        z_t = z_t + alpha * (y - z_t)

            croston = (1.0 - alpha / 2.0) * z_c / max(p_c, 1e-12) if initialized else 0.0
            tsb = prob_t * z_t if initialized else 0.0
            out_rows.append({
                "base_sku": sku,
                "branch_code": branch,
                "forecast_origin": pd.Timestamp(r.month),
                "pred_croston_sba": max(float(croston), 0.0),
                "pred_tsb": max(float(tsb), 0.0),
            })
    return pd.DataFrame(out_rows)


def build_rolling_supervised_panel(
    pair_panel_path: str,
    frozen_test_start: str,
    work9_root: str,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> pd.DataFrame:
    deps = _load_dependencies(Path(work9_root))
    pair_panel = pd.read_parquet(pair_panel_path)
    pair_panel["month"] = pd.to_datetime(pair_panel["month"]).dt.to_period("M").dt.to_timestamp()

    origin = deps["build_pair_origin_features"](pair_panel)
    target_cols = [
        "base_sku", "branch_code", "month", "actual_gross_m2", "actual_observed",
        "actual_positive", "actual_negative_only", "target_available", "zero_semantics",
    ]
    target = pair_panel[[c for c in target_cols if c in pair_panel.columns]].copy().rename(columns={
        "month": "target_month",
        "actual_gross_m2": "target_actual_gross_m2",
        "actual_observed": "target_actual_observed",
        "actual_positive": "target_actual_positive",
        "actual_negative_only": "target_actual_negative_only",
        "zero_semantics": "target_zero_semantics",
    })

    frozen_start = pd.Timestamp(frozen_test_start)
    frames = []
    for h in (1, 2, 3):
        x = origin.copy().rename(columns={"month": "forecast_origin"})
        x["horizon"] = h
        x["target_month"] = pd.to_datetime(x["forecast_origin"]) + pd.offsets.DateOffset(months=h)
        x = x.loc[x["target_month"].lt(frozen_start)].copy()
        x = x.merge(target, on=["base_sku", "branch_code", "target_month"], how="left", validate="many_to_one")
        frames.append(x)

    out = pd.concat(frames, ignore_index=True)
    cal = deps["build_target_calendar_table"](out["target_month"].unique())
    out = out.merge(cal, on="target_month", how="left", validate="many_to_one")

    states = precompute_intermittent_states(pair_panel, alpha=alpha, beta=beta)
    out = out.merge(states, on=["base_sku", "branch_code", "forecast_origin"], how="left", validate="many_to_one")
    out[["pred_croston_sba", "pred_tsb"]] = out[["pred_croston_sba", "pred_tsb"]].fillna(0.0)

    out["forecast_origin"] = pd.to_datetime(out["forecast_origin"])
    out["target_month"] = pd.to_datetime(out["target_month"])
    out["feature_information_max_month"] = pd.to_datetime(out["feature_information_max_month"])
    out["target_available"] = out["target_available"].fillna(False).astype(bool)
    out["rolling_backtest_feature_version"] = "rolling_backtest_feature_v01"

    keys = ["forecast_origin", "target_month", "horizon", "base_sku", "branch_code"]
    if out.duplicated(keys).any():
        raise ValueError("Duplicate rolling supervised grain")
    if not out["feature_information_max_month"].le(out["forecast_origin"]).all():
        raise ValueError("Origin-safety failure in rolling feature reconstruction")
    if not out["target_month"].lt(frozen_start).all():
        raise ValueError("Frozen-test target leaked into rolling supervised panel")
    return out.sort_values(keys).reset_index(drop=True)


def feature_parity_audit(
    rolling_panel: pd.DataFrame,
    canonical_pair_feature_path: str,
    selected_features: Sequence[str],
    parity_origin: str,
) -> dict:
    keys = ["base_sku", "branch_code", "forecast_origin", "target_month", "horizon"]
    origin = pd.Timestamp(parity_origin)
    selected = [c for c in selected_features if c != "horizon"]
    needed = list(dict.fromkeys(keys + selected))
    canonical = pd.read_parquet(canonical_pair_feature_path, columns=needed)
    canonical["forecast_origin"] = pd.to_datetime(canonical["forecast_origin"])
    canonical["target_month"] = pd.to_datetime(canonical["target_month"])
    canonical = canonical.loc[canonical["forecast_origin"].eq(origin)].copy()
    rolling = rolling_panel.loc[rolling_panel["forecast_origin"].eq(origin), needed].copy()

    merged = rolling.merge(canonical, on=keys, how="outer", suffixes=("_roll", "_canon"), indicator=True)
    mismatch_cols: List[str] = []
    if not merged["_merge"].eq("both").all():
        mismatch_cols.append("__grain__")
    both = merged.loc[merged["_merge"].eq("both")]
    for c in selected:
        a = both[f"{c}_roll"]
        b = both[f"{c}_canon"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), rtol=1e-9, atol=1e-9, equal_nan=True)
            if not bool(np.all(ok)):
                mismatch_cols.append(c)
        else:
            aa = a.astype("string").fillna("__NA__")
            bb = b.astype("string").fillna("__NA__")
            if not bool(aa.eq(bb).all()):
                mismatch_cols.append(c)
    return {
        "parity_origin": str(origin.date()),
        "rolling_rows": int(len(rolling)),
        "canonical_rows": int(len(canonical)),
        "pass": len(mismatch_cols) == 0,
        "mismatch_columns": mismatch_cols,
    }


def rolling_origins(contract: dict) -> List[pd.Timestamp]:
    cfg = contract["origin_window"]
    start = pd.Timestamp(cfg["start"])
    end = pd.Timestamp(cfg["end"])
    frozen = pd.Timestamp(cfg["frozen_test_start"])
    origins = list(pd.date_range(start=start, end=end, freq="MS"))
    horizons = [int(h) for h in contract.get("horizons", [1, 2, 3])]
    origins = [o for o in origins if max(o + pd.offsets.DateOffset(months=h) for h in horizons) < frozen]
    return origins


def split_origin_honest(panel: pd.DataFrame, origin: pd.Timestamp, contract: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cal_months = int(contract["calibration"].get("target_months", 3))
    cal_start = origin - pd.offsets.DateOffset(months=cal_months - 1)

    available_past = panel.loc[panel["target_available"] & panel["target_month"].le(origin)].copy()
    fit = available_past.loc[available_past["target_month"].lt(cal_start)].copy()
    cal = available_past.loc[available_past["target_month"].between(cal_start, origin)].copy()
    if contract["calibration"].get("current_active_only", True):
        cal = cal.loc[_primary_mask(cal)].copy()

    evaluate = panel.loc[
        panel["forecast_origin"].eq(origin)
        & panel["target_month"].gt(origin)
        & _primary_mask(panel)
    ].copy()

    audit = {
        "origin": str(origin.date()),
        "calibration_start_target_month": str(pd.Timestamp(cal_start).date()),
        "fit_rows": int(len(fit)),
        "calibration_rows": int(len(cal)),
        "evaluation_rows": int(len(evaluate)),
    }
    return fit, cal, evaluate, audit


def _aligned_frames(deps: dict, fit: pd.DataFrame, cal: pd.DataFrame, ev: pd.DataFrame, features: Sequence[str]):
    Xfit, feats, cats = deps["_prepare_frame"](fit, features, direct_horizon=True)
    Xcal, _, _ = deps["_prepare_frame"](cal, feats, direct_horizon=True)
    Xev, _, _ = deps["_prepare_frame"](ev, feats, direct_horizon=True)
    for c in cats:
        categories = Xfit[c].cat.categories
        Xcal[c] = pd.Categorical(Xcal[c], categories=categories)
        Xev[c] = pd.Categorical(Xev[c], categories=categories)
    return Xfit, Xcal, Xev, feats, cats



def fit_lightgbm_fixed(X_train, y_train, contract: dict, deps: dict, binary: bool, n_estimators: int, preferred_device: str | None = None):
    from lightgbm import LGBMClassifier, LGBMRegressor

    requested = list(contract["gpu"].get("lightgbm_device_order", ["cuda", "gpu", "cpu"]))
    if not contract["gpu"].get("prefer_gpu", True) or not deps["nvidia_gpu_available"]():
        requested = ["cpu"]
    if preferred_device and preferred_device in requested:
        requested = [preferred_device] + [d for d in requested if d != preferred_device]
    errors = []
    for device in requested:
        params = deps["_lgbm_params"](contract, device, binary=binary)
        params.pop("_early_stopping_rounds", None)
        params["n_estimators"] = max(int(n_estimators), 1)
        try:
            cls = LGBMClassifier if binary else LGBMRegressor
            model = cls(**params)
            model.fit(X_train, y_train)
            return model, {
                "library": "lightgbm",
                "requested": requested[0],
                "actual": device,
                "fallback_reason": None if not errors else " | ".join(errors),
            }
        except Exception as e:
            errors.append(f"{device}:{type(e).__name__}:{str(e)[:180]}")
    raise RuntimeError("LightGBM fixed refit failed on all devices: " + " | ".join(errors))


def best_iteration_count(model, contract: dict) -> int:
    best = getattr(model, "best_iteration_", None)
    if best is not None and int(best) > 0:
        return int(best)
    fitted = getattr(model, "n_estimators_", None)
    if fitted is not None and int(fitted) > 0:
        return int(fitted)
    return int(contract.get("lightgbm_tweedie", {}).get("n_estimators", 1000))

def _add_lightweight_baselines(df: pd.DataFrame, deps: dict) -> pd.DataFrame:
    out = df.copy()
    base = deps["baseline_predictions"](out)
    for name, values in base.items():
        out[f"pred_{name}"] = values
    if "pred_croston_sba" not in out.columns or "pred_tsb" not in out.columns:
        raise ValueError("Precomputed intermittent baselines missing")
    return out


def select_gate_from_calibration(
    calibration_predictions: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    expert_cols: Dict[str, str],
    contract: dict,
    origin: pd.Timestamp,
    deps: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    cal = calibration_predictions.copy()
    ev = evaluation_predictions.copy()
    cal["behavior_segment"] = deps["behavior_segment"](cal, contract)
    ev["behavior_segment"] = deps["behavior_segment"](ev, contract)
    min_rows = int(contract["behavior"].get("min_segment_calibration_rows", 50))
    allowed_key = {
        "regular": "regular_experts",
        "intermittent": "intermittent_experts",
        "very_sparse": "very_sparse_experts",
    }
    gate_rows = []
    final = pd.Series(index=ev.index, dtype=float)
    for h in sorted(ev["horizon"].astype(int).unique()):
        cal_h = cal.loc[cal["horizon"].astype(int).eq(h)]
        for seg in ["regular", "intermittent", "very_sparse"]:
            cal_seg = cal_h.loc[cal_h["behavior_segment"].eq(seg)]
            ev_mask = ev["horizon"].astype(int).eq(h) & ev["behavior_segment"].eq(seg)
            ev_seg = ev.loc[ev_mask]
            allowed = contract["behavior"].get(allowed_key[seg], list(expert_cols))
            candidates = [e for e in allowed if e in expert_cols and expert_cols[e] in cal.columns and expert_cols[e] in ev.columns]
            if not candidates:
                raise ValueError(f"No rolling gate candidates for H{h} {seg}")
            if len(cal_seg) >= min_rows:
                selection_scope = "CALIBRATION_SEGMENT"
                selection = cal_seg
            else:
                selection_scope = "CALIBRATION_HORIZON_FALLBACK"
                selection = cal_h
            scores = [(e, deps["wape"](selection[PAIR_TARGET], selection[expert_cols[e]])) for e in candidates]
            finite = [(e, s) for e, s in scores if np.isfinite(s)]
            if not finite:
                raise ValueError(f"No finite rolling gate score for H{h} {seg}")
            chosen, sel_wape = min(finite, key=lambda x: x[1])
            if len(ev_seg):
                final.loc[ev_mask] = ev.loc[ev_mask, expert_cols[chosen]].to_numpy()
                eval_wape = deps["wape"](ev_seg[PAIR_TARGET], ev_seg[expert_cols[chosen]])
            else:
                eval_wape = float("nan")
            gate_rows.append({
                "forecast_origin": origin,
                "horizon": h,
                "behavior_segment": seg,
                "calibration_n_rows": int(len(cal_seg)),
                "evaluation_n_rows": int(len(ev_seg)),
                "selection_scope": selection_scope,
                "chosen_expert": chosen,
                "calibration_selection_wape": float(sel_wape),
                "evaluation_segment_wape": float(eval_wape) if np.isfinite(eval_wape) else np.nan,
            })
    return pd.DataFrame(gate_rows), final.fillna(0.0)


def _score_predictions(pred: pd.DataFrame, expert_cols: Dict[str, str], deps: dict, universe: str = "ROLLING_PRIMARY") -> pd.DataFrame:
    rows = []
    for name, col in expert_cols.items():
        if col not in pred.columns:
            continue
        rows.append(deps["metric_row"](name, pred[PAIR_TARGET], pred[col], universe=universe))
        for h in (1, 2, 3):
            m = pred["horizon"].astype(int).eq(h)
            rows.append(deps["metric_row"](name, pred.loc[m, PAIR_TARGET], pred.loc[m, col], horizon=h, universe=universe))
        if "behavior_segment" in pred.columns:
            for seg in ["regular", "intermittent", "very_sparse"]:
                m = pred["behavior_segment"].eq(seg)
                rows.append(deps["metric_row"](name, pred.loc[m, PAIR_TARGET], pred.loc[m, col], segment=seg, universe=universe))
    return pd.DataFrame(rows)


def _by_origin_score(pred: pd.DataFrame, expert_cols: Dict[str, str], deps: dict) -> pd.DataFrame:
    rows = []
    for origin, g in pred.groupby("forecast_origin", sort=True):
        for name, col in expert_cols.items():
            rows.append({"forecast_origin": origin, **deps["metric_row"](name, g[PAIR_TARGET], g[col], universe="ROLLING_PRIMARY")})
            for h in (1, 2, 3):
                gh = g.loc[g["horizon"].astype(int).eq(h)]
                rows.append({"forecast_origin": origin, **deps["metric_row"](name, gh[PAIR_TARGET], gh[col], horizon=h, universe="ROLLING_PRIMARY")})
    return pd.DataFrame(rows)


def cumulative_by_origin(pair3: pd.DataFrame, expert_cols: Dict[str, str], deps: dict) -> pd.DataFrame:
    frames = []
    for origin, g in pair3.groupby("forecast_origin", sort=True):
        s = deps["cumulative_3m_scoreboard"](g, expert_cols)
        if not s.empty:
            s.insert(0, "forecast_origin", origin)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def model_stability_summary(
    by_origin: pd.DataFrame,
    cumulative_score: pd.DataFrame,
    expert_cols: Dict[str, str],
) -> pd.DataFrame:
    overall = by_origin.loc[by_origin["horizon"].isna() & by_origin["segment"].isna()].copy()
    if overall.empty:
        return pd.DataFrame()
    wins = overall.loc[overall.groupby("forecast_origin")["wape"].idxmin(), "model"].value_counts()
    rows = []
    for model, g in overall.groupby("model"):
        c_pair = cumulative_score.loc[(cumulative_score["model"].eq(model)) & (cumulative_score["level"].eq("PAIR"))]
        c_sku = cumulative_score.loc[(cumulative_score["model"].eq(model)) & (cumulative_score["level"].eq("BASE_SKU"))]
        rows.append({
            "model": model,
            "n_origins": int(g["forecast_origin"].nunique()),
            "mean_origin_wape": float(g["wape"].mean()),
            "median_origin_wape": float(g["wape"].median()),
            "std_origin_wape": float(g["wape"].std(ddof=0)),
            "mean_abs_bias_ratio": float(g["bias_ratio"].abs().mean()),
            "max_abs_bias_ratio": float(g["bias_ratio"].abs().max()),
            "underforecast_origin_rate": float((g["bias_ratio"] < 0).mean()),
            "origin_wins": int(wins.get(model, 0)),
            "pair_wape_3m": float(c_pair["wape_3m"].iloc[0]) if len(c_pair) else np.nan,
            "base_sku_wape_3m": float(c_sku["wape_3m"].iloc[0]) if len(c_sku) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["median_origin_wape", "mean_abs_bias_ratio"]).reset_index(drop=True)


def run_rolling_backtest(
    pair_panel_path: str,
    canonical_pair_feature_path: str,
    selected_feature_path: str,
    model_contract_path: str,
    rolling_contract_path: str,
    output_dir: str,
    run_id: str,
    work9_root: str,
) -> dict:
    work9 = Path(work9_root)
    deps = _load_dependencies(work9)
    model_contract = yaml.safe_load(Path(model_contract_path).read_text(encoding="utf-8"))
    rolling_contract = yaml.safe_load(Path(rolling_contract_path).read_text(encoding="utf-8"))

    # Rolling contract inherits proven model hyperparameters, then replaces routing choices where specified.
    contract = dict(model_contract)
    for key in ["behavior", "hurdle", "gpu"]:
        if key in rolling_contract:
            base = dict(contract.get(key, {}))
            base.update(rolling_contract[key])
            contract[key] = base

    selected = deps["validate_feature_safety"](
        deps["load_selected_features"](Path(selected_feature_path)), model_contract
    )
    ib = model_contract.get("intermittent_baselines", {})
    panel = build_rolling_supervised_panel(
        pair_panel_path=pair_panel_path,
        frozen_test_start=rolling_contract["origin_window"]["frozen_test_start"],
        work9_root=work9_root,
        alpha=float(ib.get("alpha", 0.1)),
        beta=float(ib.get("beta", 0.1)),
    )

    parity = feature_parity_audit(
        panel,
        canonical_pair_feature_path=canonical_pair_feature_path,
        selected_features=selected,
        parity_origin=rolling_contract["feature_parity"]["origin"],
    )
    if rolling_contract["feature_parity"].get("required", True) and not parity["pass"]:
        raise ValueError(f"Rolling feature parity failed: {parity}")

    origins = rolling_origins(rolling_contract)
    min_origins = int(rolling_contract["origin_window"].get("min_origins", 6))
    if len(origins) < min_origins:
        raise ValueError(f"Insufficient rolling origins: {len(origins)} < {min_origins}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_eval = []
    all_cal = []
    all_gate = []
    all_threshold = []
    devices = []
    origin_audits = []

    for origin in origins:
        fit, cal, ev, audit = split_origin_honest(panel, origin, rolling_contract)
        print(f"[ROLLING] origin={origin.date()} fit={len(fit):,} cal={len(cal):,} eval={len(ev):,}")
        if ev.empty:
            raise ValueError(f"No evaluation rows for rolling origin {origin.date()}")

        cal_pred = _add_lightweight_baselines(cal, deps)
        ev_pred = _add_lightweight_baselines(ev, deps)
        expert_cols: Dict[str, str] = {
            "naive_1": "pred_naive_1",
            "seasonal_naive_12": "pred_seasonal_naive_12",
            "moving_average_3": "pred_moving_average_3",
            "croston_sba": "pred_croston_sba",
            "tsb": "pred_tsb",
        }

        audit["horizon_rows"] = {}
        for h in (1, 2, 3):
            fh = fit.loc[fit["horizon"].astype(int).eq(h)].copy()
            ch = cal.loc[cal["horizon"].astype(int).eq(h)].copy()
            eh = ev.loc[ev["horizon"].astype(int).eq(h)].copy()
            min_fit = int(rolling_contract["calibration"].get("min_fit_rows_per_horizon", 1000))
            min_cal = int(rolling_contract["calibration"].get("min_calibration_rows_per_horizon", 100))
            if len(fh) < min_fit or len(ch) < min_cal or eh.empty:
                raise ValueError(
                    f"Insufficient origin split H{h} at {origin.date()}: fit={len(fh)}, cal={len(ch)}, eval={len(eh)}"
                )
            audit["horizon_rows"][f"H{h}"] = {"fit": len(fh), "calibration": len(ch), "evaluation": len(eh)}

            Xfit, Xcal, Xev, feats, cats = _aligned_frames(deps, fh, ch, eh, selected)
            yfit = fh[PAIR_TARGET].astype(float).to_numpy()
            ycal = ch[PAIR_TARGET].astype(float).to_numpy()
            yev = eh[PAIR_TARGET].astype(float).to_numpy()

            # Selection fit: older FIT labels train the model; trailing CALIBRATION labels select
            # the stopping iteration. Evaluation labels are not visible here.
            lgbm_select, dev = deps["fit_lightgbm_direct"](Xfit, yfit, Xcal, ycal, contract, binary=False)
            devices.append(asdict(dev) | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "lightgbm_tweedie", "phase": "selection_fit"})
            cal_pred.loc[ch.index, "pred_lightgbm_tweedie"] = deps["predict_nonnegative"](lgbm_select, Xcal)
            lgbm_iters = best_iteration_count(lgbm_select, contract)

            # Operational refit: after iteration selection, refit on every label known at origin
            # (FIT + CALIBRATION), then forecast H1/H2/H3.
            full_h = pd.concat([fh, ch], axis=0).sort_index()
            Xfull, _, Xev_full, _, _ = _aligned_frames(deps, full_h, ch, eh, selected)
            yfull = full_h[PAIR_TARGET].astype(float).to_numpy()
            lgbm_final, final_dev = fit_lightgbm_fixed(
                Xfull, yfull, contract, deps, binary=False, n_estimators=lgbm_iters, preferred_device=dev.actual
            )
            devices.append(final_dev | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "lightgbm_tweedie", "phase": "final_refit", "n_estimators": lgbm_iters})
            ev_pred.loc[eh.index, "pred_lightgbm_tweedie"] = deps["predict_nonnegative"](lgbm_final, Xev_full)

            clf_select, cdev = deps["fit_lightgbm_direct"](
                Xfit, (yfit > 0).astype(int), Xcal, (ycal > 0).astype(int), contract, binary=True
            )
            devices.append(asdict(cdev) | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "hurdle_occurrence", "phase": "selection_fit"})
            p_cal = np.asarray(clf_select.predict_proba(Xcal)[:, 1], dtype=float)
            clf_iters = best_iteration_count(clf_select, contract)
            clf_final, clf_final_dev = fit_lightgbm_fixed(
                Xfull, (yfull > 0).astype(int), contract, deps, binary=True, n_estimators=clf_iters, preferred_device=cdev.actual
            )
            devices.append(clf_final_dev | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "hurdle_occurrence", "phase": "final_refit", "n_estimators": clf_iters})
            p_ev = np.asarray(clf_final.predict_proba(Xev_full)[:, 1], dtype=float)

            pos_fit = yfit > 0
            pos_cal = ycal > 0
            pos_full = yfull > 0
            min_pos = int(contract["hurdle"].get("min_positive_train_rows", 500))
            if int(pos_fit.sum()) >= min_pos and int(pos_cal.sum()) > 0:
                preg_select, pdev = deps["fit_lightgbm_direct"](
                    Xfit.loc[pos_fit], yfit[pos_fit], Xcal.loc[pos_cal], ycal[pos_cal], contract, binary=False
                )
                devices.append(asdict(pdev) | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "hurdle_positive", "phase": "selection_fit"})
                pos_cal_pred = deps["predict_nonnegative"](preg_select, Xcal)
                preg_iters = best_iteration_count(preg_select, contract)
                preg_final, preg_final_dev = fit_lightgbm_fixed(
                    Xfull.loc[pos_full], yfull[pos_full], contract, deps, binary=False,
                    n_estimators=preg_iters, preferred_device=pdev.actual
                )
                devices.append(preg_final_dev | {"forecast_origin": str(origin.date()), "horizon": h, "candidate": "hurdle_positive", "phase": "final_refit", "n_estimators": preg_iters})
                pos_ev_pred = deps["predict_nonnegative"](preg_final, Xev_full)
                threshold, cal_hurdle, report = deps["calibrate_hurdle_threshold"](
                    ycal, p_cal, pos_cal_pred, contract
                )
                ev_hurdle = np.where(p_ev < threshold, 0.0, p_ev * pos_ev_pred)
                report["forecast_origin"] = origin
                report["horizon"] = h
                all_threshold.append(report)
            else:
                threshold = None
                cal_hurdle = cal_pred.loc[ch.index, "pred_lightgbm_tweedie"].to_numpy()
                ev_hurdle = ev_pred.loc[eh.index, "pred_lightgbm_tweedie"].to_numpy()
            cal_pred.loc[ch.index, "pred_hurdle"] = cal_hurdle
            ev_pred.loc[eh.index, "pred_hurdle"] = ev_hurdle
            audit["horizon_rows"][f"H{h}"]["hurdle_threshold"] = threshold

        expert_cols.update({"lightgbm_tweedie": "pred_lightgbm_tweedie", "hurdle": "pred_hurdle"})
        gate_df, gated_eval = select_gate_from_calibration(cal_pred, ev_pred, expert_cols, contract, origin, deps)
        ev_pred["pred_behavior_gated"] = gated_eval
        ev_pred["behavior_segment"] = deps["behavior_segment"](ev_pred, contract)
        cal_pred["behavior_segment"] = deps["behavior_segment"](cal_pred, contract)
        expert_cols["behavior_gated"] = "pred_behavior_gated"

        ev_pred["backtest_origin"] = origin
        cal_pred["backtest_origin"] = origin
        all_eval.append(ev_pred)
        all_cal.append(cal_pred)
        all_gate.append(gate_df)
        origin_audits.append(audit)

    predictions = pd.concat(all_eval, ignore_index=True)
    calibration_predictions = pd.concat(all_cal, ignore_index=True)
    gate = pd.concat(all_gate, ignore_index=True)
    thresholds = pd.concat(all_threshold, ignore_index=True) if all_threshold else pd.DataFrame()
    expert_cols = {
        "naive_1": "pred_naive_1",
        "seasonal_naive_12": "pred_seasonal_naive_12",
        "moving_average_3": "pred_moving_average_3",
        "croston_sba": "pred_croston_sba",
        "tsb": "pred_tsb",
        "lightgbm_tweedie": "pred_lightgbm_tweedie",
        "hurdle": "pred_hurdle",
        "behavior_gated": "pred_behavior_gated",
    }

    score = _score_predictions(predictions, expert_cols, deps)
    by_origin = _by_origin_score(predictions, expert_cols, deps)
    pair3, coverage = deps["build_complete_3m_pair_windows"](predictions, expert_cols)
    cumulative = deps["cumulative_3m_scoreboard"](pair3, expert_cols)
    cumulative_origin = cumulative_by_origin(pair3, expert_cols, deps)
    revision, revision_detail = deps["forecast_revision_scoreboard"](predictions, expert_cols)
    stability = model_stability_summary(by_origin, cumulative, expert_cols)

    predictions.to_parquet(out_dir / "rolling_origin_predictions.parquet", index=False)
    calibration_predictions.to_parquet(out_dir / "rolling_calibration_predictions.parquet", index=False)
    score.to_csv(out_dir / "rolling_origin_scoreboard.csv", index=False)
    by_origin.to_csv(out_dir / "rolling_origin_by_origin.csv", index=False)
    gate.to_csv(out_dir / "rolling_behavior_gate.csv", index=False)
    thresholds.to_csv(out_dir / "rolling_hurdle_thresholds.csv", index=False)
    pair3.to_parquet(out_dir / "rolling_cumulative_3m_pair_predictions.parquet", index=False)
    cumulative.to_csv(out_dir / "rolling_cumulative_3m_scoreboard.csv", index=False)
    cumulative_origin.to_csv(out_dir / "rolling_cumulative_3m_by_origin.csv", index=False)
    revision.to_csv(out_dir / "forecast_revision_scoreboard.csv", index=False)
    revision_detail.to_parquet(out_dir / "forecast_revision_detail.parquet", index=False)
    stability.to_csv(out_dir / "model_stability_summary.csv", index=False)
    pd.json_normalize(origin_audits).to_csv(out_dir / "origin_audit.csv", index=False)

    frozen = pd.Timestamp(rolling_contract["origin_window"]["frozen_test_start"])
    safety = {
        "supabase_accessed": False,
        "frozen_test_touched": bool(predictions["target_month"].ge(frozen).any()),
        "reconciliation_run": False,
        "model_freeze_run": False,
        "production_published": False,
        "current_status_used_as_predictor": False,
        "evaluation_labels_used_for_fit_or_calibration": False,
    }
    if safety["frozen_test_touched"]:
        raise ValueError("Frozen test touched by rolling backtest")

    manifest = {
        "run_id": run_id,
        "run_type": "ROLLING_ORIGIN_BACKTEST_V01",
        "status": "PASS",
        "backtest_version": ROLLING_BACKTEST_VERSION,
        "source_model_candidate_version": rolling_contract["lineage"]["required_model_candidate_version"],
        "origins": [str(o.date()) for o in origins],
        "n_origins": len(origins),
        "feature_parity": parity,
        "evaluation_rows": int(len(predictions)),
        "calibration_rows": int(len(calibration_predictions)),
        "cumulative_3m_coverage": coverage,
        "revision_evaluable": bool(not revision.empty),
        "revision_rows": int(len(revision_detail)),
        "devices": devices,
        "origin_audits": origin_audits,
        "safety": safety,
        "input_sha256": {
            "pair_panel": sha256_file(Path(pair_panel_path)),
            "canonical_pair_feature": sha256_file(Path(canonical_pair_feature_path)),
            "selected_feature_list": sha256_file(Path(selected_feature_path)),
            "model_contract": sha256_file(Path(model_contract_path)),
            "rolling_contract": sha256_file(Path(rolling_contract_path)),
        },
    }
    manifest_path = out_dir / "backtest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
