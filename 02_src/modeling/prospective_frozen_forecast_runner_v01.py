from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

FROZEN_FORECAST_VERSION = "prospective_frozen_forecast_v01"
CHAMPION_MODEL = "soft_two_part_expected"
PAIR_TARGET = "target_actual_gross_m2"
PAIR_KEYS = ["base_sku", "branch_code"]
FORECAST_KEYS = ["base_sku", "branch_code", "forecast_origin", "target_month", "horizon"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def soft_expected_demand(p_positive, positive_quantity) -> np.ndarray:
    p = np.asarray(p_positive, dtype=float)
    q = np.asarray(positive_quantity, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p_positive and positive_quantity must have identical shape")
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    q = np.maximum(np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return p * q


def _load_modules(work9_root: Path):
    modeling = work9_root / "02_src" / "modeling"
    features = work9_root / "02_src" / "features"
    for p in [modeling, features]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import model_runner_v02 as model
    import rolling_backtest_runner_v01 as rolling
    import feature_builder_v013 as feature_builder
    return model, rolling, feature_builder


def _normalize_month(s) -> pd.Timestamp:
    return pd.Timestamp(s).to_period("M").to_timestamp()


def validate_pair_panel_cutoff(pair_panel: pd.DataFrame, forecast_origin: str | pd.Timestamp) -> pd.Timestamp:
    if "month" not in pair_panel.columns:
        raise ValueError("pair_panel missing month")
    origin = _normalize_month(forecast_origin)
    m = pd.to_datetime(pair_panel["month"]).dt.to_period("M").dt.to_timestamp()
    if m.empty:
        raise ValueError("pair_panel is empty")
    panel_max = pd.Timestamp(m.max())
    if panel_max != origin:
        raise ValueError(
            f"Prospective freeze requires pair panel max month == forecast origin; max={panel_max.date()} origin={origin.date()}"
        )
    return panel_max


def add_target_calendar(df: pd.DataFrame, feature_builder) -> pd.DataFrame:
    out = df.copy()
    out["target_month"] = pd.to_datetime(out["target_month"]).dt.to_period("M").dt.to_timestamp()
    cal = feature_builder.build_target_calendar_table(out["target_month"].unique())
    out = out.merge(cal, on="target_month", how="left", validate="many_to_one")
    out["horizon_label"] = "h" + out["horizon"].astype(int).astype(str)
    return out


def build_prospective_frames(
    pair_panel: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
    feature_builder,
    horizons: Sequence[int] = (1, 2, 3),
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    origin = _normalize_month(forecast_origin)
    p = pair_panel.copy()
    p["month"] = pd.to_datetime(p["month"]).dt.to_period("M").dt.to_timestamp()
    validate_pair_panel_cutoff(p, origin)

    origin_features = feature_builder.build_pair_origin_features(p)
    origin_features["month"] = pd.to_datetime(origin_features["month"]).dt.to_period("M").dt.to_timestamp()
    origin_features["feature_information_max_month"] = pd.to_datetime(origin_features["feature_information_max_month"])
    origin_features = origin_features.loc[origin_features["month"].le(origin)].copy()

    target_cols = [
        "base_sku", "branch_code", "month", "actual_gross_m2", "actual_observed",
        "actual_positive", "actual_negative_only", "target_available", "zero_semantics",
    ]
    target = p[[c for c in target_cols if c in p.columns]].copy().rename(columns={
        "month": "target_month",
        "actual_gross_m2": PAIR_TARGET,
        "actual_observed": "target_actual_observed",
        "actual_positive": "target_actual_positive",
        "actual_negative_only": "target_actual_negative_only",
        "zero_semantics": "target_zero_semantics",
    })

    train_frames: List[pd.DataFrame] = []
    forecast_frames: List[pd.DataFrame] = []
    origin_now = origin_features.loc[origin_features["month"].eq(origin)].copy()
    prod_mask = (
        origin_now["known_pair_asof_origin"].fillna(False).astype(bool)
        & origin_now["current_production_forecast_mask"].fillna(False).astype(bool)
    )
    origin_prod = origin_now.loc[prod_mask].copy()
    if origin_prod.empty:
        raise ValueError("Prospective production universe is empty at forecast origin")

    for h in [int(x) for x in horizons]:
        x = origin_features.copy().rename(columns={"month": "forecast_origin"})
        x["horizon"] = h
        x["target_month"] = x["forecast_origin"] + pd.offsets.DateOffset(months=h)
        x = x.merge(target, on=PAIR_KEYS + ["target_month"], how="left", validate="many_to_one")
        x = x.loc[x["target_month"].le(origin)].copy()
        x["target_available"] = x["target_available"].fillna(False).astype(bool)
        train_frames.append(x)

        f = origin_prod.copy().rename(columns={"month": "forecast_origin"})
        f["horizon"] = h
        f["target_month"] = f["forecast_origin"] + pd.offsets.DateOffset(months=h)
        forecast_frames.append(f)

    supervised = pd.concat(train_frames, ignore_index=True)
    supervised = add_target_calendar(supervised, feature_builder)
    future = pd.concat(forecast_frames, ignore_index=True)
    future = add_target_calendar(future, feature_builder)

    supervised["forecast_origin"] = pd.to_datetime(supervised["forecast_origin"])
    supervised["target_month"] = pd.to_datetime(supervised["target_month"])
    future["forecast_origin"] = pd.to_datetime(future["forecast_origin"])
    future["target_month"] = pd.to_datetime(future["target_month"])

    if supervised["target_month"].gt(origin).any():
        raise ValueError("Future label leaked into prospective supervised training panel")
    if not supervised["feature_information_max_month"].le(supervised["forecast_origin"]).all():
        raise ValueError("Origin safety failure in supervised training features")
    if not future["forecast_origin"].eq(origin).all():
        raise ValueError("Future forecast frame contains wrong forecast origin")
    if not future["target_month"].gt(origin).all():
        raise ValueError("Future forecast frame contains non-future target")
    if future.duplicated(FORECAST_KEYS).any():
        raise ValueError("Duplicate prospective forecast keys")

    expected_targets = [origin + pd.offsets.DateOffset(months=h) for h in [int(x) for x in horizons]]
    actual_targets = sorted(pd.unique(future["target_month"]))
    if list(actual_targets) != list(expected_targets):
        raise ValueError(f"Unexpected future target months: {actual_targets}")

    audit = {
        "forecast_origin": str(origin.date()),
        "pair_panel_max_month": str(pd.Timestamp(p["month"].max()).date()),
        "known_active_pairs_at_origin": int(len(origin_prod)),
        "forecast_rows": int(len(future)),
        "future_target_months": [str(pd.Timestamp(x).date()) for x in expected_targets],
        "training_supervised_rows_before_target_available_filter": int(len(supervised)),
    }
    return supervised.sort_values(FORECAST_KEYS).reset_index(drop=True), future.sort_values(FORECAST_KEYS).reset_index(drop=True), audit


def split_fit_calibration(
    supervised: pd.DataFrame,
    horizon: int,
    forecast_origin: str | pd.Timestamp,
    calibration_target_months: int = 3,
    calibration_current_active_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    origin = _normalize_month(forecast_origin)
    h = int(horizon)
    cal_start = origin - pd.offsets.DateOffset(months=int(calibration_target_months) - 1)
    d = supervised.loc[
        supervised["horizon"].astype(int).eq(h)
        & supervised["target_available"].fillna(False).astype(bool)
        & supervised["target_month"].le(origin)
    ].copy()
    fit = d.loc[d["target_month"].lt(cal_start)].copy()
    cal = d.loc[d["target_month"].between(cal_start, origin)].copy()
    if calibration_current_active_only:
        cal = cal.loc[
            cal["known_pair_asof_origin"].fillna(False).astype(bool)
            & cal["current_production_forecast_mask"].fillna(False).astype(bool)
        ].copy()
    if fit.empty or cal.empty:
        raise ValueError(f"Empty fit/calibration split for H{h}")
    return fit, cal, {
        "horizon": h,
        "calibration_start_target_month": str(pd.Timestamp(cal_start).date()),
        "fit_rows": int(len(fit)),
        "calibration_rows": int(len(cal)),
        "fit_target_max": str(pd.Timestamp(fit["target_month"].max()).date()),
        "calibration_target_min": str(pd.Timestamp(cal["target_month"].min()).date()),
        "calibration_target_max": str(pd.Timestamp(cal["target_month"].max()).date()),
    }


def _save_lightgbm_model(fitted_model, path: Path) -> None:
    booster = getattr(fitted_model, "booster_", None)
    if booster is None:
        raise ValueError("Expected fitted LightGBM sklearn model with booster_")
    booster.save_model(str(path))


def forecast_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in pred.groupby("horizon", sort=True):
        rows.append({
            "horizon": int(h),
            "target_month": str(pd.Timestamp(g["target_month"].iloc[0]).date()),
            "n_pairs": int(len(g)),
            "n_base_sku": int(g["base_sku"].nunique()),
            "n_branch": int(g["branch_code"].nunique()),
            "forecast_sum_m2": float(g["forecast_m2"].sum()),
            "forecast_mean_m2": float(g["forecast_m2"].mean()),
            "mean_p_positive": float(g["p_positive"].mean()),
            "mean_positive_quantity_m2": float(g["pred_positive_quantity"].mean()),
        })
    return pd.DataFrame(rows)


def run_prospective_frozen_forecast(
    pair_panel_path: str,
    selected_feature_path: str,
    model_contract_path: str,
    frozen_forecast_contract_path: str,
    architecture_freeze_contract_path: str,
    architecture_freeze_doc_path: str,
    dataset_pointer: dict,
    selection_pointer: dict,
    architecture_freeze_pointer: dict,
    run_dir: str,
    report_dir: str,
    run_id: str,
    work9_root: str,
) -> dict:
    work9 = Path(work9_root)
    model, rolling, feature_builder = _load_modules(work9)
    model_contract = yaml.safe_load(Path(model_contract_path).read_text(encoding="utf-8"))
    forecast_contract = yaml.safe_load(Path(frozen_forecast_contract_path).read_text(encoding="utf-8"))
    freeze_contract = yaml.safe_load(Path(architecture_freeze_contract_path).read_text(encoding="utf-8"))

    if dataset_pointer.get("status") != "PASS" or dataset_pointer.get("dataset_version") != "dataset_v012":
        raise ValueError("Current dataset pointer is not accepted dataset_v012")
    if selection_pointer.get("status") != "PASS" or selection_pointer.get("selection_version") != "feature_selection_v04":
        raise ValueError("Current feature selection pointer is not accepted feature_selection_v04")
    if architecture_freeze_pointer.get("status") != "APPROVED":
        raise ValueError("Model architecture freeze is not APPROVED")
    if architecture_freeze_pointer.get("champion_model") != CHAMPION_MODEL:
        raise ValueError("Architecture freeze champion is not soft_two_part_expected")
    if freeze_contract.get("champion", {}).get("name") != CHAMPION_MODEL:
        raise ValueError("Freeze contract champion mismatch")

    origin = _normalize_month(forecast_contract["forecast_origin"])
    horizons = [int(h) for h in forecast_contract.get("horizons", [1, 2, 3])]
    if horizons != [1, 2, 3]:
        raise ValueError("V1 prospective frozen forecast requires direct horizons [1,2,3]")

    pair_panel = pd.read_parquet(pair_panel_path)
    panel_max = validate_pair_panel_cutoff(pair_panel, origin)
    selected = model.validate_feature_safety(model.load_selected_features(Path(selected_feature_path)), model_contract)
    expected_count = int(forecast_contract["features"].get("selected_pair_feature_count", 54))
    if len(selected) != expected_count:
        raise ValueError(f"Selected feature count mismatch: {len(selected)} != {expected_count}")

    supervised, future, frame_audit = build_prospective_frames(pair_panel, origin, feature_builder, horizons=horizons)
    future["p_positive"] = np.nan
    future["pred_positive_quantity"] = np.nan
    future["forecast_m2"] = np.nan

    out = Path(run_dir)
    report = Path(report_dir)
    out.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    model_dir = out / "models"
    pred_dir = out / "predictions"
    contracts_dir = out / "contracts"
    model_dir.mkdir()
    pred_dir.mkdir()
    contracts_dir.mkdir()

    cal_months = int(forecast_contract["training"].get("calibration_target_months", 3))
    cal_active = bool(forecast_contract["training"].get("calibration_current_active_only", True))
    min_pos_fit = int(forecast_contract["training"].get("min_positive_fit_rows_per_horizon", 500))
    min_pos_cal = int(forecast_contract["training"].get("min_positive_calibration_rows_per_horizon", 50))
    training_audit: List[dict] = []
    devices: List[dict] = []

    for h in horizons:
        fit, cal, audit = split_fit_calibration(
            supervised, h, origin,
            calibration_target_months=cal_months,
            calibration_current_active_only=cal_active,
        )
        ev = future.loc[future["horizon"].astype(int).eq(h)].copy()
        if ev.empty:
            raise ValueError(f"Empty prospective forecast H{h}")

        Xfit, Xcal, Xev, feats, cats = rolling._aligned_frames(
            {"_prepare_frame": model._prepare_frame}, fit, cal, ev, selected
        )
        yfit = fit[PAIR_TARGET].astype(float).to_numpy()
        ycal = cal[PAIR_TARGET].astype(float).to_numpy()

        clf_select, cdev = model.fit_lightgbm_direct(
            Xfit, (yfit > 0).astype(int), Xcal, (ycal > 0).astype(int), model_contract, binary=True
        )
        clf_iters = rolling.best_iteration_count(clf_select, model_contract)

        full_h = pd.concat([fit, cal], axis=0).sort_index()
        Xfull, _, Xev_full, _, _ = rolling._aligned_frames(
            {"_prepare_frame": model._prepare_frame}, full_h, cal, ev, selected
        )
        yfull = full_h[PAIR_TARGET].astype(float).to_numpy()
        clf_final, clf_dev = rolling.fit_lightgbm_fixed(
            Xfull, (yfull > 0).astype(int), model_contract,
            {"nvidia_gpu_available": model.nvidia_gpu_available, "_lgbm_params": model._lgbm_params},
            binary=True, n_estimators=clf_iters, preferred_device=cdev.actual,
        )
        p_ev = np.asarray(clf_final.predict_proba(Xev_full)[:, 1], dtype=float)

        pos_fit = yfit > 0
        pos_cal = ycal > 0
        pos_full = yfull > 0
        if int(pos_fit.sum()) < min_pos_fit or int(pos_cal.sum()) < min_pos_cal:
            raise ValueError(
                f"Insufficient positive rows H{h}: fit={int(pos_fit.sum())} cal={int(pos_cal.sum())}"
            )
        preg_select, pdev = model.fit_lightgbm_direct(
            Xfit.loc[pos_fit], yfit[pos_fit], Xcal.loc[pos_cal], ycal[pos_cal], model_contract, binary=False
        )
        preg_iters = rolling.best_iteration_count(preg_select, model_contract)
        preg_final, preg_dev = rolling.fit_lightgbm_fixed(
            Xfull.loc[pos_full], yfull[pos_full], model_contract,
            {"nvidia_gpu_available": model.nvidia_gpu_available, "_lgbm_params": model._lgbm_params},
            binary=False, n_estimators=preg_iters, preferred_device=pdev.actual,
        )
        q_ev = model.predict_nonnegative(preg_final, Xev_full)
        expected = soft_expected_demand(p_ev, q_ev)

        idx = future["horizon"].astype(int).eq(h)
        future.loc[idx, "p_positive"] = p_ev
        future.loc[idx, "pred_positive_quantity"] = q_ev
        future.loc[idx, "forecast_m2"] = expected

        occ_model_path = model_dir / f"occurrence_h{h}_lightgbm.txt"
        qty_model_path = model_dir / f"positive_quantity_h{h}_lightgbm_tweedie.txt"
        _save_lightgbm_model(clf_final, occ_model_path)
        _save_lightgbm_model(preg_final, qty_model_path)

        audit.update({
            "forecast_rows": int(len(ev)),
            "positive_fit_rows": int(pos_fit.sum()),
            "positive_calibration_rows": int(pos_cal.sum()),
            "occurrence_n_estimators": int(clf_iters),
            "positive_quantity_n_estimators": int(preg_iters),
            "occurrence_device": clf_dev.get("actual"),
            "positive_quantity_device": preg_dev.get("actual"),
        })
        training_audit.append(audit)
        devices.extend([
            asdict(cdev) | {"horizon": h, "component": "occurrence", "phase": "iteration_selection"},
            clf_dev | {"horizon": h, "component": "occurrence", "phase": "final_refit", "n_estimators": clf_iters},
            asdict(pdev) | {"horizon": h, "component": "positive_quantity", "phase": "iteration_selection"},
            preg_dev | {"horizon": h, "component": "positive_quantity", "phase": "final_refit", "n_estimators": preg_iters},
        ])

    if future[["p_positive", "pred_positive_quantity", "forecast_m2"]].isna().any().any():
        raise ValueError("Missing prospective frozen predictions")
    if not future["p_positive"].between(0, 1, inclusive="both").all():
        raise ValueError("p_positive outside [0,1]")
    if (future["pred_positive_quantity"] < 0).any() or (future["forecast_m2"] < 0).any():
        raise ValueError("Negative prospective forecast")
    if future.duplicated(FORECAST_KEYS).any():
        raise ValueError("Duplicate final prospective forecast keys")

    pred_cols = FORECAST_KEYS + [
        "known_pair_asof_origin", "current_production_forecast_mask",
        "p_positive", "pred_positive_quantity", "forecast_m2",
    ]
    pred = future[pred_cols].copy().sort_values(FORECAST_KEYS).reset_index(drop=True)
    pred_path = pred_dir / "prospective_frozen_pair_forecast_v01.parquet"
    pred_csv_path = pred_dir / "prospective_frozen_pair_forecast_v01.csv"
    pred.to_parquet(pred_path, index=False)
    pred.to_csv(pred_csv_path, index=False)

    summary = forecast_summary(pred)
    summary_path = report / "prospective_frozen_forecast_summary.csv"
    audit_path = report / "prospective_frozen_training_audit.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(training_audit).to_csv(audit_path, index=False)

    snapshot_paths = [
        Path(selected_feature_path), Path(model_contract_path), Path(frozen_forecast_contract_path),
        Path(architecture_freeze_contract_path), Path(architecture_freeze_doc_path),
    ]
    for p in snapshot_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        shutil.copy2(p, contracts_dir / p.name)

    model_hashes = {p.name: sha256_file(p) for p in sorted(model_dir.glob("*.txt"))}
    safety = {
        "supabase_accessed": False,
        "future_actual_labels_read": False,
        "future_actual_labels_used_for_fit_or_calibration": False,
        "pair_panel_max_month_equals_forecast_origin": panel_max == origin,
        "production_universe_current_active_known_pair_only": bool(
            pred["known_pair_asof_origin"].all() and pred["current_production_forecast_mask"].all()
        ),
        "hard_zero_threshold_used": False,
        "posthoc_global_bias_scaling_used": False,
        "pair_level_rounding_used": False,
        "architecture_changed_after_freeze": False,
        "frozen_test_evaluation_run": False,
        "production_published": False,
    }
    if not all([
        safety["pair_panel_max_month_equals_forecast_origin"],
        safety["production_universe_current_active_known_pair_only"],
    ]):
        raise ValueError(f"Prospective frozen forecast safety failure: {safety}")

    reservation = {
        "status": "LOCKED_PENDING_ACTUALS",
        "forecast_origin": str(origin.date()),
        "target_months": [str(pd.Timestamp(origin + pd.offsets.DateOffset(months=h)).date()) for h in horizons],
        "evaluation_rule": "Evaluate this immutable vintage only after all Jul-Aug-Sep 2026 actuals are closed/loaded; do not use these labels to alter V1 architecture.",
        "champion_model": CHAMPION_MODEL,
        "prediction_path": str(pred_path),
        "prediction_sha256": sha256_file(pred_path),
    }
    reservation_path = report / "prospective_frozen_test_reservation.json"
    reservation_path.write_text(json.dumps(reservation, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "run_type": "PROSPECTIVE_FROZEN_FORECAST_V01",
        "status": "PASS",
        "frozen_forecast_version": FROZEN_FORECAST_VERSION,
        "architecture_freeze_run_id": architecture_freeze_pointer.get("freeze_id") or architecture_freeze_pointer.get("run_id"),
        "champion_model": CHAMPION_MODEL,
        "formula": "P(Y>0|X) * E(Y|Y>0,X)",
        "dataset_run_id": dataset_pointer.get("run_id"),
        "feature_selection_run_id": selection_pointer.get("run_id"),
        "forecast_origin": str(origin.date()),
        "target_months": reservation["target_months"],
        "selected_feature_count": int(len(selected)),
        "frame_audit": frame_audit,
        "training_audit": training_audit,
        "forecast_rows": int(len(pred)),
        "unique_pairs": int(pred[["base_sku", "branch_code"]].drop_duplicates().shape[0]),
        "forecast_summary": summary.to_dict(orient="records"),
        "devices": devices,
        "safety": safety,
        "artifacts": {
            "prediction_parquet": str(pred_path),
            "prediction_csv": str(pred_csv_path),
            "summary_csv": str(summary_path),
            "training_audit_csv": str(audit_path),
            "reservation_json": str(reservation_path),
            "models_dir": str(model_dir),
        },
        "input_sha256": {
            "pair_panel": sha256_file(Path(pair_panel_path)),
            "selected_feature_list": sha256_file(Path(selected_feature_path)),
            "model_contract": sha256_file(Path(model_contract_path)),
            "frozen_forecast_contract": sha256_file(Path(frozen_forecast_contract_path)),
            "architecture_freeze_contract": sha256_file(Path(architecture_freeze_contract_path)),
            "architecture_freeze_doc": sha256_file(Path(architecture_freeze_doc_path)),
        },
        "output_sha256": {
            "prediction_parquet": sha256_file(pred_path),
            "prediction_csv": sha256_file(pred_csv_path),
            "summary_csv": sha256_file(summary_path),
            "training_audit_csv": sha256_file(audit_path),
            "reservation_json": sha256_file(reservation_path),
            "models": model_hashes,
        },
    }
    manifest_path = out / "prospective_frozen_forecast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
