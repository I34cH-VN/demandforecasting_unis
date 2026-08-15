from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

CHALLENGER_VERSION = "soft_two_part_challenger_v01"
PAIR_TARGET = "target_actual_gross_m2"
EVAL_KEYS = ["base_sku", "branch_code", "forecast_origin", "target_month", "horizon"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def soft_expected_demand(p_positive, positive_quantity) -> np.ndarray:
    """E[Y|X] = P(Y>0|X) * E[Y|Y>0,X], with no hard zero threshold."""
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
    return model, rolling


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["forecast_origin", "target_month"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c])
    if "horizon" in d.columns:
        d["horizon"] = pd.to_numeric(d["horizon"], errors="raise").astype(int)
    return d


def align_reference_predictions(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    reference_col: str = "pred_lightgbm_tweedie",
    atol: float = 1e-9,
) -> pd.DataFrame:
    ch = _normalize_keys(challenger)
    ref = _normalize_keys(reference)
    need_ch = set(EVAL_KEYS + [PAIR_TARGET, "pred_soft_two_part_expected", "p_positive", "pred_positive_quantity"])
    need_ref = set(EVAL_KEYS + [PAIR_TARGET, reference_col])
    miss_ch = need_ch - set(ch.columns)
    miss_ref = need_ref - set(ref.columns)
    if miss_ch:
        raise ValueError(f"Challenger predictions missing columns: {sorted(miss_ch)}")
    if miss_ref:
        raise ValueError(f"Reference predictions missing columns: {sorted(miss_ref)}")
    if ch.duplicated(EVAL_KEYS).any():
        raise ValueError("Duplicate challenger evaluation keys")
    if ref.duplicated(EVAL_KEYS).any():
        raise ValueError("Duplicate reference evaluation keys")
    if len(ch) != len(ref):
        raise ValueError(f"Evaluation row count mismatch: challenger={len(ch)} reference={len(ref)}")
    ref_keep = ref[EVAL_KEYS + [PAIR_TARGET, reference_col]].rename(
        columns={PAIR_TARGET: f"{PAIR_TARGET}_reference", reference_col: "pred_lightgbm_reference"}
    )
    out = ch.merge(ref_keep, on=EVAL_KEYS, how="outer", indicator=True, validate="one_to_one")
    if not out["_merge"].eq("both").all():
        counts = out["_merge"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Reference/challenger evaluation key mismatch: {counts}")
    out = out.drop(columns="_merge")
    a = pd.to_numeric(out[PAIR_TARGET], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(out[f"{PAIR_TARGET}_reference"], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
        raise ValueError("Reference/challenger actual target mismatch")
    return out.drop(columns=f"{PAIR_TARGET}_reference")


def zero_positive_diagnostics(pred: pd.DataFrame, pred_cols: Dict[str, str], model) -> pd.DataFrame:
    rows: List[dict] = []
    for horizon in [None, 1, 2, 3]:
        d = pred if horizon is None else pred.loc[pred["horizon"].astype(int).eq(horizon)]
        y = pd.to_numeric(d[PAIR_TARGET], errors="coerce").to_numpy(dtype=float)
        zero = y == 0
        pos = y > 0
        for name, col in pred_cols.items():
            p = np.maximum(pd.to_numeric(d[col], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0)
            rows.append({
                "model": name,
                "horizon": horizon,
                "n_rows": int(len(d)),
                "actual_zero_rows": int(zero.sum()),
                "zero_forecast_sum_m2": float(p[zero].sum()),
                "zero_forecast_mean_m2": float(p[zero].mean()) if zero.any() else np.nan,
                "zero_false_positive_rate": model.zero_false_positive_rate(y, p),
                "actual_positive_rows": int(pos.sum()),
                "positive_actual_sum_m2": float(y[pos].sum()),
                "positive_forecast_sum_m2": float(p[pos].sum()),
                "positive_wape": model.wape(y[pos], p[pos]) if pos.any() else np.nan,
                "positive_bias_ratio": model.bias_ratio(y[pos], p[pos]) if pos.any() else np.nan,
                "overall_wape": model.wape(y, p),
                "overall_bias_ratio": model.bias_ratio(y, p),
            })
    return pd.DataFrame(rows)


def probability_diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in [None, 1, 2, 3]:
        d = pred if horizon is None else pred.loc[pred["horizon"].astype(int).eq(horizon)]
        y = (pd.to_numeric(d[PAIR_TARGET], errors="coerce").to_numpy(dtype=float) > 0).astype(float)
        p = np.clip(pd.to_numeric(d["p_positive"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
        pos = y == 1
        zero = y == 0
        q = np.maximum(pd.to_numeric(d["pred_positive_quantity"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0)
        actual = pd.to_numeric(d[PAIR_TARGET], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "horizon": horizon,
            "n_rows": int(len(d)),
            "actual_positive_rate": float(y.mean()) if len(y) else np.nan,
            "mean_p_positive": float(p.mean()) if len(p) else np.nan,
            "mean_p_on_actual_zero": float(p[zero].mean()) if zero.any() else np.nan,
            "mean_p_on_actual_positive": float(p[pos].mean()) if pos.any() else np.nan,
            "probability_brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
            "positive_quantity_wape_on_positive_actual": (
                float(np.sum(np.abs(actual[pos] - q[pos])) / np.sum(actual[pos])) if pos.any() and actual[pos].sum() > 0 else np.nan
            ),
            "positive_quantity_bias_ratio_on_positive_actual": (
                float((q[pos].sum() - actual[pos].sum()) / actual[pos].sum()) if pos.any() and actual[pos].sum() > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def score_predictions(pred: pd.DataFrame, pred_cols: Dict[str, str], model) -> pd.DataFrame:
    rows = []
    for name, col in pred_cols.items():
        rows.append(model.metric_row(name, pred[PAIR_TARGET], pred[col], universe="ROLLING_PRIMARY_SAME_ROWS"))
        for h in (1, 2, 3):
            m = pred["horizon"].astype(int).eq(h)
            rows.append(model.metric_row(name, pred.loc[m, PAIR_TARGET], pred.loc[m, col], horizon=h, universe="ROLLING_PRIMARY_SAME_ROWS"))
    return pd.DataFrame(rows)


def by_origin_score(pred: pd.DataFrame, pred_cols: Dict[str, str], model) -> pd.DataFrame:
    rows = []
    for origin, g in pred.groupby("forecast_origin", sort=True):
        for name, col in pred_cols.items():
            rows.append({"forecast_origin": origin, **model.metric_row(name, g[PAIR_TARGET], g[col], universe="ROLLING_PRIMARY_SAME_ROWS")})
            for h in (1, 2, 3):
                gh = g.loc[g["horizon"].astype(int).eq(h)]
                rows.append({"forecast_origin": origin, **model.metric_row(name, gh[PAIR_TARGET], gh[col], horizon=h, universe="ROLLING_PRIMARY_SAME_ROWS")})
    return pd.DataFrame(rows)


def comparison_review(
    monthly: pd.DataFrame,
    cumulative: pd.DataFrame,
    zero_pos: pd.DataFrame,
    revision: pd.DataFrame,
) -> dict:
    ref = "lightgbm_reference"
    chal = "soft_two_part_expected"
    out = {"status": "REVIEW_REQUIRED", "auto_promote": False, "auto_freeze": False, "metrics": {}}

    def get_monthly(model_name, field):
        x = monthly.loc[(monthly["model"].eq(model_name)) & monthly["horizon"].isna()]
        return float(x[field].iloc[0]) if len(x) else np.nan

    out["metrics"]["monthly_wape"] = {ref: get_monthly(ref, "wape"), chal: get_monthly(chal, "wape")}
    out["metrics"]["monthly_bias_ratio"] = {ref: get_monthly(ref, "bias_ratio"), chal: get_monthly(chal, "bias_ratio")}

    for level in ["PAIR", "BASE_SKU", "BRANCH", "PORTFOLIO"]:
        rec = {}
        for m in [ref, chal]:
            x = cumulative.loc[(cumulative["model"].eq(m)) & (cumulative["level"].eq(level))]
            rec[m] = {
                "wape_3m": float(x["wape_3m"].iloc[0]) if len(x) else np.nan,
                "bias_ratio_3m": float(x["bias_ratio_3m"].iloc[0]) if len(x) else np.nan,
            }
        out["metrics"][f"cumulative_3m_{level.lower()}"] = rec

    z = zero_pos.loc[zero_pos["horizon"].isna()].set_index("model")
    for field in ["zero_forecast_sum_m2", "positive_wape", "positive_bias_ratio"]:
        out["metrics"][field] = {
            ref: float(z.loc[ref, field]) if ref in z.index else np.nan,
            chal: float(z.loc[chal, field]) if chal in z.index else np.nan,
        }

    rev = {}
    for transition in ["H2_TO_H1", "H3_TO_H2"]:
        rev[transition] = {}
        for m in [ref, chal]:
            x = revision.loc[(revision["model"].eq(m)) & (revision["level"].eq("PAIR")) & (revision["transition"].eq(transition))]
            rev[transition][m] = float(x["revision_ratio_vs_old_forecast"].iloc[0]) if len(x) else np.nan
    out["metrics"]["pair_revision_ratio"] = rev
    return out


def run_soft_two_part_challenger(
    pair_panel_path: str,
    canonical_pair_feature_path: str,
    selected_feature_path: str,
    model_contract_path: str,
    rolling_contract_path: str,
    challenger_contract_path: str,
    reference_prediction_path: str,
    rolling_pointer: dict,
    diagnosis_pointer: dict,
    output_dir: str,
    run_id: str,
    work9_root: str,
) -> dict:
    work9 = Path(work9_root)
    model, rolling = _load_modules(work9)
    model_contract = yaml.safe_load(Path(model_contract_path).read_text(encoding="utf-8"))
    rolling_contract = yaml.safe_load(Path(rolling_contract_path).read_text(encoding="utf-8"))
    challenger_contract = yaml.safe_load(Path(challenger_contract_path).read_text(encoding="utf-8"))

    if rolling_pointer.get("status") != "PASS" or rolling_pointer.get("backtest_version") != "rolling_backtest_v01":
        raise ValueError("Current rolling pointer is not PASS rolling_backtest_v01")
    if diagnosis_pointer.get("status") != "PASS" or diagnosis_pointer.get("diagnosis_version") != "underforecast_diagnosis_v01":
        raise ValueError("05B diagnosis pointer is not PASS underforecast_diagnosis_v01")
    if diagnosis_pointer.get("source_rolling_run_id") != rolling_pointer.get("run_id"):
        raise ValueError("05B diagnosis does not point to current rolling run")

    selected = model.validate_feature_safety(model.load_selected_features(Path(selected_feature_path)), model_contract)
    panel = rolling.build_rolling_supervised_panel(
        pair_panel_path=pair_panel_path,
        frozen_test_start=rolling_contract["origin_window"]["frozen_test_start"],
        work9_root=work9_root,
        alpha=float(model_contract.get("intermittent_baselines", {}).get("alpha", 0.1)),
        beta=float(model_contract.get("intermittent_baselines", {}).get("beta", 0.1)),
    )
    parity = rolling.feature_parity_audit(
        panel,
        canonical_pair_feature_path=canonical_pair_feature_path,
        selected_features=selected,
        parity_origin=rolling_contract["feature_parity"]["origin"],
    )
    if not parity.get("pass", False):
        raise ValueError(f"05C feature parity failed: {parity}")

    reference = pd.read_parquet(reference_prediction_path)
    reference = _normalize_keys(reference)
    origins = rolling.rolling_origins(rolling_contract)
    if len(origins) < int(rolling_contract["origin_window"].get("min_origins", 12)):
        raise ValueError("05C rolling origin count below contract minimum")

    min_pos_fit = int(challenger_contract["training"].get("min_positive_fit_rows_per_horizon", 500))
    min_pos_cal = int(challenger_contract["training"].get("min_positive_calibration_rows_per_horizon", 50))
    all_eval = []
    devices = []
    origin_audits = []

    for origin in origins:
        fit, cal, ev, audit = rolling.split_origin_honest(panel, origin, rolling_contract)
        print(f"[05C] origin={origin.date()} fit={len(fit):,} cal={len(cal):,} eval={len(ev):,}")
        out_ev = ev[EVAL_KEYS + [PAIR_TARGET]].copy()
        out_ev["p_positive"] = np.nan
        out_ev["pred_positive_quantity"] = np.nan
        out_ev["pred_soft_two_part_expected"] = np.nan
        audit["horizon_rows"] = {}

        for h in (1, 2, 3):
            fh = fit.loc[fit["horizon"].astype(int).eq(h)].copy()
            ch = cal.loc[cal["horizon"].astype(int).eq(h)].copy()
            eh = ev.loc[ev["horizon"].astype(int).eq(h)].copy()
            if eh.empty:
                raise ValueError(f"05C empty evaluation H{h} at {origin.date()}")
            Xfit, Xcal, Xev, feats, cats = rolling._aligned_frames(
                {"_prepare_frame": model._prepare_frame}, fh, ch, eh, selected
            )
            yfit = fh[PAIR_TARGET].astype(float).to_numpy()
            ycal = ch[PAIR_TARGET].astype(float).to_numpy()

            # Occurrence component: select stopping iteration on trailing CAL only.
            clf_select, cdev = model.fit_lightgbm_direct(
                Xfit, (yfit > 0).astype(int), Xcal, (ycal > 0).astype(int), model_contract, binary=True
            )
            clf_iters = rolling.best_iteration_count(clf_select, model_contract)

            full_h = pd.concat([fh, ch], axis=0).sort_index()
            Xfull, _, Xev_full, _, _ = rolling._aligned_frames(
                {"_prepare_frame": model._prepare_frame}, full_h, ch, eh, selected
            )
            yfull = full_h[PAIR_TARGET].astype(float).to_numpy()
            clf_final, clf_dev = rolling.fit_lightgbm_fixed(
                Xfull, (yfull > 0).astype(int), model_contract,
                {"nvidia_gpu_available": model.nvidia_gpu_available, "_lgbm_params": model._lgbm_params},
                binary=True, n_estimators=clf_iters, preferred_device=cdev.actual
            )
            p_ev = np.asarray(clf_final.predict_proba(Xev_full)[:, 1], dtype=float)

            # Positive quantity component: only positive labels; no zero rows dilute magnitude learning.
            pos_fit = yfit > 0
            pos_cal = ycal > 0
            pos_full = yfull > 0
            if int(pos_fit.sum()) < min_pos_fit or int(pos_cal.sum()) < min_pos_cal:
                raise ValueError(
                    f"05C insufficient positive rows H{h} at {origin.date()}: fit={int(pos_fit.sum())}, cal={int(pos_cal.sum())}"
                )
            preg_select, pdev = model.fit_lightgbm_direct(
                Xfit.loc[pos_fit], yfit[pos_fit], Xcal.loc[pos_cal], ycal[pos_cal], model_contract, binary=False
            )
            preg_iters = rolling.best_iteration_count(preg_select, model_contract)
            preg_final, preg_dev = rolling.fit_lightgbm_fixed(
                Xfull.loc[pos_full], yfull[pos_full], model_contract,
                {"nvidia_gpu_available": model.nvidia_gpu_available, "_lgbm_params": model._lgbm_params},
                binary=False, n_estimators=preg_iters, preferred_device=pdev.actual
            )
            q_ev = model.predict_nonnegative(preg_final, Xev_full)
            expected = soft_expected_demand(p_ev, q_ev)

            idx = out_ev["horizon"].astype(int).eq(h)
            # out_ev preserves ev row order, so assign by ordered H rows.
            out_ev.loc[idx, "p_positive"] = p_ev
            out_ev.loc[idx, "pred_positive_quantity"] = q_ev
            out_ev.loc[idx, "pred_soft_two_part_expected"] = expected

            devices.extend([
                asdict(cdev) | {"forecast_origin": str(origin.date()), "horizon": h, "component": "occurrence", "phase": "selection_fit"},
                clf_dev | {"forecast_origin": str(origin.date()), "horizon": h, "component": "occurrence", "phase": "final_refit", "n_estimators": clf_iters},
                asdict(pdev) | {"forecast_origin": str(origin.date()), "horizon": h, "component": "positive_quantity", "phase": "selection_fit"},
                preg_dev | {"forecast_origin": str(origin.date()), "horizon": h, "component": "positive_quantity", "phase": "final_refit", "n_estimators": preg_iters},
            ])
            audit["horizon_rows"][f"H{h}"] = {
                "fit": int(len(fh)), "calibration": int(len(ch)), "evaluation": int(len(eh)),
                "positive_fit": int(pos_fit.sum()), "positive_calibration": int(pos_cal.sum()),
                "occurrence_n_estimators": int(clf_iters), "positive_quantity_n_estimators": int(preg_iters),
            }

        if out_ev[["p_positive", "pred_positive_quantity", "pred_soft_two_part_expected"]].isna().any().any():
            raise ValueError(f"05C missing challenger predictions at origin {origin.date()}")
        all_eval.append(out_ev)
        origin_audits.append(audit)

    challenger = pd.concat(all_eval, ignore_index=True)
    pred = align_reference_predictions(
        challenger, reference, reference_col=challenger_contract["reference"]["prediction_column"]
    )
    pred_cols = {
        "lightgbm_reference": "pred_lightgbm_reference",
        "soft_two_part_expected": "pred_soft_two_part_expected",
    }

    monthly = score_predictions(pred, pred_cols, model)
    by_origin = by_origin_score(pred, pred_cols, model)
    pair3, coverage = model.build_complete_3m_pair_windows(pred, pred_cols)
    cumulative = model.cumulative_3m_scoreboard(pair3, pred_cols)
    cumulative_origin = rolling.cumulative_by_origin(pair3, pred_cols, {"cumulative_3m_scoreboard": model.cumulative_3m_scoreboard})
    revision, revision_detail = model.forecast_revision_scoreboard(pred, pred_cols)
    stability = rolling.model_stability_summary(by_origin, cumulative, pred_cols)
    zero_pos = zero_positive_diagnostics(pred, pred_cols, model)
    prob_diag = probability_diagnostics(pred)
    review = comparison_review(monthly, cumulative, zero_pos, revision)

    frozen = pd.Timestamp(rolling_contract["origin_window"]["frozen_test_start"])
    safety = {
        "supabase_accessed": False,
        "frozen_test_touched": bool(pd.to_datetime(pred["target_month"]).ge(frozen).any()),
        "reconciliation_run": False,
        "model_freeze_run": False,
        "production_published": False,
        "current_status_used_as_predictor": False,
        "evaluation_labels_used_for_fit_or_iteration_selection": False,
        "hard_zero_threshold_used": False,
        "posthoc_global_bias_scaling_used": False,
    }
    if safety["frozen_test_touched"]:
        raise ValueError("05C touched Frozen Test period")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    pred.to_parquet(out / "soft_two_part_predictions.parquet", index=False)
    monthly.to_csv(out / "soft_two_part_scoreboard.csv", index=False)
    by_origin.to_csv(out / "soft_two_part_by_origin.csv", index=False)
    pair3.to_parquet(out / "soft_two_part_cumulative_3m_pair_predictions.parquet", index=False)
    cumulative.to_csv(out / "soft_two_part_cumulative_3m_scoreboard.csv", index=False)
    cumulative_origin.to_csv(out / "soft_two_part_cumulative_3m_by_origin.csv", index=False)
    revision.to_csv(out / "soft_two_part_revision_scoreboard.csv", index=False)
    revision_detail.to_parquet(out / "soft_two_part_revision_detail.parquet", index=False)
    stability.to_csv(out / "soft_two_part_stability_summary.csv", index=False)
    zero_pos.to_csv(out / "soft_two_part_zero_positive_diagnostics.csv", index=False)
    prob_diag.to_csv(out / "soft_two_part_probability_diagnostics.csv", index=False)
    pd.json_normalize(origin_audits).to_csv(out / "soft_two_part_origin_audit.csv", index=False)
    (out / "soft_two_part_decision_review.json").write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "run_type": "SOFT_TWO_PART_CHALLENGER_V01",
        "status": "PASS",
        "challenger_version": CHALLENGER_VERSION,
        "source_rolling_run_id": rolling_pointer["run_id"],
        "source_diagnosis_run_id": diagnosis_pointer["run_id"],
        "reference_model": "lightgbm_tweedie",
        "challenger_model": "soft_two_part_expected",
        "formula": "P(Y>0|X) * E(Y|Y>0,X)",
        "n_origins": int(len(origins)),
        "evaluation_rows": int(len(pred)),
        "selected_feature_count": int(len(selected)),
        "feature_parity": parity,
        "cumulative_3m_coverage": coverage,
        "decision": review,
        "devices": devices,
        "origin_audits": origin_audits,
        "safety": safety,
        "input_sha256": {
            "pair_panel": sha256_file(Path(pair_panel_path)),
            "canonical_pair_feature": sha256_file(Path(canonical_pair_feature_path)),
            "selected_feature_list": sha256_file(Path(selected_feature_path)),
            "model_contract": sha256_file(Path(model_contract_path)),
            "rolling_contract": sha256_file(Path(rolling_contract_path)),
            "challenger_contract": sha256_file(Path(challenger_contract_path)),
            "reference_predictions": sha256_file(Path(reference_prediction_path)),
        },
    }
    (out / "soft_two_part_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
