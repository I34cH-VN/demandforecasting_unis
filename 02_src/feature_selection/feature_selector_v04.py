from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SELECTION_VERSION = "feature_selection_v04"
PAIR_FEATURE_VERSION = "pair_feature_v013"
BRANCH_FEATURE_VERSION = "branch_feature_v013"
DATASET_VERSION = "dataset_v012"


@dataclass(frozen=True)
class SelectionConfig:
    missing_rate_drop: float = 0.95
    corr_review_threshold: float = 0.995
    corr_sample_rows: int = 50000
    permutation_repeats: int = 2
    min_perm_wape_gain: float = 0.0005
    pair_min_selected: int = 12
    branch_min_selected: int = 8
    random_state: int = 42
    pair_max_train_rows: int = 180000
    branch_max_train_rows: int = 10000
    screening_max_iter: int = 160
    screening_learning_rate: float = 0.06
    screening_max_leaf_nodes: int = 31
    screening_l2: float = 1.0


PAIR_TARGET = "target_actual_gross_m2"
BRANCH_TARGET = "target_branch_gross_m2"


PAIR_FAMILIES = {
    "pair_history": (
        "pair_lag_", "pair_roll_", "pair_history_", "pair_target_", "pair_positive_",
        "months_since_", "pair_last_", "pair_mean_", "pair_peak_"
    ),
    "demand_state": ("pair_adi_", "pair_cv2_"),
    "sku_cross_branch": ("sku_", "cross_branch_"),
    "size_branch": ("size_branch_", "tile_size_code"),
    "calendar": (
        "target_month_num", "target_quarter", "target_year", "target_days_in_month",
        "target_month_sin", "target_month_cos", "target_weekday_count",
        "target_public_holiday_event_count", "target_statutory_holiday_nominal_days", "target_working_days_proxy",
        "target_lunar_month_mid", "target_lunar_month_mid_is_leap", "target_lunar_month_sin", "target_lunar_month_cos",
        "target_tet_day_of_month", "target_days_to_next_tet_from_midmonth", "target_days_since_prev_tet_from_midmonth",
        "target_is_tet_month", "target_is_pre_tet_month", "target_is_post_tet_month", "horizon"
    ),
}

BRANCH_FAMILIES = {
    "branch_history": (
        "branch_lag_", "branch_roll_", "branch_history_", "branch_observed_month_count_",
        "branch_positive_month_count_", "branch_positive_rate_", "months_since_last_branch_positive"
    ),
    "branch_composition": (
        "branch_observed_pair_", "branch_positive_pair_", "branch_explicit_zero_", "branch_negative_only_",
        "branch_new_positive_", "branch_reactivated_", "branch_average_"
    ),
    "branch_concentration": ("branch_pair_",),
    "calendar": (
        "target_month_num", "target_quarter", "target_year", "target_days_in_month",
        "target_month_sin", "target_month_cos", "target_weekday_count",
        "target_public_holiday_event_count", "target_statutory_holiday_nominal_days", "target_working_days_proxy",
        "target_lunar_month_mid", "target_lunar_month_mid_is_leap", "target_lunar_month_sin", "target_lunar_month_cos",
        "target_tet_day_of_month", "target_days_to_next_tet_from_midmonth", "target_days_since_prev_tet_from_midmonth",
        "target_is_tet_month", "target_is_pre_tet_month", "target_is_post_tet_month", "horizon"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    denom = np.abs(y_true).sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _neg_wape_scorer(estimator, X, y) -> float:
    return -wape(np.asarray(y, dtype=float), estimator.predict(X))


def _safe_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:  # sklearn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def build_screening_pipeline(X: pd.DataFrame, cfg: SelectionConfig) -> Pipeline:
    categorical = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("category"))]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if categorical:
        transformers.append(("cat", _safe_one_hot_encoder(), categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=cfg.screening_max_iter,
        learning_rate=cfg.screening_learning_rate,
        max_leaf_nodes=cfg.screening_max_leaf_nodes,
        l2_regularization=cfg.screening_l2,
        random_state=cfg.random_state,
        early_stopping=False,
    )
    return Pipeline([("prep", prep), ("model", model)])


def deterministic_cap(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    # Preserve horizon and target-positive composition before deterministic sampling.
    work = df.copy()
    pos = work.iloc[:, 0:0].copy()  # index-only helper
    if "horizon" in work.columns:
        strata = work["horizon"].astype(str)
    else:
        strata = pd.Series("ALL", index=work.index)
    if "_target_for_sampling" in work.columns:
        strata = strata + "_" + work["_target_for_sampling"].gt(0).astype(int).astype(str)
    parts = []
    counts = strata.value_counts()
    for key, n in counts.items():
        idx = strata[strata.eq(key)].index
        take = max(1, int(round(max_rows * n / len(work))))
        take = min(take, len(idx))
        parts.append(work.loc[idx].sample(n=take, random_state=random_state))
    out = pd.concat(parts).drop_duplicates().sort_index()
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    elif len(out) < max_rows:
        remaining = work.drop(index=out.index)
        need = min(max_rows - len(out), len(remaining))
        if need:
            out = pd.concat([out, remaining.sample(n=need, random_state=random_state)]).sort_index()
    return out


def _series_fingerprint(s: pd.Series) -> str:
    # Hash pandas-normalized values; exact Series.equals verification is still required on collisions.
    hv = pd.util.hash_pandas_object(s, index=False, categorize=True).values
    h = hashlib.sha256()
    h.update(str(s.dtype).encode("utf-8"))
    h.update(hv.tobytes())
    return h.hexdigest()


def exact_duplicate_map(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {}
    duplicates: Dict[str, str] = {}
    for c in columns:
        fp = _series_fingerprint(df[c])
        prior = buckets.setdefault(fp, [])
        matched = None
        for p in prior:
            if df[c].equals(df[p]):
                matched = p
                break
        if matched is not None:
            duplicates[c] = matched
        else:
            prior.append(c)
    return duplicates


def correlation_review(df: pd.DataFrame, columns: Sequence[str], cfg: SelectionConfig) -> pd.DataFrame:
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    sample = df[numeric]
    if len(sample) > cfg.corr_sample_rows:
        sample = sample.sample(n=cfg.corr_sample_rows, random_state=cfg.random_state)
    corr = sample.corr(numeric_only=True).abs()
    rows = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and float(v) >= cfg.corr_review_threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_corr": float(v)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])


def infer_family(feature: str, family_map: Dict[str, Tuple[str, ...]]) -> str:
    for family, patterns in family_map.items():
        for p in patterns:
            if feature == p or feature.startswith(p):
                return family
    return "other"


def static_screen(
    train_df: pd.DataFrame,
    candidate_features: Sequence[str],
    cfg: SelectionConfig,
) -> Tuple[List[str], pd.DataFrame, Dict[str, str]]:
    rows = []
    prelim = []
    for c in candidate_features:
        missing = float(train_df[c].isna().mean())
        unique = int(train_df[c].nunique(dropna=True))
        if missing > cfg.missing_rate_drop:
            status = "DROP_MISSING_RATE"
        elif unique <= 1:
            status = "DROP_CONSTANT"
        else:
            status = "PASS_STATIC"
            prelim.append(c)
        rows.append({"feature_name": c, "train_missing_rate": missing, "train_n_unique": unique, "static_status": status})
    dup = exact_duplicate_map(train_df, prelim)
    for r in rows:
        c = r["feature_name"]
        if c in dup:
            r["static_status"] = "DROP_EXACT_DUPLICATE"
            r["duplicate_of"] = dup[c]
        else:
            r["duplicate_of"] = None
    kept = [c for c in prelim if c not in dup]
    return kept, pd.DataFrame(rows), dup


def fit_and_score(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    cfg: SelectionConfig,
    max_train_rows: int,
) -> Tuple[Pipeline, Dict[str, float]]:
    t = train[list(features) + [target, "horizon"]].copy() if "horizon" not in features else train[list(features) + [target]].copy()
    t["_target_for_sampling"] = train[target].to_numpy()
    t = deterministic_cap(t, max_train_rows, cfg.random_state)
    y_train = t[target].astype(float)
    X_train = t[list(features)]
    X_val = val[list(features)]
    y_val = val[target].astype(float).to_numpy()
    pipe = build_screening_pipeline(X_train, cfg)
    pipe.fit(X_train, y_train)
    pred = np.maximum(pipe.predict(X_val), 0.0)
    metrics: Dict[str, float] = {"wape": wape(y_val, pred)}
    for h in (1, 2, 3):
        m = val["horizon"].eq(h).to_numpy()
        metrics[f"h{h}_wape"] = wape(y_val[m], pred[m]) if m.any() else float("nan")
    return pipe, metrics


def score_fitted_pipeline(pipe: Pipeline, val: pd.DataFrame, features: Sequence[str], target: str) -> Dict[str, float]:
    if val.empty:
        return {"wape": float("nan"), "h1_wape": float("nan"), "h2_wape": float("nan"), "h3_wape": float("nan")}
    y_val = val[target].astype(float).to_numpy()
    pred = np.maximum(pipe.predict(val[list(features)]), 0.0)
    metrics: Dict[str, float] = {"wape": wape(y_val, pred)}
    for h in (1, 2, 3):
        m = val["horizon"].eq(h).to_numpy()
        metrics[f"h{h}_wape"] = wape(y_val[m], pred[m]) if m.any() else float("nan")
    return metrics


def permutation_report(
    pipe: Pipeline,
    val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    cfg: SelectionConfig,
) -> pd.DataFrame:
    rows = []
    scopes = [("ALL", val)] + [(f"h{h}", val.loc[val["horizon"].eq(h)]) for h in (1, 2, 3)]
    for scope, d in scopes:
        if d.empty:
            continue
        result = permutation_importance(
            pipe,
            d[list(features)],
            d[target].astype(float),
            scoring=_neg_wape_scorer,
            n_repeats=cfg.permutation_repeats,
            random_state=cfg.random_state,
            n_jobs=1,
        )
        for c, mean, std in zip(features, result.importances_mean, result.importances_std):
            rows.append({"scope": scope, "feature_name": c, "importance_wape_mean": float(mean), "importance_wape_std": float(std)})
    return pd.DataFrame(rows)


def family_ablation_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    family_map: Dict[str, Tuple[str, ...]],
    full_metrics: Dict[str, float],
    cfg: SelectionConfig,
    max_train_rows: int,
) -> pd.DataFrame:
    rows = []
    for family in family_map:
        removed = [c for c in features if infer_family(c, family_map) == family]
        kept = [c for c in features if c not in removed]
        if not removed or not kept:
            continue
        _, metrics = fit_and_score(train, val, kept, target, cfg, max_train_rows)
        row = {
            "family": family,
            "removed_feature_count": len(removed),
            "full_wape": full_metrics["wape"],
            "ablated_wape": metrics["wape"],
            "delta_wape_ablated_minus_full": metrics["wape"] - full_metrics["wape"],
        }
        for h in (1, 2, 3):
            row[f"h{h}_delta_wape"] = metrics[f"h{h}_wape"] - full_metrics[f"h{h}_wape"]
        rows.append(row)
    return pd.DataFrame(rows)


def choose_features(
    static_kept: Sequence[str],
    perm: pd.DataFrame,
    family_map: Dict[str, Tuple[str, ...]],
    cfg: SelectionConfig,
    min_selected: int,
) -> Tuple[List[str], pd.DataFrame]:
    pivot = perm.pivot_table(index="feature_name", columns="scope", values="importance_wape_mean", aggfunc="mean")
    for col in ["ALL", "h1", "h2", "h3"]:
        if col not in pivot:
            pivot[col] = np.nan
    pivot = pivot[["ALL", "h1", "h2", "h3"]]
    pivot["best_importance"] = pivot.max(axis=1, skipna=True)
    pivot["selected_by_importance"] = (
        pivot["ALL"].fillna(-np.inf).gt(cfg.min_perm_wape_gain)
        | pivot[["h1", "h2", "h3"]].max(axis=1, skipna=True).fillna(-np.inf).gt(cfg.min_perm_wape_gain)
    )
    ranked = pivot.sort_values(["selected_by_importance", "best_importance"], ascending=[False, False])
    selected = [c for c in ranked.index if c in static_kept and bool(ranked.loc[c, "selected_by_importance"])]
    if len(selected) < min_selected:
        for c in ranked.index:
            if c in static_kept and c not in selected:
                selected.append(c)
                if len(selected) >= min_selected:
                    break
    # Horizon is structural for the pooled model family; Direct models may drop it as constant per horizon.
    if "horizon" in static_kept and "horizon" not in selected:
        selected.append("horizon")
    decision = pivot.reset_index()
    decision["family"] = decision["feature_name"].map(lambda c: infer_family(c, family_map))
    decision["selected"] = decision["feature_name"].isin(selected)
    decision["selection_reason"] = np.where(
        decision["feature_name"].eq("horizon") & decision["selected"],
        "MANDATORY_FOR_POOLED",
        np.where(decision["selected_by_importance"], "PERMUTATION_IMPORTANCE", np.where(decision["selected"], "MIN_FEATURE_FLOOR", "NOT_SELECTED")),
    )
    return selected, decision.sort_values(["selected", "best_importance"], ascending=[False, False]).reset_index(drop=True)


def select_track(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    track: str,
    target: str,
    family_map: Dict[str, Tuple[str, ...]],
    cfg: SelectionConfig,
    max_train_rows: int,
    min_selected: int,
) -> Dict[str, object]:
    cand_flag = inventory["initial_model_candidate"].map(lambda x: x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() == "true")
    candidates = inventory.loc[(inventory["track"].eq(track)) & cand_flag, "feature_name"].tolist()
    candidates = [c for c in candidates if c in df.columns]
    train = df.loc[df["historical_train_mask"].astype(bool) & df["target_available"].astype(bool)].copy()
    val_all = df.loc[df["official_validation_mask"].astype(bool) & df["target_available"].astype(bool)].copy()

    # WORK9 business objective: model selection is evaluated on the CURRENT portfolio.
    # Current status is a row-selection mask only; it is never a predictive feature and never rewrites TRAIN history.
    if "current_production_forecast_mask" not in val_all.columns:
        raise ValueError(f"{track}: current_production_forecast_mask missing from feature panel")
    primary_mask = val_all["current_production_forecast_mask"].fillna(False).astype(bool)
    if track == "PAIR":
        if "known_pair_asof_origin" not in val_all.columns:
            raise ValueError("PAIR: known_pair_asof_origin missing from feature panel")
        primary_mask &= val_all["known_pair_asof_origin"].fillna(False).astype(bool)
    val = val_all.loc[primary_mask].copy()

    if train.empty or val.empty:
        raise ValueError(f"{track}: train/current-active validation eligible rows missing")
    # Hard safety: Stage 6 development artifact cannot contain Frozen-Test targets.
    if pd.to_datetime(df["target_month"]).ge(pd.Timestamp("2026-04-01")).any():
        raise ValueError(f"{track}: Frozen Test target detected in Stage 6 input")

    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(train[c])]
    inf_cols = [c for c in numeric_candidates if np.isinf(pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).any()]
    if inf_cols:
        raise ValueError(f"{track}: non-finite infinite candidate values detected: {inf_cols}")
    if not np.isfinite(pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"{track}: non-finite TRAIN target detected")
    if not np.isfinite(pd.to_numeric(val[target], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"{track}: non-finite validation target detected")

    static_kept, static_report, duplicate_map = static_screen(train, candidates, cfg)
    if not static_kept:
        raise ValueError(f"{track}: no features remain after static screening")
    corr = correlation_review(train, static_kept, cfg)
    pipe, full_metrics = fit_and_score(train, val, static_kept, target, cfg, max_train_rows)
    secondary_all_metrics = score_fitted_pipeline(pipe, val_all, static_kept, target)
    perm = permutation_report(pipe, val, static_kept, target, cfg)
    ablation = family_ablation_report(train, val, static_kept, target, family_map, full_metrics, cfg, max_train_rows)
    selected, decision = choose_features(static_kept, perm, family_map, cfg, min_selected)

    quality = static_report.merge(
        decision[["feature_name", "family", "ALL", "h1", "h2", "h3", "best_importance", "selected", "selection_reason"]],
        on="feature_name", how="left",
    )
    quality["selected"] = quality["selected"].astype("boolean").fillna(False).astype(bool)
    return {
        "selected": selected,
        "candidate_features": candidates,
        "static_kept": static_kept,
        "static_report": static_report,
        "correlation_review": corr,
        "permutation_importance": perm,
        "family_ablation": ablation,
        "quality_report": quality,
        "full_metrics": full_metrics,
        "secondary_all_validation_metrics": secondary_all_metrics,
        "duplicate_map": duplicate_map,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "validation_rows_primary_current_active": int(len(val)),
        "validation_rows_secondary_all": int(len(val_all)),
        "validation_policy": "CURRENT_ACTIVE_KNOWN_PAIR" if track == "PAIR" else "CURRENT_ACTIVE_BRANCH",
    }


CALENDAR_V013_CANDIDATES = {
    "target_month_num", "target_quarter", "target_year", "target_days_in_month",
    "target_month_sin", "target_month_cos", "target_weekday_count",
    "target_public_holiday_event_count", "target_statutory_holiday_nominal_days", "target_working_days_proxy",
    "target_lunar_month_mid", "target_lunar_month_mid_is_leap", "target_lunar_month_sin", "target_lunar_month_cos",
    "target_tet_day_of_month", "target_days_to_next_tet_from_midmonth", "target_days_since_prev_tet_from_midmonth",
    "target_is_tet_month", "target_is_pre_tet_month", "target_is_post_tet_month",
}

def validate_selection_result(result: Dict[str, object], track: str, min_selected: int) -> Dict[str, object]:
    selected = result["selected"]
    quality = result["quality_report"]
    checks = {
        "nonempty_selected_list": len(selected) >= min_selected,
        "selected_subset_static_kept": set(selected).issubset(set(result["static_kept"])),
        "no_constant_selected": not quality.loc[quality["selected"], "static_status"].eq("DROP_CONSTANT").any(),
        "no_high_missing_selected": not quality.loc[quality["selected"], "static_status"].eq("DROP_MISSING_RATE").any(),
        "no_exact_duplicate_selected": not quality.loc[quality["selected"], "static_status"].eq("DROP_EXACT_DUPLICATE").any(),
        "horizon_present_for_pooled": "horizon" in selected,
        "validation_wape_finite": np.isfinite(result["full_metrics"]["wape"]),
        "primary_validation_nonempty": int(result.get("validation_rows_primary_current_active", 0)) > 0,
        "secondary_validation_not_smaller": int(result.get("validation_rows_secondary_all", 0)) >= int(result.get("validation_rows_primary_current_active", 0)),
        "calendar_v013_candidates_seen": CALENDAR_V013_CANDIDATES.issubset(set(result.get("candidate_features", []))),
        "no_target_or_snapshot_selected": not any((c.endswith("_snapshot") or c.startswith("target_actual") or c.startswith("target_branch") or c in {"target_available", "base_sku", "branch_code", "forecast_origin", "target_month"}) for c in selected),
    }
    # Normalize numpy.bool_ / pandas scalar booleans to native Python bool so
    # validation objects are JSON-serializable and stable in manifests.
    checks = {k: bool(v) for k, v in checks.items()}
    return {"track": track, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def write_yaml_feature_list(path: Path, track: str, selected: Sequence[str], source_run_id: str, selection_run_id: str):
    # Minimal YAML writer keeps notebook dependency-free beyond PyYAML already installed by Colab in most cases.
    lines = [
        f"version: {SELECTION_VERSION}",
        f"track: {track}",
        f"source_feature_run_id: {source_run_id}",
        f"selection_run_id: {selection_run_id}",
        "status: CANDIDATE_PENDING_AUDIT",
        "selected_features:",
    ]
    lines.extend([f"  - {x}" for x in selected])
    lines.extend([
        "stage7_notes:",
        "  pooled_model_requires_horizon: true",
        "  direct_per_horizon_model_may_drop_horizon_as_constant: true",
        "  frozen_test_used: false",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
