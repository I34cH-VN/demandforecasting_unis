from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATASET_VERSION = "dataset_v012"


def _text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _ascii_lower(v):
    s = _text(v)
    if not s:
        return None
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D").lower()


def normalize_status(v) -> str:
    s = _ascii_lower(v)
    if s in {"active", "hoat dong"}:
        return "active"
    if s in {"inactive", "vo hieu hoa"}:
        return "inactive"
    return "unknown"


def parse_base_sku(base_sku) -> Dict[str, object]:
    base = _text(base_sku)
    parts = base.split(".") if base else []
    g1_num = len(parts) >= 1 and bool(re.fullmatch(r"\d+", parts[0]))
    g3_num = len(parts) >= 3 and bool(re.fullmatch(r"\d+", parts[2]))
    structure_valid = len(parts) == 4 and g1_num and g3_num
    is_l1 = len(parts) >= 2 and parts[1] == "L1"
    return {
        "base_group_count": len(parts),
        "base_group1": parts[0] if len(parts) > 0 else None,
        "base_group2": parts[1] if len(parts) > 1 else None,
        "base_group3": parts[2] if len(parts) > 2 else None,
        "base_group4": parts[3] if len(parts) > 3 else None,
        "tile_size_code": parts[2] if len(parts) > 2 else None,
        "group1_numeric": g1_num,
        "group3_numeric": g3_num,
        "structure_valid": structure_valid,
        "is_l1": is_l1,
    }


def _assert_unique(df: pd.DataFrame, keys, label: str):
    if df.duplicated(keys).any():
        sample = df.loc[df.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise ValueError(f"{label} duplicate grain {keys}. sample={sample}")


def build_bridge(master_sku: pd.DataFrame) -> pd.DataFrame:
    required = {"bravo_sku", "base_sku", "master_status"}
    missing = required - set(master_sku.columns)
    if missing:
        raise ValueError(f"raw.master_sku missing {sorted(missing)}")

    df = master_sku.copy()
    df["bravo_sku_raw"] = df["bravo_sku"]
    df["base_sku_raw"] = df["base_sku"]
    df["bravo_sku"] = df["bravo_sku"].map(_text)
    df["base_sku"] = df["base_sku"].map(_text)
    if df["bravo_sku"].isna().any():
        raise ValueError("Blank Bravo SKU in master")
    _assert_unique(df, ["bravo_sku"], "master_sku")

    flags = pd.DataFrame([parse_base_sku(x) for x in df["base_sku"]], index=df.index)
    df = pd.concat([df, flags], axis=1)
    df["mapping_status"] = np.where(df["base_sku"].notna(), "MAPPED", "UNMAPPED")
    df["current_sku_status"] = df["master_status"].map(normalize_status)
    df["modeling_exclusion_reason"] = np.select(
        [
            df["mapping_status"].eq("UNMAPPED"),
            ~df["structure_valid"],
            df["structure_valid"] & ~df["is_l1"],
        ],
        ["UNMAPPED", "INVALID_STRUCTURE", "NON_L1"],
        default="ELIGIBLE",
    )
    df["modeling_universe_valid"] = df["modeling_exclusion_reason"].eq("ELIGIBLE")
    df["source_row_key"] = df["bravo_sku"]
    if "source_row_no" not in df.columns:
        df["source_row_no"] = pd.NA
    return df


def _single_non_null(s: pd.Series, field: str, base: str):
    vals = pd.unique(s.dropna())
    if len(vals) == 0:
        return None
    if len(vals) > 1:
        raise ValueError(f"BASE_ATTRIBUTE_CONFLICT base={base} field={field} values={vals[:10].tolist()}")
    return vals[0]


def build_dim_base(bridge: pd.DataFrame) -> pd.DataFrame:
    valid = bridge.loc[bridge["modeling_universe_valid"]].copy()
    stable = ["product_group", "brand", "price_group", "factory_code", "pull_source"]
    rows = []
    for base, g in valid.groupby("base_sku", sort=True):
        first = parse_base_sku(base)
        row = {
            "base_sku": base,
            **{k: first[k] for k in ["base_group1", "base_group2", "base_group3", "base_group4", "tile_size_code"]},
            "bravo_variant_count": int(g["bravo_sku"].nunique()),
            "current_active_variant_count": int(g["current_sku_status"].eq("active").sum()),
            "current_inactive_variant_count": int(g["current_sku_status"].eq("inactive").sum()),
            "current_unknown_variant_count": int(g["current_sku_status"].eq("unknown").sum()),
            "base_current_active": bool(g["current_sku_status"].eq("active").any()),
            "mixed_current_status_flag": bool(g["current_sku_status"].nunique(dropna=True) > 1),
        }
        for c in stable:
            if c in g.columns:
                row[c] = _single_non_null(g[c], c, base)
        rows.append(row)
    out = pd.DataFrame(rows)
    _assert_unique(out, ["base_sku"], "dim_base")
    return out.sort_values("base_sku").reset_index(drop=True)


def build_dim_branch(master_channel: pd.DataFrame) -> pd.DataFrame:
    required = {"branch_code", "master_status"}
    missing = required - set(master_channel.columns)
    if missing:
        raise ValueError(f"raw.master_channel missing {sorted(missing)}")
    df = master_channel.copy()
    df["branch_code"] = df["branch_code"].map(_text)
    if df["branch_code"].isna().any():
        raise ValueError("Blank branch_code in master_channel")
    _assert_unique(df, ["branch_code"], "master_channel")
    df["branch_current_status"] = df["master_status"].map(normalize_status)
    df["branch_current_active"] = df["branch_current_status"].eq("active")
    if "brand" in df.columns:
        df = df.rename(columns={"brand": "branch_brand"})
    keep = [
        "branch_code", "branch_name", "region", "branch_brand", "master_status",
        "branch_current_status", "branch_current_active", "source_file", "source_row_no", "loaded_at",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.sort_values("branch_code").reset_index(drop=True)


def _channel_relation(sku_channel, branch_brand) -> str:
    sc = _text(sku_channel)
    bb = _text(branch_brand)
    if not sc or not bb:
        return "UNKNOWN"
    scu = sc.upper()
    bbu = bb.upper()
    if scu == bbu:
        return "EXACT_MATCH"
    if scu == "BAN CHUNG":
        return "COMMON_CHANNEL"
    if scu == "HANG CT":
        return "PROJECT_CHANNEL"
    if "+" in scu:
        parts = [x.strip() for x in scu.split("+")]
        if bbu in parts:
            return "COMBO_MATCH"
    return "CROSS_CHANNEL"


def _observation_status(pos: bool, zero: bool, neg: bool) -> str:
    if pos and zero and neg:
        return "MIXED_ALL"
    if pos and neg:
        return "MIXED_POSITIVE_NEGATIVE"
    if pos and zero:
        return "MIXED_ZERO_POSITIVE"
    if zero and neg:
        return "MIXED_ZERO_NEGATIVE"
    if pos:
        return "POSITIVE_ONLY"
    if zero:
        return "EXPLICIT_ZERO_ONLY"
    if neg:
        return "NEGATIVE_ONLY"
    return "NO_SIGNAL"


def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    required = {"source_file", "source_row_no", "bravo_sku", "branch_code", "unit", "month", "total_quantity"}
    missing = required - set(sales.columns)
    if missing:
        raise ValueError(f"raw.sales_monthly missing {sorted(missing)}")
    sx = sales.copy()
    sx["bravo_sku_raw"] = sx["bravo_sku"]
    sx["branch_code_raw"] = sx["branch_code"]
    sx["bravo_sku"] = sx["bravo_sku"].map(_text)
    sx["branch_code"] = sx["branch_code"].map(_text)
    sx["unit_norm"] = sx["unit"].map(lambda v: _text(v).upper() if _text(v) else None)
    sx["month"] = pd.to_datetime(sx["month"], errors="coerce")
    sx["qty"] = pd.to_numeric(sx["total_quantity"], errors="coerce")
    sx["amount"] = pd.to_numeric(sx.get("total_amount"), errors="coerce") if "total_amount" in sx.columns else np.nan
    sx["line_count_num"] = pd.to_numeric(sx.get("line_count"), errors="coerce").fillna(0) if "line_count" in sx.columns else 0

    req_bad = sx["bravo_sku"].isna() | sx["branch_code"].isna() | sx["unit_norm"].isna() | sx["month"].isna() | sx["qty"].isna()
    sx["invalid_required_field"] = req_bad
    aligned = sx["month"].dt.day.eq(1) | sx["month"].isna()
    if (~aligned).any():
        raise ValueError("INVALID_MONTH_ALIGNMENT")

    natural = ["bravo_sku", "branch_code", "unit_norm", "month"]
    dup = sx.loc[~req_bad].duplicated(natural, keep=False)
    if dup.any():
        sample = sx.loc[dup, natural + ["source_file", "source_row_no"]].head(20).to_dict("records")
        raise ValueError(f"SOURCE_DUPLICATE_REVIEW_REQUIRED sample={sample}")
    return sx


def build_bravo_observation(sales: pd.DataFrame, bridge: pd.DataFrame, dim_branch: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sx = prepare_sales(sales)
    sx["high_level_bucket"] = np.where(sx["invalid_required_field"], "INVALID_REQUIRED_FIELD", np.where(sx["unit_norm"].eq("M2"), "M2_PENDING", "NON_M2"))
    m2 = sx.loc[sx["unit_norm"].eq("M2") & ~sx["invalid_required_field"]].copy()

    bcols = [
        "bravo_sku", "base_sku", "mapping_status", "structure_valid", "is_l1",
        "modeling_universe_valid", "modeling_exclusion_reason", "current_sku_status",
        "sku_name", "product_group", "brand", "branch_channel", "price_group", "factory_code",
        "factory_sku", "sale_sku", "replacement_sku", "base_group3", "tile_size_code",
    ]
    bcols = [c for c in bcols if c in bridge.columns]
    m2 = m2.merge(bridge[bcols], on="bravo_sku", how="left", validate="many_to_one", indicator="_sku_merge")

    br = dim_branch.copy()
    m2 = m2.merge(
        br[[c for c in ["branch_code", "branch_name", "region", "branch_brand", "branch_current_status", "branch_current_active"] if c in br.columns]],
        on="branch_code", how="left", validate="many_to_one", indicator="_branch_merge"
    )

    m2["mapping_status"] = np.where(m2["_sku_merge"].eq("left_only"), "UNMATCHED_MASTER", m2["mapping_status"])
    m2["branch_mapping_status"] = np.where(m2["_branch_merge"].eq("left_only"), "UNMATCHED_BRANCH", "MAPPED")
    m2["primary_exclusion_reason"] = np.select(
        [
            m2["mapping_status"].eq("UNMATCHED_MASTER"),
            m2["mapping_status"].eq("UNMAPPED"),
            m2["modeling_exclusion_reason"].eq("INVALID_STRUCTURE"),
            m2["modeling_exclusion_reason"].eq("NON_L1"),
            m2["branch_mapping_status"].eq("UNMATCHED_BRANCH"),
        ],
        ["UNMATCHED_MASTER", "UNMAPPED", "INVALID_STRUCTURE", "NON_L1", "UNMATCHED_BRANCH"],
        default="ELIGIBLE",
    )
    m2["row_disposition"] = np.where(m2["primary_exclusion_reason"].eq("ELIGIBLE"), "MODELING_VALID", "EXCLUDED")
    m2["secondary_flags"] = ""
    m2["positive_component_m2"] = m2["qty"].where(m2["qty"].gt(0), 0.0)
    m2["negative_component_m2"] = m2["qty"].where(m2["qty"].lt(0), 0.0)
    m2["observed_positive"] = m2["qty"].gt(0)
    m2["observed_explicit_zero"] = m2["qty"].eq(0)
    m2["observed_negative"] = m2["qty"].lt(0)
    m2["observation_status"] = [
        _observation_status(p, z, n) for p, z, n in zip(m2["observed_positive"], m2["observed_explicit_zero"], m2["observed_negative"])
    ]
    m2["sku_branch_channel_snapshot"] = m2.get("branch_channel")
    m2["branch_brand_snapshot"] = m2.get("branch_brand")
    m2["channel_relation_snapshot"] = [
        _channel_relation(a, b) for a, b in zip(m2["sku_branch_channel_snapshot"], m2["branch_brand_snapshot"])
    ]
    m2["current_sku_status_snapshot"] = m2.get("current_sku_status")
    m2["branch_current_status_snapshot"] = m2.get("branch_current_status")
    m2["gross_positive_m2"] = m2["positive_component_m2"]
    m2["negative_m2"] = m2["negative_component_m2"]
    m2["net_m2"] = m2["qty"]
    m2["total_amount_audit"] = m2["amount"]
    m2["source_line_count"] = m2["line_count_num"]
    m2["source_row_count"] = 1
    m2["source_file_count"] = 1

    # Reconciliation bucket for all source rows.
    audit = sx[[c for c in sx.columns if c not in []]].copy()
    return m2.sort_values(["bravo_sku", "branch_code", "month"]).reset_index(drop=True), audit


def build_fact(bravo_obs: pd.DataFrame) -> pd.DataFrame:
    valid = bravo_obs.loc[bravo_obs["row_disposition"].eq("MODELING_VALID")].copy()
    valid["observed_bravo_flag"] = True
    grain = ["base_sku", "branch_code", "month"]
    fact = valid.groupby(grain, as_index=False).agg(
        gross_positive_m2=("gross_positive_m2", "sum"),
        negative_m2=("negative_m2", "sum"),
        net_m2=("net_m2", "sum"),
        total_amount_audit=("total_amount_audit", "sum"),
        source_line_count=("source_line_count", "sum"),
        observed_source_row_count=("source_row_count", "sum"),
        observed_bravo_count=("bravo_sku", "nunique"),
        observed_positive=("observed_positive", "any"),
        observed_explicit_zero=("observed_explicit_zero", "any"),
        observed_negative=("observed_negative", "any"),
    )
    fact["observation_status"] = [
        _observation_status(p, z, n) for p, z, n in zip(fact["observed_positive"], fact["observed_explicit_zero"], fact["observed_negative"])
    ]
    _assert_unique(fact, grain, "fact")
    return fact.sort_values(grain).reset_index(drop=True)


def _month_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    # a - b in whole calendar months
    aa = pd.to_datetime(a)
    bb = pd.to_datetime(b)
    return (aa.dt.year - bb.dt.year) * 12 + (aa.dt.month - bb.dt.month)


def build_pair_panel(fact: pd.DataFrame, dim_base: pd.DataFrame, dim_branch: pd.DataFrame, panel_end: pd.Timestamp) -> pd.DataFrame:
    if fact.empty:
        raise ValueError("No valid M2 fact rows")
    first = fact.groupby(["base_sku", "branch_code"], as_index=False)["month"].min().rename(columns={"month": "first_observed_month"})
    frames = []
    for r in first.itertuples(index=False):
        months = pd.date_range(r.first_observed_month, panel_end, freq="MS")
        frames.append(pd.DataFrame({"base_sku": r.base_sku, "branch_code": r.branch_code, "month": months, "first_observed_month": r.first_observed_month}))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(fact, on=["base_sku", "branch_code", "month"], how="left", validate="one_to_one")

    # V0.1.2 business rule: once a Pair is known and the month is inside the closed
    # panel calendar, absence of a sales row means observed zero gross demand.
    # Preserve whether a raw/source row existed separately for lineage/audit.
    panel["source_row_observed"] = panel["observed_source_row_count"].notna()
    implicit_zero = ~panel["source_row_observed"]

    for c in ["gross_positive_m2", "negative_m2", "net_m2", "total_amount_audit", "source_line_count"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
    for c in ["observed_source_row_count", "observed_bravo_count"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0).astype(int)
    for c in ["observed_positive", "observed_explicit_zero", "observed_negative"]:
        if c in panel.columns:
            panel[c] = panel[c].astype("boolean").fillna(False).astype(bool)
    panel["observation_status"] = panel["observation_status"].fillna("IMPLICIT_ZERO_NO_SOURCE_ROW")

    # Dense closed-month panel rows are observed even when the Pair had no transaction.
    panel["actual_observed"] = True
    panel["actual_positive"] = panel["observed_positive"].astype(bool)
    panel["actual_explicit_zero"] = panel["observation_status"].eq("EXPLICIT_ZERO_ONLY")
    panel["actual_negative_only"] = panel["observation_status"].eq("NEGATIVE_ONLY")

    # Negative-only rows remain return/adjustment evidence and are not exposed as
    # trainable zero-demand targets. No-source rows, however, are trainable zeros.
    panel["target_available"] = ~panel["actual_negative_only"]
    panel["actual_gross_m2"] = panel["gross_positive_m2"].where(panel["target_available"], np.nan)
    panel["zero_semantics"] = np.select(
        [
            implicit_zero,
            panel["actual_explicit_zero"],
            panel["actual_negative_only"],
        ],
        ["OBSERVED_ZERO_IMPLICIT", "OBSERVED_ZERO", "NEGATIVE_ONLY_GROSS_ZERO"],
        default="OBSERVED_NONZERO_OR_MIXED",
    )
    panel["known_pair"] = True

    keys = ["base_sku", "branch_code"]
    panel = panel.sort_values(keys + ["month"]).reset_index(drop=True)
    panel["_obs_month"] = panel["month"].where(panel["actual_observed"])
    panel["_pos_month"] = panel["month"].where(panel["actual_positive"])
    panel["last_observed_month_to_date"] = panel.groupby(keys)["_obs_month"].ffill()
    panel["last_positive_month_to_date"] = panel.groupby(keys)["_pos_month"].ffill()
    # Origin-safe first-positive lifecycle: before the first positive event, the field must remain NULL.
    first_pos = panel.loc[panel["actual_positive"]].groupby(keys, as_index=False)["month"].min().rename(columns={"month": "_first_positive_month_final"})
    panel = panel.merge(first_pos, on=keys, how="left", validate="many_to_one")
    positive_seen_to_date = panel.groupby(keys)["actual_positive"].cummax()
    panel["first_positive_month_to_date"] = panel["_first_positive_month_final"].where(positive_seen_to_date)
    panel["months_since_first_observed"] = _month_diff(panel["month"], panel["first_observed_month"])
    panel["months_since_first_positive"] = _month_diff(panel["month"], panel["first_positive_month_to_date"])
    panel["months_since_last_observed"] = _month_diff(panel["month"], panel["last_observed_month_to_date"])
    panel["months_since_last_positive"] = _month_diff(panel["month"], panel["last_positive_month_to_date"])
    panel["observed_month_count_to_date"] = panel.groupby(keys)["actual_observed"].cumsum().astype(int)
    panel["positive_month_count_to_date"] = panel.groupby(keys)["actual_positive"].cumsum().astype(int)

    panel = panel.merge(dim_base, on="base_sku", how="left", validate="many_to_one")
    panel = panel.merge(dim_branch, on="branch_code", how="left", validate="many_to_one", suffixes=("", "_branchdim"))
    panel["pair_dataset_version"] = DATASET_VERSION
    panel = panel.drop(columns=["_obs_month", "_pos_month", "_first_positive_month_final"], errors="ignore")
    _assert_unique(panel, ["base_sku", "branch_code", "month"], "pair_panel")
    return panel


def _concentration(g: pd.DataFrame) -> pd.Series:
    x = g.loc[g["gross_positive_m2"].gt(0), "gross_positive_m2"].sort_values(ascending=False)
    total = x.sum()
    if total <= 0:
        return pd.Series({"pair_top1_share": np.nan, "pair_top5_share": np.nan, "pair_top10_share": np.nan, "sku_top1_share": np.nan, "sku_top5_share": np.nan, "pair_hhi": np.nan})
    shares = x / total
    return pd.Series({
        "pair_top1_share": float(shares.head(1).sum()),
        "pair_top5_share": float(shares.head(5).sum()),
        "pair_top10_share": float(shares.head(10).sum()),
        "sku_top1_share": float(shares.head(1).sum()),
        "sku_top5_share": float(shares.head(5).sum()),
        "pair_hhi": float((shares ** 2).sum()),
    })


def build_branch_panel(fact: pd.DataFrame, dim_branch: pd.DataFrame, panel_end: pd.Timestamp) -> pd.DataFrame:
    f = fact.copy().sort_values(["base_sku", "branch_code", "month"])
    f["positive_pair"] = f["gross_positive_m2"].gt(0)
    f["explicit_zero_pair"] = f["observation_status"].eq("EXPLICIT_ZERO_ONLY")
    f["negative_only_pair"] = f["observation_status"].eq("NEGATIVE_ONLY")

    pos = f.loc[f["positive_pair"], ["base_sku", "branch_code", "month"]].copy()
    pos = pos.sort_values(["base_sku", "branch_code", "month"])
    pos["prev_positive_month"] = pos.groupby(["base_sku", "branch_code"])["month"].shift(1)
    pos["positive_gap_months"] = _month_diff(pos["month"], pos["prev_positive_month"])
    first_pos = pos.groupby(["base_sku", "branch_code"])["month"].transform("min")
    pos["new_positive_pair"] = pos["month"].eq(first_pos)
    pos["reactivated_pair"] = pos["prev_positive_month"].notna() & pos["positive_gap_months"].ge(2)
    f = f.merge(pos[["base_sku", "branch_code", "month", "new_positive_pair", "reactivated_pair"]], on=["base_sku", "branch_code", "month"], how="left")
    f["new_positive_pair"] = f["new_positive_pair"].astype("boolean").fillna(False).astype(bool)
    f["reactivated_pair"] = f["reactivated_pair"].astype("boolean").fillna(False).astype(bool)

    agg = f.groupby(["branch_code", "month"], as_index=False).agg(
        branch_gross_m2=("gross_positive_m2", "sum"),
        branch_negative_m2=("negative_m2", "sum"),
        branch_net_m2=("net_m2", "sum"),
        observed_pair_count=("base_sku", "nunique"),
        positive_pair_count=("positive_pair", "sum"),
        explicit_zero_pair_count=("explicit_zero_pair", "sum"),
        negative_only_pair_count=("negative_only_pair", "sum"),
        observed_base_sku_count=("base_sku", "nunique"),
        positive_base_sku_count=("positive_pair", "sum"),
        observed_bravo_count=("observed_bravo_count", "sum"),
        new_positive_pair_count=("new_positive_pair", "sum"),
        reactivated_pair_count=("reactivated_pair", "sum"),
    )
    agg["new_positive_sku_count"] = agg["new_positive_pair_count"]

    conc_rows = []
    for (branch_code, month), g in f.groupby(["branch_code", "month"], sort=True):
        r = _concentration(g).to_dict()
        r.update({"branch_code": branch_code, "month": month})
        conc_rows.append(r)
    conc = pd.DataFrame(conc_rows)
    agg = agg.merge(conc, on=["branch_code", "month"], how="left", validate="one_to_one")

    first = agg.groupby("branch_code", as_index=False)["month"].min().rename(columns={"month": "branch_first_observed_month"})
    frames = []
    for r in first.itertuples(index=False):
        months = pd.date_range(r.branch_first_observed_month, panel_end, freq="MS")
        frames.append(pd.DataFrame({"branch_code": r.branch_code, "month": months, "branch_first_observed_month": r.branch_first_observed_month}))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(agg, on=["branch_code", "month"], how="left", validate="one_to_one")
    panel["branch_source_row_observed"] = panel["observed_pair_count"].notna()
    # A closed Branch-month with no sales rows is a valid zero month.
    zero_fill_cols = [
        "branch_gross_m2", "branch_negative_m2", "branch_net_m2",
        "observed_pair_count", "positive_pair_count", "explicit_zero_pair_count",
        "negative_only_pair_count", "observed_base_sku_count", "positive_base_sku_count",
        "observed_bravo_count", "new_positive_pair_count", "reactivated_pair_count",
        "new_positive_sku_count",
    ]
    for c in zero_fill_cols:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)
    panel["branch_observed"] = True
    panel["branch_positive"] = panel["branch_gross_m2"].gt(0)
    panel = panel.sort_values(["branch_code", "month"]).reset_index(drop=True)
    panel["branch_observed_month_count_to_date"] = panel.groupby("branch_code")["branch_observed"].cumsum().astype(int)
    panel["branch_positive_month_count_to_date"] = panel.groupby("branch_code")["branch_positive"].cumsum().astype(int)
    panel = panel.merge(dim_branch, on="branch_code", how="left", validate="many_to_one")
    panel["branch_dataset_version"] = DATASET_VERSION
    _assert_unique(panel, ["branch_code", "month"], "branch_panel")
    return panel


def build_calendar(start_month: pd.Timestamp, end_month: pd.Timestamp) -> pd.DataFrame:
    months = pd.date_range(start_month, end_month, freq="MS")
    cal = pd.DataFrame({"month": months})
    cal["month_num"] = cal["month"].dt.month
    cal["quarter"] = cal["month"].dt.quarter
    cal["year"] = cal["month"].dt.year
    cal["days_in_month"] = cal["month"].dt.days_in_month
    _assert_unique(cal, ["month"], "calendar")
    return cal


def validate_core(sales_raw: pd.DataFrame, bravo_obs: pd.DataFrame, bridge: pd.DataFrame, dim_base: pd.DataFrame, dim_branch: pd.DataFrame, fact: pd.DataFrame, pair: pd.DataFrame, branch: pd.DataFrame, calendar: pd.DataFrame) -> Dict[str, object]:
    checks = {}
    def add(name, ok, detail=None):
        checks[name] = {"pass": bool(ok), "detail": detail}

    add("bridge_unique", ~bridge.duplicated(["bravo_sku"]).any())
    add("dim_base_unique", ~dim_base.duplicated(["base_sku"]).any())
    add("dim_branch_unique", ~dim_branch.duplicated(["branch_code"]).any())
    add("fact_unique", ~fact.duplicated(["base_sku", "branch_code", "month"]).any())
    add("pair_unique", ~pair.duplicated(["base_sku", "branch_code", "month"]).any())
    add("branch_unique", ~branch.duplicated(["branch_code", "month"]).any())
    add("calendar_unique", ~calendar.duplicated(["month"]).any())
    add("fact_base_subset_dim", set(fact["base_sku"]).issubset(set(dim_base["base_sku"])))
    add("gross_nonnegative", bool(fact["gross_positive_m2"].ge(0).all()))
    add("negative_nonpositive", bool(fact["negative_m2"].le(0).all()))
    implicit = ~pair["source_row_observed"]
    add("implicit_missing_pair_is_observed_zero", bool(
        pair.loc[implicit, "actual_observed"].all()
        and pair.loc[implicit, "target_available"].all()
        and pair.loc[implicit, "actual_gross_m2"].eq(0).all()
        and pair.loc[implicit, "zero_semantics"].eq("OBSERVED_ZERO_IMPLICIT").all()
    ))
    add("dense_branch_month_is_observed", bool(branch["branch_observed"].all()))
    neg_only = pair["actual_negative_only"]
    add("negative_only_target_unavailable", bool((~pair.loc[neg_only, "target_available"]).all() and pair.loc[neg_only, "actual_gross_m2"].isna().all()))
    fp = pair["first_positive_month_to_date"]
    add("first_positive_origin_safe", bool((fp.isna() | fp.le(pair["month"])).all()))

    eligible = bravo_obs.loc[bravo_obs["row_disposition"].eq("MODELING_VALID")]
    src_pos = float(eligible["gross_positive_m2"].sum())
    fact_pos = float(fact["gross_positive_m2"].sum())
    src_neg = float(eligible["negative_m2"].sum())
    fact_neg = float(fact["negative_m2"].sum())
    add("positive_quantity_reconciles", np.isclose(src_pos, fact_pos, rtol=0, atol=1e-8), {"source": src_pos, "fact": fact_pos})
    add("negative_quantity_reconciles", np.isclose(src_neg, fact_neg, rtol=0, atol=1e-8), {"source": src_neg, "fact": fact_neg})

    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAIL"
    return {"status": status, "checks": checks}


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
