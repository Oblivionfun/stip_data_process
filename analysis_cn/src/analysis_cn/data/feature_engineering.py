from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from ..utils.io import resolve_output, save_json


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_likert(series: pd.Series, min_score: float, max_score: float) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") - min_score) / (max_score - min_score)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.divide(den).replace([pd.NA, pd.NaT], pd.NA)


def _one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        series = series.astype("Int64").astype("string")
    else:
        series = series.astype("string")
    dummies = pd.get_dummies(series, prefix=col, dummy_na=True)
    return dummies


def _infer_base_dir(config_path: Path) -> Path:
    parent = config_path.parent
    if (parent / "data").exists():
        return parent
    if parent.name == "config" and parent.parent.exists():
        return parent.parent
    return parent


def run_feature_engineering(config_path: str | Path) -> Dict[str, str]:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    base_dir = _infer_base_dir(config_path)

    input_path = _resolve_path(cfg["clean_data_path"], base_dir)
    df = _load_dataframe(input_path)

    params = cfg.get("feature_params", {})
    commute = params.get("commute_columns", {})
    delay_cols = params.get("delay_columns", {})
    likert_cfg = params.get("likert_scales", {"min_score": 1, "max_score": 5})
    info_cols = params.get("info_columns", [])

    features = df.copy()

    m_col = commute.get("morning")
    e_col = commute.get("evening")
    cm_col = commute.get("congestion_morning")
    ce_col = commute.get("congestion_evening")

    if m_col in features.columns and e_col in features.columns:
        morning = pd.to_numeric(features[m_col], errors="coerce")
        evening = pd.to_numeric(features[e_col], errors="coerce")
        features["delta_commute_minutes"] = evening - morning
        features["avg_commute_minutes"] = pd.concat([morning, evening], axis=1).mean(axis=1)
    if cm_col in features.columns and m_col in features.columns:
        features["congestion_ratio_morning"] = _safe_divide(features[cm_col], features[m_col]).clip(0, 2)
    if ce_col in features.columns and e_col in features.columns:
        features["congestion_ratio_evening"] = _safe_divide(features[ce_col], features[e_col]).clip(0, 2)
    if cm_col in features.columns and ce_col in features.columns and m_col in features.columns and e_col in features.columns:
        features["overall_congestion_ratio"] = _safe_divide(features[cm_col] + features[ce_col], features[m_col] + features[e_col]).clip(0, 2)

    delay_col = delay_cols.get("threshold")
    wait_col = delay_cols.get("wait")
    if delay_col in features.columns and wait_col in features.columns:
        features["delay_minus_wait"] = pd.to_numeric(features[delay_col], errors="coerce") - pd.to_numeric(features[wait_col], errors="coerce")

    min_score = likert_cfg.get("min_score", 1)
    max_score = likert_cfg.get("max_score", 5)

    likert_norm = lambda col: _normalize_likert(features.get(col, pd.Series(dtype=float)), min_score, max_score)

    features["risk_aversion_index"] = (
        (1 - likert_norm("q14_19_aggressive_self"))
        + likert_norm("q14_20_cautious_self")
        + (1 - likert_norm("q8_2_reroute_frequency"))
    ) / 3

    features["comfort_preference_index"] = (
        likert_norm("q14_8_choose_smooth") + likert_norm("q12_3_weather")
    ) / 2

    if delay_col in features.columns:
        delay_norm = 1 - (pd.to_numeric(features[delay_col], errors="coerce") / 60).clip(0, 1)
    else:
        delay_norm = 0
    features["efficiency_orientation_index"] = (
        likert_norm("q14_9_immediate_detour") + delay_norm
    ) / 2

    info_usage = features.get("num_info_channels")
    if info_usage is None:
        info_usage = pd.Series(0, index=features.index)
    info_norm = (pd.to_numeric(info_usage, errors="coerce") / max(len(info_cols), 1)).clip(0, 1)
    features["information_dependency_index"] = (
        likert_norm("q11_1_seek_info") + info_norm
    ) / 2

    features["target_reroute_high"] = (pd.to_numeric(features.get("q8_2_reroute_frequency"), errors="coerce") >= 4).astype(int)
    features["target_low_delay_tolerance"] = (pd.to_numeric(features.get(wait_col), errors="coerce") <= 15).astype(int)

    for cat_col in params.get("categorical_one_hot", []):
        if cat_col in features.columns:
            dummies = _one_hot(features, cat_col)
            features = pd.concat([features, dummies], axis=1)

    engineered_cols = [
        "delta_commute_minutes",
        "avg_commute_minutes",
        "congestion_ratio_morning",
        "congestion_ratio_evening",
        "overall_congestion_ratio",
        "delay_minus_wait",
        "risk_aversion_index",
        "comfort_preference_index",
        "efficiency_orientation_index",
        "information_dependency_index",
        "target_reroute_high",
        "target_low_delay_tolerance",
    ]

    outputs = cfg.get("outputs", {})
    feature_path = resolve_output(_resolve_path(outputs.get("features", "data/processed/CN_features.parquet"), base_dir))
    metadata_path = resolve_output(_resolve_path(outputs.get("metadata", "results/features_meta.json"), base_dir))
    summary_path = resolve_output(_resolve_path(outputs.get("summary", "results/features_summary.json"), base_dir))

    features.to_parquet(feature_path, index=False)

    metadata = {
        "engineered_columns": engineered_cols,
        "categorical_one_hot": params.get("categorical_one_hot", []),
        "total_columns": list(features.columns),
    }
    save_json(metadata, metadata_path)

    summary = features[engineered_cols].describe(include="all").to_dict()
    save_json(summary, summary_path)

    return {
        "features": str(feature_path),
        "metadata": str(metadata_path),
        "summary": str(summary_path),
        "row_count": len(features),
    }
