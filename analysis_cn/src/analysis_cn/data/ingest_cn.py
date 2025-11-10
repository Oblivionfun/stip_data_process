from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from ..utils.io import resolve_output, save_json


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_rename_map(columns: List[str], mapping_cfg: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    used: Dict[str, int] = {}
    exact = mapping_cfg.get("exact", {})
    contains = mapping_cfg.get("contains", {})

    for col in columns:
        target = None
        if col in exact:
            target = exact[col]
        else:
            for pattern, alias in contains.items():
                if pattern in col:
                    target = alias
                    break
        if target is None:
            slug = re.sub(r"\W+", "_", col).strip("_")
            target = slug.lower() or "col"

        if target in used:
            used[target] += 1
            unique_name = f"{target}_{used[target]}"
        else:
            used[target] = 0
            unique_name = target
        rename_map[col] = unique_name
    return rename_map


def parse_duration(series: pd.Series) -> pd.Series:
    def _to_seconds(value):
        if pd.isna(value):
            return pd.NA
        text = str(value)
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return pd.NA
        return float(match.group(1))

    return series.apply(_to_seconds)


def convert_columns_to_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_quality_flags(df: pd.DataFrame, info_cols: List[str]) -> pd.DataFrame:
    info_cols = [c for c in info_cols if c in df.columns]
    if info_cols:
        df["num_info_channels"] = df[info_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        df["info_channel_binary"] = (df[info_cols] > 0).any(axis=1).astype(int)
    else:
        df["num_info_channels"] = 0
        df["info_channel_binary"] = 0
    return df


def clean_dataframe(df: pd.DataFrame, config: Dict, base_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rename_cfg = load_yaml(_resolve_path(config["rename_map"], base_dir))
    rename_map = build_rename_map(list(df.columns), rename_cfg)
    df = df.rename(columns=rename_map)

    counts = {"raw_rows": len(df)}

    df["duration_seconds"] = parse_duration(df.get("duration_raw"))
    df["submitted_at"] = pd.to_datetime(df.get("submitted_at"), errors="coerce")

    if dup_col := config.get("filters", {}).get("drop_duplicates_on"):
        if dup_col in df.columns:
            df = df.sort_values("submitted_at").drop_duplicates(subset=dup_col, keep="last")
    counts["after_dedup"] = len(df)

    convert_columns_to_numeric(df, config.get("numeric_columns", []))
    add_quality_flags(df, config.get("quality_flags", {}).get("info_channel_cols", []))

    max_duration = config.get("filters", {}).get("max_duration_seconds")
    if max_duration is not None and "duration_seconds" in df.columns:
        df["duration_flag_long"] = df["duration_seconds"] > max_duration
        df = df.loc[~df["duration_flag_long"].fillna(False)]
    counts["after_duration_filter"] = len(df)

    drop_cols = config.get("filters", {}).get("drop_columns", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    critical_cols = ["q3_morning_commute_minutes", "q4_evening_commute_minutes"]
    df["missing_commute_flag"] = df[critical_cols].isna().any(axis=1)
    df = df.loc[~df["missing_commute_flag"]]
    counts["after_commute_filter"] = len(df)

    df = df.drop(columns=["duration_flag_long", "missing_commute_flag"], errors="ignore")
    df.reset_index(drop=True, inplace=True)
    return df, counts


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _infer_base_dir(config_path: Path) -> Path:
    parent = config_path.parent
    if (parent / "data").exists():
        return parent
    if parent.name == "config" and parent.parent.exists():
        return parent.parent
    return parent


def run_ingest(config_path: str | Path) -> Dict[str, str]:
    config_path = Path(config_path).resolve()
    base_dir = _infer_base_dir(config_path)
    config = load_yaml(config_path)
    raw_path = _resolve_path(config["raw_path"], base_dir)
    df = pd.read_excel(raw_path, engine="openpyxl")

    clean_df, stats = clean_dataframe(df, config, base_dir)

    outputs = config.get("outputs", {})
    csv_path = resolve_output(_resolve_path(outputs.get("clean_csv", "data/processed/CN_cleaned.csv"), base_dir))
    parquet_path = resolve_output(_resolve_path(outputs.get("clean_parquet", "data/processed/CN_cleaned.parquet"), base_dir))
    quality_path = resolve_output(_resolve_path(outputs.get("quality_report", "results/quality_report.json"), base_dir))

    clean_df.to_csv(csv_path, index=False)
    clean_df.to_parquet(parquet_path, index=False)

    stats["clean_rows"] = len(clean_df)
    stats["columns"] = list(clean_df.columns)
    save_json(stats, quality_path)

    return {
        "clean_csv": str(csv_path),
        "clean_parquet": str(parquet_path),
        "quality_report": str(quality_path),
        "row_count": len(clean_df),
    }
