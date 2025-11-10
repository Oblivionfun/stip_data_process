from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ..utils.io import resolve_output, save_json


def _resolve_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _infer_base_dir(config_path: Path) -> Path:
    parent = config_path.parent
    if (parent / "data").exists():
        return parent
    if parent.name == "config" and parent.parent.exists():
        return parent.parent
    return parent


def run_dataset_builder(config_path: str | Path) -> Dict[str, str]:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    base_dir = _infer_base_dir(config_path)

    features_path = _resolve_path(cfg["features_path"], base_dir)
    df = pd.read_parquet(features_path)

    split_cfg = cfg.get("split", {})
    train_size = split_cfg.get("train", 0.7)
    val_size = split_cfg.get("val", 0.15)
    test_size = split_cfg.get("test", 0.15)
    random_state = split_cfg.get("random_state", 42)
    stratify_col = split_cfg.get("stratify_on")

    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train/val/test split ratios must sum to 1")

    stratify_series = df[stratify_col] if stratify_col in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify_series,
    )

    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_df[stratify_col] if stratify_col in temp_df.columns else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio,
        random_state=random_state,
        stratify=stratify_temp,
    )

    outputs = cfg.get("outputs", {})
    train_path = resolve_output(_resolve_path(outputs.get("train", "data/processed/train.csv"), base_dir))
    val_path = resolve_output(_resolve_path(outputs.get("val", "data/processed/val.csv"), base_dir))
    test_path = resolve_output(_resolve_path(outputs.get("test", "data/processed/test.csv"), base_dir))
    stats_path = resolve_output(_resolve_path(outputs.get("stats", "results/dataset_stats.json"), base_dir))

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    stats = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "targets": cfg.get("targets", {}),
    }
    save_json(stats, stats_path)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "stats": str(stats_path),
    }
