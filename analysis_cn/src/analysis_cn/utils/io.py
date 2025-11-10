from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load dataframe from parquet/csv/xlsx. Raises FileNotFoundError if missing."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {suffix}")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_output(path: str | Path) -> Path:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    return file_path
