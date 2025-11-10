from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatSummary:
    count: int
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


def describe_series(series: pd.Series) -> StatSummary:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return StatSummary(0, None, None, None, None, None)
    return StatSummary(
        count=int(clean.count()),
        mean=float(clean.mean()),
        median=float(clean.median()),
        std=float(clean.std()),
        min=float(clean.min()),
        max=float(clean.max()),
    )


def paired_t_test(a: pd.Series, b: pd.Series) -> Dict[str, float | None]:
    a_clean = pd.to_numeric(a, errors="coerce")
    b_clean = pd.to_numeric(b, errors="coerce")
    mask = a_clean.notna() & b_clean.notna()
    if mask.sum() < 2:
        return {"t_stat": None, "p_value": None}
    t_stat, p_value = stats.ttest_rel(a_clean[mask], b_clean[mask])
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def correlation(a: pd.Series, b: pd.Series, method: str = "pearson") -> Dict[str, float | None]:
    a_clean = pd.to_numeric(a, errors="coerce")
    b_clean = pd.to_numeric(b, errors="coerce")
    mask = a_clean.notna() & b_clean.notna()
    if mask.sum() < 2:
        return {"corr": None, "p_value": None}
    if method == "pearson":
        corr, p_value = stats.pearsonr(a_clean[mask], b_clean[mask])
    else:
        corr, p_value = stats.spearmanr(a_clean[mask], b_clean[mask])
    return {"corr": float(corr), "p_value": float(p_value)}


def cronbach_alpha(df: pd.DataFrame) -> float | None:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna()
    n_items = numeric.shape[1]
    if n_items < 2:
        return None
    variances = numeric.var(axis=0, ddof=1)
    total_var = numeric.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return None
    alpha = (n_items / (n_items - 1)) * (1 - variances.sum() / total_var)
    return float(alpha)


def describe_likert_block(df: pd.DataFrame) -> Dict[str, Dict[str, float | int | None]]:
    block_stats: Dict[str, Dict[str, float | int | None]] = {}
    for col in df.columns:
        stat = describe_series(df[col]).to_dict()
        block_stats[col] = stat
    alpha = cronbach_alpha(df)
    block_stats["cronbach_alpha"] = alpha
    return block_stats
