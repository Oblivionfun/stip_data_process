from __future__ import annotations

from typing import Dict, List

import pandas as pd
from scipy import stats

from ...utils import stats as stats_utils


def reroute_label(df: pd.DataFrame, col: str, threshold: int = 4) -> pd.Series:
    series = pd.to_numeric(df.get(col), errors="coerce")
    return (series >= threshold).astype(int)


def info_channel_usage(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    usage = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return usage.sum(axis=1)


def reroute_analysis(df: pd.DataFrame, config: Dict) -> Dict:
    reroute_col = config.get("reroute_col", "Q8_2")
    info_cols = config.get("info_columns", [])
    result: Dict[str, Dict] = {}
    if reroute_col in df.columns:
        series = pd.to_numeric(df[reroute_col], errors="coerce")
        result["distribution"] = stats_utils.describe_series(series).to_dict()
        result["high_intent_ratio"] = float((series >= config.get("reroute_threshold", 4)).mean())
    if info_cols and all(col in df.columns for col in info_cols):
        usage = info_channel_usage(df, info_cols)
        df = df.copy()
        df["info_usage"] = usage
        df["high_info"] = (usage >= config.get("high_info_cut", 3)).astype(int)
        if reroute_col in df.columns:
            high = pd.to_numeric(df.loc[df["high_info"] == 1, reroute_col], errors="coerce")
            low = pd.to_numeric(df.loc[df["high_info"] == 0, reroute_col], errors="coerce")
            high = high.dropna()
            low = low.dropna()
            if len(high) > 0 and len(low) > 0:
                u_stat, p_value = stats.mannwhitneyu(high, low, alternative="two-sided")
                result["info_vs_reroute"] = {"u_stat": float(u_stat), "p_value": float(p_value)}
    return result
