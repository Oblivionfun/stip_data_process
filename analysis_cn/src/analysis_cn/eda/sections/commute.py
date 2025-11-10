from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ...utils import plotting, stats


def commute_summary(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        summary[col] = stats.describe_series(df[col]).to_dict()
    if len(columns) >= 2 and all(col in df.columns for col in columns[:2]):
        summary["paired_t_test"] = stats.paired_t_test(df[columns[0]], df[columns[1]])
    return summary


def congestion_summary(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
    output: Dict[str, Dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        output[col] = stats.describe_series(df[col]).to_dict()
    if len(columns) >= 2 and all(col in df.columns for col in columns[:2]):
        output["correlation"] = stats.correlation(df[columns[0]], df[columns[1]])
    return output


def threshold_summary(df: pd.DataFrame, delay_col: str | None, wait_col: str | None) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    if delay_col and delay_col in df.columns:
        result[delay_col] = stats.describe_series(df[delay_col]).to_dict()
    if wait_col and wait_col in df.columns:
        result[wait_col] = stats.describe_series(df[wait_col]).to_dict()
    if delay_col and wait_col and delay_col in df.columns and wait_col in df.columns:
        result["correlation"] = stats.correlation(df[delay_col], df[wait_col])
    return result


def plot_commute_figures(df: pd.DataFrame, config: Dict, figures_dir: Path) -> List[str]:
    plotting.setup_style()
    created: List[str] = []
    commute_cols = [c for c in config.get("commute", []) if c in df.columns]
    if commute_cols:
        out = figures_dir / "commute_distribution.png"
        plotting.plot_histograms(df, commute_cols, "通勤时长分布", out)
        created.append(str(out))
    congestion_cols = [c for c in config.get("congestion", []) if c in df.columns]
    if congestion_cols:
        out = figures_dir / "congestion_distribution.png"
        plotting.plot_histograms(df, congestion_cols, "拥堵时长分布", out)
        created.append(str(out))
    if len(commute_cols) >= 2:
        out = figures_dir / "commute_boxpair.png"
        plotting.plot_boxpairs(df, [(commute_cols[0], commute_cols[1])], "早晚通勤对比", out)
        created.append(str(out))
    return created
