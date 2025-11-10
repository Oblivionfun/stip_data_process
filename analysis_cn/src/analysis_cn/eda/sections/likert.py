from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ...utils import plotting, stats


def likert_blocks_summary(df: pd.DataFrame, blocks: List[List[str]]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for idx, block in enumerate(blocks, start=1):
        valid_cols = [col for col in block if col in df.columns]
        if not valid_cols:
            continue
        block_df = df[valid_cols]
        block_stats = stats.describe_likert_block(block_df)
        summary[f"block_{idx}"] = block_stats
    return summary


def plot_likert_blocks(df: pd.DataFrame, blocks: List[List[str]], figures_dir: Path) -> List[str]:
    plotting.setup_style()
    created: List[str] = []
    for idx, block in enumerate(blocks, start=1):
        valid_cols = [col for col in block if col in df.columns]
        if not valid_cols:
            continue
        subset = df[valid_cols].apply(pd.to_numeric, errors="coerce")
        melted = subset.melt(var_name="question", value_name="score").dropna()
        if melted.empty:
            continue
        plt.figure(figsize=(max(6, len(valid_cols) * 1.5), 4))
        sns.violinplot(data=melted, x="question", y="score", inner="quartile", cut=0)
        plt.ylim(1, 5)
        plt.title(f"Likert Block {idx}")
        out_path = figures_dir / f"likert_block_{idx}.png"
        plotting.save_current_fig(out_path)
        created.append(str(out_path))
    return created
