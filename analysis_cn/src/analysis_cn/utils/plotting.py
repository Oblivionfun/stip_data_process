from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io import ensure_dir

DEFAULT_FONT = "SimHei"


def setup_style(dpi: int = 120) -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = [DEFAULT_FONT]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = dpi


def save_current_fig(path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_histograms(df: pd.DataFrame, cols: Iterable[str], title: str, out_path: Path) -> None:
    n_cols = len(list(cols))
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(pd.to_numeric(df[col], errors="coerce"), kde=True, ax=ax, color="#4C78A8")
        ax.set_title(col)
    fig.suptitle(title)
    save_current_fig(out_path)


def plot_boxpairs(df: pd.DataFrame, pairs: List[tuple[str, str]], title: str, out_path: Path) -> None:
    n = len(pairs)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (col1, col2) in zip(axes, pairs):
        sns.boxplot(data=pd.melt(df[[col1, col2]].apply(pd.to_numeric, errors="coerce"), var_name="period", value_name="value"),
                    x="period", y="value", ax=ax)
        ax.set_title(f"{col1} vs {col2}")
    fig.suptitle(title)
    save_current_fig(out_path)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str, out_path: Path) -> None:
    subset = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = subset.corr(method="pearson", min_periods=2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    save_current_fig(out_path)
