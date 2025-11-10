from __future__ import annotations

from pathlib import Path

import click

from .eda.run_eda import run_eda
from .data.ingest_cn import run_ingest
from .data.feature_engineering import run_feature_engineering
from .data.dataset_builder import run_dataset_builder


@click.group()
def cli() -> None:
    """analysis_cn command line interface."""


@cli.command("run-eda")
@click.option("--config", "config_path", default="config/eda.yml", show_default=True)
def run_eda_cmd(config_path: str) -> None:
    """Execute the EDA pipeline and produce report/figures."""

    result = run_eda(config_path)
    click.echo("EDA 完成：")
    click.echo(f"  报告: {result['report']}")
    click.echo(f"  统计: {result['stats_json']}")
    click.echo(f"  图表数量: {len(result['figures'])}")


@cli.command("clean-data")
@click.option("--config", "config_path", default="config/ingest.yml", show_default=True)
def clean_data_cmd(config_path: str) -> None:
    """从原始 Excel 清洗生成结构化数据。"""

    result = run_ingest(config_path)
    click.echo("数据清洗完成：")
    click.echo(f"  CSV: {result['clean_csv']}")
    click.echo(f"  Parquet: {result['clean_parquet']}")
    click.echo(f"  质量报告: {result['quality_report']}")


@cli.command("build-features")
@click.option("--config", "config_path", default="config/features.yml", show_default=True)
def build_features_cmd(config_path: str) -> None:
    """生成特征与偏好指标。"""

    result = run_feature_engineering(config_path)
    click.echo("特征工程完成：")
    click.echo(f"  特征文件: {result['features']}")
    click.echo(f"  元数据: {result['metadata']}")


@cli.command("build-dataset")
@click.option("--config", "config_path", default="config/dataset.yml", show_default=True)
def build_dataset_cmd(config_path: str) -> None:
    """切分 train/val/test 数据集。"""

    result = run_dataset_builder(config_path)
    click.echo("数据集切分完成：")
    click.echo(f"  Train: {result['train']}")
    click.echo(f"  Val: {result['val']}")
    click.echo(f"  Test: {result['test']}")


if __name__ == "__main__":
    cli()
