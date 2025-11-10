from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jinja2 import Template

from ..utils.io import load_dataframe, resolve_output, save_json
from ..utils import plotting
from .sections import (
    commute_summary,
    congestion_summary,
    threshold_summary,
    plot_commute_figures,
    reroute_analysis,
    likert_blocks_summary,
    plot_likert_blocks,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_template.html"


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


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def render_report(context: Dict[str, Any], report_path: Path) -> None:
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = Template(template_text)
    html = template.render(**context)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")


def run_eda(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    base_dir = _infer_base_dir(config_path)
    config = load_config(config_path)
    data_path = _resolve_path(config["input_path"], base_dir)
    df = load_dataframe(data_path)

    outputs = config.get("outputs", {})
    report_path = resolve_output(_resolve_path(outputs.get("report", "reports/01_EDA_report.html"), base_dir))
    stats_json_path = resolve_output(_resolve_path(outputs.get("stats_json", "results/eda_statistics.json"), base_dir))
    figures_dir = _resolve_path(outputs.get("figures_dir", "figures/eda"), base_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_bundle: Dict[str, Any] = {}
    figures: List[str] = []

    commute_cols = config.get("variables", {}).get("commute", [])
    stats_bundle["commute_stats"] = commute_summary(df, commute_cols)
    stats_bundle["congestion_stats"] = congestion_summary(df, config.get("variables", {}).get("congestion", []))
    threshold_cfg = config.get("variables", {}).get("thresholds", {})
    stats_bundle["threshold_stats"] = threshold_summary(
        df,
        threshold_cfg.get("delay"),
        threshold_cfg.get("wait"),
    )
    figures.extend(plot_commute_figures(df, config.get("variables", {}), Path(figures_dir)))

    reroute_cfg = config.get("reroute", {
        "reroute_col": "Q8_2",
        "reroute_threshold": 4,
        "info_columns": ["Q9_1", "Q9_2", "Q9_3"],
        "high_info_cut": 3,
    })
    stats_bundle["reroute_stats"] = reroute_analysis(df, reroute_cfg)

    likert_blocks = config.get("variables", {}).get("likert_blocks", [])
    stats_bundle["likert_stats"] = likert_blocks_summary(df, likert_blocks)
    figures.extend(plot_likert_blocks(df, likert_blocks, Path(figures_dir)))

    corr_cols = config.get("correlations", {}).get("include", [])
    corr_cols = [col for col in corr_cols if col in df.columns]
    if corr_cols:
        plotting.setup_style()
        corr_fig = Path(figures_dir) / "correlation_heatmap.png"
        plotting.plot_correlation_heatmap(df, corr_cols, "Pearson 相关", corr_fig)
        figures.append(str(corr_fig))

    save_json(stats_bundle, stats_json_path)

    context = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "commute_stats_json": json.dumps(stats_bundle.get("commute_stats", {}), ensure_ascii=False, indent=2),
        "threshold_stats_json": json.dumps(stats_bundle.get("threshold_stats", {}), ensure_ascii=False, indent=2),
        "reroute_stats_json": json.dumps(stats_bundle.get("reroute_stats", {}), ensure_ascii=False, indent=2),
        "likert_stats_json": json.dumps(stats_bundle.get("likert_stats", {}), ensure_ascii=False, indent=2),
        "figures": figures,
    }
    render_report(context, report_path)

    return {
        "report": str(report_path),
        "stats_json": str(stats_json_path),
        "figures": figures,
    }
