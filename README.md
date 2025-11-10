# 一个中国数据分析的Baseline

面向中国驾驶员的路径决策分析。流水线自原始 `CN_dataset_FULL.xlsx` 出发，依次完成：

- **数据清洗（clean-data）**：`ingest_cn.py` 解析 Excel，按 `rename_map` 自动重命名中文列、统一字符串时长、数值化 Likert/多选题，过滤掉重复/超长作答/关键字段缺失的数据，同时生成信息渠道使用等质量标记，输出 `data/processed/CN_cleaned.csv|parquet` 及 `results/quality_report.json`。
- **特征工程（build-features）**：`feature_engineering.py` 基于清洗数据计算通勤差值、拥堵占比、延误-等待差异、偏好指数（风险规避/舒适偏好/效率导向/信息依赖），派生目标标签与类别哑变量，形成 `data/processed/CN_features.parquet`、`results/features_meta.json`、`results/features_summary.json`。
- **数据集构建（build-dataset）**：`dataset_builder.py` 读取特征文件，根据 `config/dataset.yml` 的比例与 stratify 列将数据切分为 train/val/test（CSV），并记录 `results/dataset_stats.json`。
- **探索分析（run-eda）**：`eda/run_eda.py` 依 `config/eda.yml` 汇总通勤/拥堵统计、改道行为、Likert 题块及相关性矩阵，输出 `reports/01_EDA_report.html`、`results/eda_statistics.json` 和 `figures/eda/*`。

## 快速开始

1. （可选）在 Python 3.8 环境下创建虚拟环境：
   ```bash
   python3.8 -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据：将原始 `CN_dataset_FULL.xlsx` 放在仓库根目录（默认配置即读取 `../CN_dataset_FULL.xlsx`）。
4. 执行流水线：
   ```bash
   cd analysis_cn
   PYTHONPATH=src python -m analysis_cn.cli clean-data     --config config/ingest.yml
   PYTHONPATH=src python -m analysis_cn.cli build-features --config config/features.yml
   PYTHONPATH=src python -m analysis_cn.cli build-dataset  --config config/dataset.yml
   PYTHONPATH=src python -m analysis_cn.cli run-eda        --config config/eda.yml
   ```

## 目录结构
```
analysis_cn/
├── config/
│   ├── ingest.yml       # 原始路径、列映射、过滤策略
│   ├── features.yml     # 特征工程参数（通勤列、信息列、哑变量等）
│   ├── dataset.yml      # 切分比例、stratify 目标
│   └── eda.yml          # EDA 输入/输出/变量集合
├── data/
│   ├── metadata/
│   │   └── columns_map_cn.yml  # 中文列 → 英文列映射（支持 exact/contains 模式）
│   └── processed/       # 清洗结果、特征、train/val/test
├── figures/             # EDA 图表（自动生成）
├── reports/             # EDA HTML 报告
├── results/             # 质量报告、特征/数据集统计
├── src/analysis_cn/
│   ├── cli.py           # Click CLI，挂载 clean-data/build-features/build-dataset/run-eda
│   ├── data/
│   │   ├── ingest_cn.py           # Excel → 清洗
│   │   ├── feature_engineering.py # 清洗 → 特征
│   │   └── dataset_builder.py     # 特征 → train/val/test
│   ├── eda/
│   │   ├── run_eda.py             # 生成统计和HTML
│   │   ├── sections/              # commute/reroute/likert 分块分析
│   │   └── templates/report_template.html
│   └── utils/                     # IO/统计/绘图工具
└── requirements.txt
```

## CLI 命令与脚本逻辑
- `clean-data` → `analysis_cn/src/analysis_cn/data/ingest_cn.py`
  - 加载 `config/ingest.yml`，解析 `rename_map`（自动去重列名），统一“所用时间”等字符串字段为秒单位。
  - `numeric_columns` 列表一键 `pd.to_numeric`，多选题（Q9.*）按二元列保留，并计算 `num_info_channels` 等质量指标。
  - 过滤逻辑：按 respondent_id 去重、剔除作答时间>1800秒、去掉控制/重复列、删除缺失通勤时长的样本。
  - 输出 `CN_cleaned.csv/parquet` + `quality_report.json`（包含各阶段样本量、最终列名等）。

- `build-features` → `analysis_cn/src/analysis_cn/data/feature_engineering.py`
  - 依据 `config/features.yml` 中的通勤/拥堵列，计算 `delta_commute_minutes`、`avg_commute_minutes`、`congestion_ratio_*` 等派生数值。
  - 结合 Q14/Q8 等 Likert 题生成 `risk_aversion_index`、`comfort_preference_index`、`efficiency_orientation_index`、`information_dependency_index`，并构造二分类目标 `target_reroute_high`、`target_low_delay_tolerance`。
  - 支持对 `categorical_one_hot` 指定列自动做 one-hot（含缺失类别）。
  - 输出 `CN_features.parquet`（全量特征）、`features_meta.json`（记录工程特征、哑变量列）、`features_summary.json`（工程特征描述统计）。

- `build-dataset` → `analysis_cn/src/analysis_cn/data/dataset_builder.py`
  - 读取 `CN_features.parquet`，按 `config/dataset.yml` 的比例/随机种子切分 train/val/test，可选 stratify 指定标签。
  - 写出 `train.csv`、`val.csv`、`test.csv` 及 `dataset_stats.json`（记录三份行数、目标列信息）。

- `run-eda` → `analysis_cn/src/analysis_cn/eda/run_eda.py`
  - 使用 `config/eda.yml` 中配置的列名（已英文化），调用 `sections/commute.py`、`sections/reroute.py`、`sections/likert.py` 完成分块统计。
  - 自动绘制通勤/拥堵直方图、箱线图、Likert 小提琴图、相关性热力图，写入 `figures/eda/`。
  - 通过 Jinja 模板输出 `reports/01_EDA_report.html`，内容包含各统计量 JSON 片段和图表预览。

## 配置说明
`config/eda.yml` 控制 EDA 输入/输出；`config/ingest.yml` 定义原始路径、列映射、过滤参数；`config/features.yml` 设定特征参数、类别哑变量；`config/dataset.yml` 设定切分比例与目标列。`config/eda.yml` 示例：
```yaml
input_path: data/processed/CN_cleaned.parquet
outputs:
  report: reports/01_EDA_report.html
  stats_json: results/eda_statistics.json
  figures_dir: figures/eda
variables:
  commute: [q3_morning_commute_minutes, q4_evening_commute_minutes]
  congestion: [q6_morning_congestion_minutes, q7_evening_congestion_minutes]
  thresholds:
    delay: q13_delay_threshold_minutes
    wait: q15_wait_before_detour_minutes
  likert_blocks:
    - [q8_1_congestion_problem, q8_2_reroute_frequency, q8_3_confidence, q8_4_return_normal]
```

## 注意事项
- requirements 中 `pandas<2.1`、`matplotlib<3.8`、`scipy<1.11`、`numpy<1.25` 均为兼容 Python 3.8 的折衷，如升级 Python 版本可按需放宽。
- 所有输出目录在运行脚本时会自动创建。
- 若某个字段不存在，脚本会在日志中给出警告并跳过。
- 运行前请确保默认字体支持中文（示例中使用 `SimHei`）。
