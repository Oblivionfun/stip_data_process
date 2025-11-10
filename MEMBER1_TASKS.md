# 成员1 - 数据工程师任务清单（改进版）

## 核心职责
数据清洗、特征工程、结构化数据集构建

---

## 任务1.1：数据摄取与初步清洗 (第1周-前3天)

### 输入
- `CN_dataset_FULL.xlsx` (1042×76)

### 输出
- `data/raw/CN_raw.csv` - 原始数据备份
- `data/processed/CN_cleaned.csv` - 清洗后数据
- `docs/dropped_columns.md` - 被删除列的记录
- `logs/data_quality_report.txt` - 质量报告

### 具体任务

#### 1.1.1 编写 `scripts/ingest_cn.py`
```python
功能：
1. 读取Excel，解析"所用时间"字段（"326秒" → 326）
2. 过滤作答时间>1800秒的18条样本（保守策略）
3. 识别常量列（如"来源详情"全NaN）并记录
4. 标准化列名：中文题干 → 英文字段名
   - "3、您早晨通勤时..." → "Q3_morning_commute_time"
   - "8.2、我在通勤时经常..." → "Q8_2_reroute_frequency"
5. 保存原始+清洗版本
```

#### 1.1.2 字段映射规则
- **Q1-Q2**: `Q1_statement`, `Q2_statement` (需查看具体内容)
- **Q3-Q4**: `Q3_morning_time`, `Q4_afternoon_time` (数值，分钟)
- **Q5**: `Q5_road_type` (类别)
- **Q6-Q7**: `Q6_morning_congestion`, `Q7_afternoon_congestion` (数值，分钟)
- **Q8.1-Q8.4**: `Q8_1_congestion_problem` ~ `Q8_4_return_normal_route` (Likert 1-5)
- **Q9**: `Q9_navi_invehicle`, `Q9_navi_mobile`, ..., `Q9_none` (二进制 0/1)
- **Q10**: `Q10_info_seeking_timing` (类别)
- **Q11.1-Q11.4**: `Q11_1_freq_seek_info` ~ `Q11_4_cancel_trip` (Likert)
- **Q12.1-Q12.5**: `Q12_1_construction`, ..., `Q12_5_accident` (Likert)
- **Q13**: `Q13_delay_threshold` (数值，分钟)
- **Q14.1-Q14.20**: `Q14_1_prefer_familiar` ~ `Q14_20_cautious_driver` (Likert)
- **Q14.10**: `Q14_10_attention_check` (控制题，正确答案=2)
- **Q15**: `Q15_congestion_wait_time` (数值，分钟)
- **Q16.1-Q16.6**: `Q16_1_commute_morning` ~ `Q16_6_emergency_evac` (Likert)
- **Q17-Q26**: `gender`, `age`, `ethnicity`, `income`, `occupation`, `education`, `vehicle_type`, `has_navigation`, `home_location`, `work_location`
- **总分**: `total_score` (保留作为质量指标)

#### 1.1.3 质量标记
在清洗后的数据中添加列：
- `quality_response_time`: 1=合格(120-1800s), 0=超时
- `quality_attention_check`: 1=通过(Q14.10==2), 0=失败
- `quality_straightlining`: 1=正常(Likert_std>0.5), 0=疑似
- `quality_info_channel_valid`: 1=至少选一个渠道, 0=全零异常
- `overall_quality_flag`: 综合标记 (1=高质量, 0=需审查)

**执行检查点**：
- [ ] 生成 `CN_cleaned.csv` (预计1024行×80+列)
- [ ] 验证控制题Q14.10全部=2
- [ ] 确认无重复respondent_id

---

## 任务1.2：构建数据字典 (第1周-第4天)

### 输出
- `data/metadata/codebook_cn.xlsx`

### 结构
| 字段名 | 原始题干 | 类型 | 编码 | 备注 |
|--------|---------|------|------|------|
| Q3_morning_time | 您早晨通勤时... | 数值 | 分钟 | 连续变量 |
| Q8_2_reroute_frequency | 我在通勤时经常... | Likert | 1=非常不同意, 5=非常同意 | 李克特 |
| Q9_navi_mobile | 手机上的导航系统 | 二进制 | 0=未选, 1=已选 | 多选题 |
| gender | 您的性别是？ | 类别 | 1=男, 2=女, 4=其他 | 需合并4→3 |
| ... | ... | ... | ... | ... |

**特殊处理**：
- Q9.* 所有子列需one-hot编码
- Q14.10 仅用于质量控制，建模时删除
- Q1/Q2 若为同意条款，可删除

---

## 任务1.3：特征工程 (第2周)

### 输出
- `scripts/feature_engineering.py`
- `data/features/preference_embeddings_cn.parquet`
- `docs/feature_definitions.json`

### 派生特征清单

#### 1. 时间相关特征
```python
# 通勤时间差异
delta_commute_time = Q4_afternoon_time - Q3_morning_time

# 拥堵占比
congestion_ratio_morning = Q6_morning_congestion / Q3_morning_time
congestion_ratio_afternoon = Q7_afternoon_congestion / Q4_afternoon_time

# 总拥堵时长
total_congestion_time = Q6_morning_congestion + Q7_afternoon_congestion

# 延误容忍度归一化
delay_tolerance_norm = Q13_delay_threshold / Q3_morning_time
```

#### 2. 复合偏好指标（核心）
```python
# 风险规避指数 (0-1, 越高越谨慎)
risk_aversion_index = (
    0.4 * (Q14_20_cautious_driver / 5) +
    0.3 * (1 - Q14_19_aggressive_driver / 5) +
    0.3 * (1 - Q8_2_reroute_frequency / 5)
)

# 舒适性偏好 (0-1, 越高越重视舒适)
comfort_preference_score = (
    0.5 * (Q14_8_smooth_over_fast / 5) +
    0.3 * (Q12_3_weather_sensitivity / 5) +
    0.2 * (Q14_4_time_reliability / 5)
)

# 效率追求度 (0-1, 越高越重视时间)
efficiency_orientation = (
    0.5 * (Q14_9_immediate_reroute / 5) +
    0.3 * (1 - Q13_delay_threshold / 60) +  # 阈值越低=越追求效率
    0.2 * (1 - Q14_8_smooth_over_fast / 5)
)

# 信息依赖度 (0-1, 越高越依赖导航)
information_dependency = (
    0.4 * (Q11_1_freq_seek_info / 5) +
    0.3 * (1 - Q11_3_ignore_guidance / 5) +
    0.2 * (Q14_16_vms_willingness / 5) +
    0.1 * (sum(Q9_*) / 10)  # 使用的信息渠道数量
)

# 社会从众性 (0-1)
social_conformity = Q14_17_follow_others / 5
```

#### 3. 行为特征
```python
# 信息渠道使用数量
num_info_channels = sum([Q9_navi_invehicle, Q9_navi_mobile, ...])

# 主要信息源
primary_info_source = argmax([Q9_navi_invehicle, Q9_navi_mobile, ...])

# 情景敏感度得分（对5种情景的平均响应）
scenario_sensitivity = mean([Q12_1, Q12_2, Q12_3, Q12_4, Q12_5])

# 路线熟悉度偏好
familiarity_preference = Q14_1_prefer_familiar / 5
```

#### 4. 人口统计学交叉特征
```python
# 年龄分组
age_group = cut(age, bins=[20, 30, 35, 40, 70], labels=['young', 'mid_young', 'mid', 'senior'])

# 高教育×高收入
high_edu_income = (education >= 5) & (income >= 5)

# 年龄×风险偏好
age_risk_interaction = age * risk_aversion_index
```

### 特征元数据 `feature_definitions.json`
```json
{
  "risk_aversion_index": {
    "formula": "0.4*Q14.20 + 0.3*(1-Q14.19) + 0.3*(1-Q8.2)",
    "range": [0, 1],
    "description": "综合风险规避倾向，越高越谨慎",
    "components": ["Q14_20_cautious", "Q14_19_aggressive", "Q8_2_reroute_freq"]
  },
  "comfort_preference_score": {...},
  ...
}
```

---

## 任务1.4：结构化数据集构建 (第3周)

### 输出文件结构
```
data/
├── raw/
│   └── CN_raw.csv
├── processed/
│   └── CN_cleaned.csv
├── features/
│   ├── preference_embeddings_cn.parquet  # 包含所有派生特征
│   └── train_val_test_split.json        # 记录划分索引
├── ml_ready/
│   ├── train.csv                         # 70% (717样本)
│   ├── val.csv                           # 15% (154样本)
│   ├── test.csv                          # 15% (153样本)
│   └── scenario_sequences.json           # 用于模仿学习的序列数据
└── metadata/
    ├── codebook_cn.xlsx
    ├── feature_definitions.json
    └── data_schema.json
```

### `scenario_sequences.json` 格式（关键）
```json
[
  {
    "respondent_id": "R001",
    "demographics": {
      "age": 32,
      "gender": "male",
      "education": 5,
      "home_location": 4,
      "work_location": 3
    },
    "preference_profile": {
      "risk_aversion": 0.67,
      "comfort_preference": 0.52,
      "efficiency_orientation": 0.78,
      "information_dependency": 0.45
    },
    "scenarios": [
      {
        "scenario_type": "construction",
        "scenario_id": "Q12_1",
        "reroute_willingness": 4,
        "reroute_binary": 1
      },
      {
        "scenario_type": "accident",
        "scenario_id": "Q12_5",
        "reroute_willingness": 5,
        "reroute_binary": 1
      },
      ...
    ],
    "behavioral_constraints": {
      "delay_threshold": 15,
      "congestion_wait_threshold": 18,
      "prefers_familiar_routes": true,
      "primary_info_source": "mobile_navigation"
    }
  },
  ...
]
```

### PyTorch DataLoader 示例
```python
# scripts/data_loader.py
import torch
from torch.utils.data import Dataset

class RoutePreferenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 偏好向量 (4维)
        preference = torch.tensor([
            sample['preference_profile']['risk_aversion'],
            sample['preference_profile']['comfort_preference'],
            sample['preference_profile']['efficiency_orientation'],
            sample['preference_profile']['information_dependency']
        ])

        # 情景-决策序列 (5个情景)
        scenarios = torch.tensor([
            s['reroute_binary'] for s in sample['scenarios']
        ])

        # 人口统计学嵌入
        demographics = torch.tensor([
            sample['demographics']['age'] / 100,  # 归一化
            sample['demographics']['gender'] - 1,  # 0-based
            sample['demographics']['education'] / 6
        ])

        return {
            'preference': preference,
            'demographics': demographics,
            'scenarios': scenarios
        }
```

---

## 任务检查清单

### 第1周
- [ ] `ingest_cn.py` 完成并测试
- [ ] `CN_cleaned.csv` 生成 (1024×80+)
- [ ] `codebook_cn.xlsx` 完成
- [ ] `dropped_columns.md` 记录完整
- [ ] 质量报告显示：控制题100%通过，超时样本已过滤

### 第2周
- [ ] `feature_engineering.py` 完成
- [ ] 四大偏好指数计算正确（0-1范围）
- [ ] `preference_embeddings_cn.parquet` 包含100+特征
- [ ] `feature_definitions.json` 完整记录公式

### 第3周
- [ ] train/val/test 划分完成（70/15/15）
- [ ] `scenario_sequences.json` 生成（1024个样本）
- [ ] PyTorch DataLoader 测试通过
- [ ] 数据集文档 `DATA_README.md` 完成

---

## 与其他成员的接口

### 交付给成员2（数据分析师）
- `CN_cleaned.csv` - 用于统计分析
- `preference_embeddings_cn.parquet` - 用于聚类
- `codebook_cn.xlsx` - 理解字段含义

### 交付给成员3（领域研究员）
- `scenario_sequences.json` - 用于情景-决策分析
- `feature_definitions.json` - 理解偏好映射

### 交付给成员4（可视化工程师）
- 所有上述数据 + `data_schema.json` - 用于仪表盘开发

---

## 代码规范

### 命名约定
- 文件：小写+下划线 `ingest_cn.py`
- 函数：小写+下划线 `load_raw_data()`
- 类：大驼峰 `PreferenceEncoder`
- 常量：大写+下划线 `LIKERT_COLUMNS`

### 文档字符串
```python
def calculate_risk_aversion(row: pd.Series) -> float:
    """
    计算风险规避指数

    Args:
        row: 包含Q14.19, Q14.20, Q8.2的Series

    Returns:
        float: 风险规避指数 (0-1)

    Formula:
        0.4*Q14.20 + 0.3*(1-Q14.19) + 0.3*(1-Q8.2)
    """
    ...
```

### 日志记录
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Loaded {len(df)} samples from raw data")
logger.warning(f"Removed {n_outliers} outliers")
```

---

## 参考资源
- Pandas文档：https://pandas.pydata.org/
- Parquet格式：https://arrow.apache.org/docs/python/parquet.html
- PyTorch DataLoader：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
