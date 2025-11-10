# 成员2 - 数据分析师任务清单（改进版）

## 核心职责
统计分析、聚类建模、决策规则提取

---

## 任务2.1：探索性数据分析 (EDA) (第1周)

### 输入
- `data/processed/CN_cleaned.csv` (来自成员1)
- `data/metadata/codebook_cn.xlsx`

### 输出
- `reports/01_EDA_report.html` - 完整EDA报告
- `results/eda_statistics.json` - 关键统计量
- `figures/eda/` - 所有EDA图表

---

### 2.1.1 通勤行为描述统计

**分析维度**：
```python
# 1. 通勤时间分布
- Q3_morning_time: 均值、中位数、四分位数、偏度
- Q4_afternoon_time: 同上
- delta_commute_time: 早晚差异（配对t检验）

# 2. 拥堵时长分布
- Q6_morning_congestion: 分布+箱线图
- Q7_afternoon_congestion: 分布+箱线图
- congestion_ratio: 拥堵占比分布（关键指标）

# 3. 延误与等待阈值
- Q13_delay_threshold: 预期延误容忍度分布
- Q15_congestion_wait_time: 实际等待容忍度分布
- 两者相关性（Pearson/Spearman）
```

**统计检验**：
1. **早晚通勤时间差异** (配对t检验)
   ```python
   from scipy.stats import ttest_rel
   t_stat, p_value = ttest_rel(df['Q3_morning_time'], df['Q4_afternoon_time'])
   # H0: 早晚通勤时间无差异
   ```

2. **通勤时长 vs 拥堵耗时** (相关性)
   ```python
   from scipy.stats import pearsonr
   corr, p = pearsonr(df['Q3_morning_time'], df['Q6_morning_congestion'])
   ```

---

### 2.1.2 改道行为频次与模式

**目标变量定义**：
- **高改道意愿** = Q8_2_reroute_frequency >= 4 (二分类标签)
- **平均改道倾向** = mean([Q12_1, Q12_2, Q12_3, Q12_4, Q12_5])

**分析**：
```python
# 1. 改道意愿分布
Q8_2分布 (1-5李克特)
高改道意愿比例 = (Q8_2>=4).sum() / len(df)

# 2. 不同情景下的改道分布
for scenario in ['construction', 'special_event', 'weather', 'peak_hour', 'accident']:
    plot Q12.* 的分布

# 3. 改道行为聚合
create_reroute_profile = mean([Q12.*]) - 个人平均改道倾向
```

**统计检验**：
- **信息渠道使用 vs 改道概率** (Mann-Whitney U检验)
  ```python
  # 分组：高信息使用者(num_info_channels>=3) vs 低使用者
  from scipy.stats import mannwhitneyu
  high_info = df[df['num_info_channels'] >= 3]['Q8_2_reroute_frequency']
  low_info = df[df['num_info_channels'] < 3]['Q8_2_reroute_frequency']
  u_stat, p = mannwhitneyu(high_info, low_info)
  ```

---

### 2.1.3 李克特量表响应分布

**分析所有Likert题目** (Q8.*, Q11.*, Q12.*, Q14.*, Q16.*):
- 均值、标准差、偏度
- 地板效应/天花板效应检测
- 响应一致性（Cronbach's Alpha，如果形成量表）

**可视化**：
- 堆叠条形图（每题的1-5分布）
- 小提琴图（展示分布+箱线图）

---

### 2.1.4 相关性矩阵

**变量选择**：
- 连续变量：Q3, Q4, Q6, Q7, Q13, Q15, age, total_score
- 派生变量：congestion_ratio, delay_tolerance_norm
- 偏好指数：risk_aversion, comfort_preference, efficiency_orientation, information_dependency

**矩阵类型**：
1. Pearson相关矩阵（假设线性关系）
2. Spearman秩相关（处理非线性）
3. 热力图可视化（成员4协助）

**关键假设检验**：
```python
# H1: 拥堵占比 ↑ → 改道意愿 ↑
corr = df[['congestion_ratio_morning', 'Q8_2_reroute_frequency']].corr()

# H2: 年龄 ↑ → 风险规避 ↑
corr = df[['age', 'risk_aversion_index']].corr()
```

---

### EDA报告结构
```markdown
# 中国驾驶员路径决策行为 - 探索性数据分析

## 1. 数据概况
- 样本量、特征数
- 数据质量总结

## 2. 通勤行为特征
- 2.1 通勤时间分布
- 2.2 拥堵时长分析
- 2.3 早晚通勤差异检验

## 3. 改道行为分析
- 3.1 总体改道意愿
- 3.2 情景敏感度分布
- 3.3 延误/等待阈值

## 4. 人口统计学分布
- 4.1 年龄、性别、教育
- 4.2 收入、职业分布

## 5. 相关性分析
- 5.1 关键变量相关矩阵
- 5.2 假设检验结果

## 6. 初步发现与假设
```

---

## 任务2.2：改道决策因素分析 (第2周)

### 目标
量化影响改道决策的关键因素及其权重

---

### 2.2.1 明确因变量与自变量

**因变量（改道决策）**：
- **二分类标签**: `high_reroute` = (Q8_2 >= 4) | (mean(Q12.*) >= 4)
- **连续标签**: `reroute_propensity` = mean([Q12_1, Q12_2, Q12_3, Q12_4, Q12_5])

**自变量（潜在影响因素）**：
```python
# 时间相关
- Q3_morning_time (通勤时长)
- congestion_ratio_morning (拥堵占比)
- Q13_delay_threshold (延误容忍度)
- Q15_congestion_wait_time (等待容忍度)

# 路线特征
- Q5_road_type (道路类型)
- Q14_1_prefer_familiar (路线熟悉度偏好)
- Q14_2_familiar_area (区域熟悉度)

# 信息行为
- num_info_channels (信息渠道数量)
- Q11_1_freq_seek_info (信息搜寻频率)
- information_dependency (信息依赖度指数)

# 驾驶风格
- risk_aversion_index (风险规避)
- efficiency_orientation (效率追求)
- Q14_19_aggressive (激进度)

# 人口统计学
- age, gender, education, income
```

---

### 2.2.2 统计建模方法

#### **方法1：逻辑回归** (解释性强)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 目标：预测高改道意愿
X = df[['congestion_ratio_morning', 'Q13_delay_threshold',
        'num_info_channels', 'efficiency_orientation', 'age']]
y = df['high_reroute']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# 系数解释
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
})
# 例如：congestion_ratio系数=1.2 → 拥堵占比每增加0.1，改道odds增加12%
```

#### **方法2：分位数回归** (处理异质性)
```python
import statsmodels.formula.api as smf

# 不同分位数下的影响因素
for q in [0.25, 0.5, 0.75]:
    model = smf.quantreg('reroute_propensity ~ congestion_ratio + age + efficiency_orientation', df)
    result = model.fit(q=q)
    print(f"\n{q*100}分位数回归系数:")
    print(result.summary())
```

#### **方法3：随机森林特征重要性** (处理非线性)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# 特征重要性排序
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# TOP 10关键因素
print(importance_df.head(10))
```

#### **方法4：梯度提升** (预测性能最优)
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=5, n_estimators=100)
xgb.fit(X_train, y_train)

# SHAP值解释
import shap
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # 全局特征重要性
```

---

### 2.2.3 情景权重分析

**目标**: 量化5种情景对改道决策的影响权重

**数据准备**：
```python
scenario_responses = df[['Q12_1_construction', 'Q12_2_event', 'Q12_3_weather',
                         'Q12_4_peak', 'Q12_5_accident']]
```

**方法1：主成分分析 (PCA)**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scenario_responses)

# 载荷矩阵
loadings = pca.components_
# 解释：PC1可能代表"整体改道倾向"，PC2代表"情景敏感性"
```

**方法2：因子分析**
```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=1)
fa.fit(scenario_responses)

# 因子载荷
loadings = fa.components_
# 权重归一化
weights = np.abs(loadings[0]) / np.sum(np.abs(loadings[0]))
# 例如：事故=0.28, 施工=0.22, 高峰=0.20, 天气=0.18, 特殊事件=0.12
```

**方法3：多元回归**
```python
# 以总体改道意愿为因变量
from sklearn.linear_model import Ridge

X = scenario_responses
y = df['Q8_2_reroute_frequency']

model = Ridge(alpha=1.0)
model.fit(X, y)

# 标准化系数作为权重
weights = model.coef_ / np.sum(np.abs(model.coef_))
```

---

### 2.2.4 时间-舒适权衡分析

**研究问题**: 驾驶员愿意为舒适性牺牲多少时间？

**关键变量**：
- Q14_8: 即使需要更长时间，也选择顺畅路线 (1-5)
- Q14_6: 如果改道更长时间就不改道 (1-5，反向题)
- Q13_delay_threshold: 可接受的延误时间

**分析**：
```python
# 创建"舒适偏好"组
df['comfort_first'] = (df['Q14_8'] >= 4)

# 比较两组的延误容忍度
import seaborn as sns
sns.boxplot(x='comfort_first', y='Q13_delay_threshold', data=df)

# t检验
from scipy.stats import ttest_ind
comfort_group = df[df['comfort_first'] == True]['Q13_delay_threshold']
speed_group = df[df['comfort_first'] == False]['Q13_delay_threshold']
t, p = ttest_ind(comfort_group, speed_group)

# 估算权衡比例
# 若舒适组可接受延误20分钟，速度组15分钟
# → 舒适性价值约=5分钟时间成本
```

---

### 2.2.5 路线熟悉度影响量化

**回归模型**：
```python
import statsmodels.api as sm

# 自变量：路线熟悉度偏好 + 区域熟悉度
X = df[['Q14_1_prefer_familiar', 'Q14_2_familiar_area', 'age', 'Q3_morning_time']]
X = sm.add_constant(X)
y = df['Q8_2_reroute_frequency']

model = sm.OLS(y, X).fit()
print(model.summary())

# 预期结果：
# Q14_1系数显著为负 → 偏好熟悉路线的人改道少
# Q14_2系数显著为正 → 熟悉区域的人改道多（有信心）
```

---

### 输出报告：`02_reroute_factors_analysis.pdf`
```markdown
## 改道决策影响因素分析

### 1. 因变量定义
- 二分类：高改道意愿 (Q8.2>=4 或 avg(Q12.*)>=4)
- 连续：改道倾向得分

### 2. 关键影响因素排序（随机森林重要性）
1. congestion_ratio (26.3%)
2. efficiency_orientation (18.7%)
3. Q13_delay_threshold (15.2%)
...

### 3. 逻辑回归结果
- 拥堵占比每增加0.1 → 改道odds增加1.35倍 (p<0.001)
- 年龄每增加10岁 → 改道odds降低0.87倍 (p<0.05)
...

### 4. 情景权重
- 事故: 28% (最高)
- 施工: 22%
- 高峰期: 20%
- 天气: 18%
- 特殊事件: 12%

### 5. 时间-舒适权衡
- 舒适优先组愿意多等待5.3分钟（95% CI: 3.8-6.8分钟）
```

---

## 任务2.3：驾驶员聚类分析 (第2周)

### 输入
- `data/features/preference_embeddings_cn.parquet` (来自成员1)

### 输出
- `results/clustering/cluster_model.pkl` - 聚类模型
- `results/clustering/cluster_labels.csv` - 每个样本的聚类标签
- `reports/03_driver_segmentation.html` - 聚类分析报告

---

### 2.3.1 特征选择

**使用成员1派生的偏好指数 + 行为变量**：
```python
clustering_features = [
    # 四大偏好维度（核心）
    'risk_aversion_index',
    'comfort_preference_score',
    'efficiency_orientation',
    'information_dependency',

    # 行为变量
    'Q13_delay_threshold',         # 延误容忍度
    'Q15_congestion_wait_time',    # 等待容忍度
    'congestion_ratio_morning',    # 拥堵占比
    'num_info_channels',           # 信息渠道数量
    'scenario_sensitivity',        # 情景敏感度 = mean(Q12.*)
    'Q14_1_prefer_familiar',       # 路线熟悉度偏好

    # 可选：人口统计学
    'age',
    'education'
]
```

**数据标准化**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[clustering_features])
```

---

### 2.3.2 确定最优聚类数 k

#### **方法1：肘部法则**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    from sklearn.metrics import silhouette_score
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制肘部图
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

#### **方法2：轮廓系数**
```python
# 选择silhouette_score最大的k
optimal_k = K_range[np.argmax(silhouettes)]
# 预期：k=4-6
```

#### **方法3：层次聚类树状图**
```python
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
# 观察树状图的自然分割点
```

---

### 2.3.3 执行聚类（假设k=5）

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=50)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# 保存聚类标签
df[['respondent_id', 'cluster_id']].to_csv('results/clustering/cluster_labels.csv', index=False)

# 保存模型
import joblib
joblib.dump(kmeans, 'results/clustering/cluster_model.pkl')
joblib.dump(scaler, 'results/clustering/scaler.pkl')
```

---

### 2.3.4 聚类画像分析

**统计每簇的特征均值**：
```python
cluster_profiles = df.groupby('cluster_id')[clustering_features].mean()
print(cluster_profiles)

# 示例输出：
#            risk_aversion  comfort_pref  efficiency  info_dependency  delay_thresh  ...
# cluster_id
# 0               0.72          0.68         0.35          0.65            22.3
# 1               0.45          0.42         0.78          0.52            12.8
# 2               0.58          0.55         0.55          0.82            18.5
# 3               0.38          0.35         0.82          0.31             9.7
# 4               0.65          0.72         0.38          0.45            25.1
```

**聚类命名与解释**（基于特征模式）：
```python
cluster_names = {
    0: "稳定保守型 (Stable & Cautious)",
    1: "效率优先型 (Efficiency-Driven)",
    2: "信息依赖型 (Information-Reliant)",
    3: "激进自主型 (Aggressive & Independent)",
    4: "舒适追求型 (Comfort-Seeking)"
}

# 详细画像
profiles = {
    0: {
        "name": "稳定保守型",
        "size": f"{(df['cluster_id']==0).sum()} ({(df['cluster_id']==0).sum()/len(df)*100:.1f}%)",
        "characteristics": [
            "高风险规避 (0.72)",
            "高舒适偏好 (0.68)",
            "低效率追求 (0.35)",
            "延误容忍度高 (22.3分钟)",
            "偏好熟悉路线"
        ],
        "reroute_behavior": "低改道频率，仅在严重拥堵时改道",
        "target_strategy": "提供稳定可靠的主路线，减少不确定性"
    },
    1: {
        "name": "效率优先型",
        "characteristics": [
            "低风险规避 (0.45)",
            "高效率追求 (0.78)",
            "延误容忍度低 (12.8分钟)",
            "主动寻求信息"
        ],
        "reroute_behavior": "高改道频率，对任何延误敏感",
        "target_strategy": "实时最优路径推荐，支持动态改道"
    },
    ...
}
```

---

### 2.3.5 聚类差异显著性检验

**ANOVA检验**（每个特征在聚类间的差异）：
```python
from scipy.stats import f_oneway

for feature in clustering_features:
    groups = [df[df['cluster_id'] == i][feature] for i in range(5)]
    f_stat, p_value = f_oneway(*groups)
    print(f"{feature}: F={f_stat:.2f}, p={p_value:.4f}")

# 预期：所有偏好指数的p<0.001（聚类成功）
```

**成对比较** (Tukey HSD):
```python
from scipy.stats import tukey_hsd

res = tukey_hsd(*[df[df['cluster_id'] == i]['efficiency_orientation'] for i in range(5)])
print(res.pvalue)  # 聚类0 vs 1, 0 vs 2, ...的p值矩阵
```

---

### 2.3.6 聚类可视化

**t-SNE降维**（成员4协助）：
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

import matplotlib.pyplot as plt
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['cluster_id'], cmap='viridis')
plt.title('Driver Clusters (t-SNE)')
```

**PCA双轴投影**：
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PC1可能代表"效率-舒适"轴
# PC2可能代表"自主-依赖"轴
```

---

### 输出：`03_driver_segmentation.html`
```markdown
## 中国驾驶员群体细分分析

### 1. 聚类方法
- 算法：K-Means
- 最优聚类数：5 (轮廓系数=0.58)
- 输入特征：12维偏好+行为特征

### 2. 五类驾驶员画像

#### 聚类0：稳定保守型 (28.3%, n=289)
- **偏好特征**：高风险规避、高舒适偏好、低效率追求
- **改道行为**：低频改道（Q8.2均值=2.8），仅在严重拥堵时改道
- **信息行为**：中度依赖导航，但更相信自己经验
- **人口统计**：年龄偏大（均值35.2岁），教育程度高

#### 聚类1：效率优先型 (23.7%, n=242)
...

### 3. 聚类间差异检验
- 所有偏好指数ANOVA: p < 0.001 ✓
- 改道行为差异显著: F=87.3, p<0.001

### 4. 实际应用建议
- 聚类0：推荐稳定路线，避免频繁改道建议
- 聚类1：提供实时最优路径，支持aggressive rerouting
...
```

---

## 任务2.4：决策规则提取 (第3周)

### 目标
提取可解释的IF-THEN规则，用于规则基线模型

---

### 2.4.1 目标变量定义

**针对特定情景的改道决策**：
```python
# 为每个情景创建二分类标签
df['reroute_construction'] = (df['Q12_1_construction'] >= 4).astype(int)
df['reroute_accident'] = (df['Q12_5_accident'] >= 4).astype(int)
df['reroute_weather'] = (df['Q12_3_weather'] >= 4).astype(int)
df['reroute_peak'] = (df['Q12_4_peak'] >= 4).astype(int)
df['reroute_event'] = (df['Q12_2_event'] >= 4).astype(int)
```

---

### 2.4.2 决策树规则提取 (CART)

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree

# 针对"事故情景"提取规则
X = df[['Q13_delay_threshold', 'Q14_2_familiar_area', 'efficiency_orientation',
        'congestion_ratio_morning', 'age', 'num_info_channels']]
y = df['reroute_accident']

# 训练浅层决策树（深度≤4，保证可解释性）
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
dt.fit(X, y)

# 导出规则
rules_text = export_text(dt, feature_names=list(X.columns))
print(rules_text)

# 示例输出：
# |--- efficiency_orientation <= 0.65
# |   |--- Q13_delay_threshold <= 15.00
# |   |   |--- class: 0 (不改道, n=237, prob=0.89)
# |   |--- Q13_delay_threshold > 15.00
# |   |   |--- class: 1 (改道, n=145, prob=0.72)
# |--- efficiency_orientation > 0.65
# |   |--- class: 1 (改道, n=312, prob=0.91)
```

**可视化决策树**（成员4协助）：
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
tree.plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.savefig('figures/decision_tree_accident.png', dpi=300)
```

---

### 2.4.3 规则提炼与验证

**提取TOP 20规则**：
```python
from sklearn.tree import _tree

def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, depth, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # 左子树 (<=)
            recurse(tree_.children_left[node], depth + 1,
                   conditions + [f"{name} <= {threshold:.2f}"])

            # 右子树 (>)
            recurse(tree_.children_right[node], depth + 1,
                   conditions + [f"{name} > {threshold:.2f}"])
        else:
            # 叶节点
            class_prob = tree_.value[node][0][1] / sum(tree_.value[node][0])
            support = sum(tree_.value[node][0])

            if class_prob >= 0.7:  # 只保留高置信度规则
                rules.append({
                    'conditions': ' AND '.join(conditions),
                    'prediction': 'REROUTE',
                    'confidence': class_prob,
                    'support': support
                })

    recurse(0, 1, [])
    return pd.DataFrame(rules).sort_values('confidence', ascending=False)

rules_df = extract_rules(dt, X.columns)
print(rules_df.head(20))
```

**示例规则**：
```
Rule 1:
IF efficiency_orientation > 0.65 AND Q13_delay_threshold <= 20
THEN reroute_accident = True
(Confidence: 92%, Support: 312 samples)

Rule 2:
IF congestion_ratio_morning > 0.35 AND Q14_2_familiar_area > 3
THEN reroute_accident = True
(Confidence: 87%, Support: 189 samples)

Rule 3:
IF age <= 32 AND num_info_channels >= 3
THEN reroute_accident = True
(Confidence: 81%, Support: 204 samples)
```

---

### 2.4.4 关联规则挖掘 (Apriori)

**用于发现信息渠道组合模式**：
```python
from mlxtend.frequent_patterns import apriori, association_rules

# 准备二进制数据（信息渠道使用）
info_channels = df[['Q9_navi_invehicle', 'Q9_navi_mobile', 'Q9_social_media',
                     'Q9_website', 'Q9_radio', 'Q9_tv', 'Q9_word_of_mouth',
                     'Q9_visual', 'Q9_none']].astype(bool)

# 添加改道标签
info_channels['high_reroute'] = df['high_reroute']

# 挖掘频繁项集
frequent_itemsets = apriori(info_channels, min_support=0.1, use_colnames=True)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 筛选：前项=信息渠道，后项=high_reroute
reroute_rules = rules[rules['consequents'] == frozenset({'high_reroute'})]
print(reroute_rules.sort_values('confidence', ascending=False))

# 示例输出：
# {mobile_navi, social_media} → high_reroute (支持度=15%, 置信度=78%)
# {mobile_navi, word_of_mouth} → high_reroute (支持度=12%, 置信度=73%)
```

---

### 2.4.5 构建决策优先级序列

**目标**: 在多情景并发时的决策顺序

**方法**: 基于情景权重 + 个人偏好
```python
# 情景基础权重（来自2.2.3）
scenario_base_weights = {
    'accident': 0.28,
    'construction': 0.22,
    'peak_hour': 0.20,
    'weather': 0.18,
    'special_event': 0.12
}

# 个人化权重调整
def personalized_priority(row, scenarios_active):
    """
    计算个人化的情景优先级

    Args:
        row: 包含偏好特征的Series
        scenarios_active: 当前激活的情景列表

    Returns:
        排序后的情景列表
    """
    adjusted_weights = {}

    for scenario in scenarios_active:
        base_weight = scenario_base_weights[scenario]

        # 根据个人偏好调整
        if row['risk_aversion_index'] > 0.7:
            # 高风险规避者更重视事故和天气
            if scenario in ['accident', 'weather']:
                base_weight *= 1.3

        if row['efficiency_orientation'] > 0.7:
            # 高效率追求者更重视高峰期
            if scenario == 'peak_hour':
                base_weight *= 1.4

        adjusted_weights[scenario] = base_weight

    return sorted(adjusted_weights.items(), key=lambda x: x[1], reverse=True)

# 测试
test_row = df.iloc[0]
priority = personalized_priority(test_row, ['accident', 'peak_hour', 'weather'])
print(priority)
# 输出：[('accident', 0.364), ('peak_hour', 0.28), ('weather', 0.18)]
```

---

### 输出：`results/decision_rules/`
```
decision_rules/
├── rules_accident.json
├── rules_construction.json
├── rules_weather.json
├── rules_peak.json
├── rules_event.json
├── info_channel_associations.csv
├── decision_priority_function.py
└── rules_report.md
```

**`rules_accident.json` 格式**：
```json
[
  {
    "rule_id": "ACC_R01",
    "conditions": [
      {"feature": "efficiency_orientation", "operator": ">", "threshold": 0.65},
      {"feature": "Q13_delay_threshold", "operator": "<=", "threshold": 20}
    ],
    "prediction": "REROUTE",
    "confidence": 0.92,
    "support": 312,
    "interpretation": "高效率追求者在延误容忍度适中时会改道"
  },
  ...
]
```

---

## 最终交付清单

### 第1周
- [ ] `reports/01_EDA_report.html`
- [ ] `results/eda_statistics.json`
- [ ] `figures/eda/` (20+张图表)

### 第2周
- [ ] `reports/02_reroute_factors_analysis.pdf`
- [ ] `reports/03_driver_segmentation.html`
- [ ] `results/clustering/cluster_model.pkl`
- [ ] `results/clustering/cluster_labels.csv`

### 第3周
- [ ] `results/decision_rules/` (所有规则文件)
- [ ] 规则验证报告（在测试集上的准确率）

---

## 代码规范

### 统计显著性报告格式
```python
print(f"t={t_stat:.3f}, p={p_value:.4f}, Cohen's d={cohens_d:.3f}")
# 例如：t=3.872, p=0.0001, Cohen's d=0.42
```

### 可视化风格
- 所有图表使用`seaborn.set_style('whitegrid')`
- 中文字体：`plt.rcParams['font.sans-serif'] = ['SimHei']`
- 颜色方案：使用成员4提供的调色板

---

## 参考资源
- scikit-learn文档：https://scikit-learn.org/
- statsmodels：https://www.statsmodels.org/
- SHAP解释：https://shap.readthedocs.io/
- 决策树可视化：https://dtreeviz.readthedocs.io/

---

## 项目实现示例（代码结构与任务映射）

> 按照原定周计划与改进建议，将成员2负责的统计/建模工作落地到可运行的代码项目中。目录位于 `analysis_cn/`，所有脚本均支持 CLI 调用并写入约定的 `reports/`、`results/`、`figures/` 目录。

```
analysis_cn/
├── README.md
├── pyproject.toml / requirements.txt
├── data/
│   ├── processed/
│   │   └── CN_cleaned.parquet
│   └── metadata/
│       └── codebook_cn.xlsx
├── config/
│   ├── eda.yml                # 输入/输出路径、图表风格、统计参数
│   ├── reroute_factors.yml    # 变量清单与建模超参
│   └── clustering.yml         # 聚类算法、k 值搜索空间
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── io.py               # 读写 parquet/csv/json
│   │   ├── stats.py            # t 检验、相关系数、α 系数
│   │   └── plotting.py         # seaborn 包装、色板、字体
│   ├── eda/
│   │   ├── run_eda.py          # 任务2.1 主入口（生成报告/图表/指标）
│   │   ├── sections/
│   │   │   ├── commute.py      # 2.1.1 通勤/拥堵统计
│   │   │   ├── reroute.py      # 2.1.2 改道频次
│   │   │   └── likert.py       # 2.1.3 Likert 分布&信度
│   │   └── templates/
│   │       └── report.jinja2   # HTML 报告模版
│   ├── models/
│   │   ├── reroute_factors.py  # 任务2.2 延误容忍/情景权重建模
│   │   ├  cluster_driver.py    # 任务2.3 聚类分析
│   │   ├── rules_extractor.py  # 任务2.4 决策树+关联规则
│   │   └── eval.py             # 模型评估、SHAP、交叉验证
│   └── cli.py                  # `python -m analysis_cn.cli <task>`
├── notebooks/
│   └── sanity_checks.ipynb     # 手工验证/探索
├── reports/
│   └── 01_EDA_report.html
├── results/
│   ├── eda_statistics.json
│   ├── clustering/
│   │   ├── cluster_model.pkl
│   │   └── cluster_labels.csv
│   └── decision_rules/
│       ├── rules_*.json
│       └── info_channel_associations.csv
└── figures/
    ├── eda/
    ├── reroute_factors/
    └── clustering/
```

### 任务到代码的映射

| 任务 | 模块/脚本 | 关键功能 | 输出 |
|------|-----------|----------|------|
|2.1.1|`src/eda/sections/commute.py`|计算通勤/拥堵/阈值统计量，执行配对 t 检验 & Pearson 相关|`results/eda_statistics.json` 中的 `commute_stats`；`figures/eda/commute_distribution.png`|
|2.1.2|`src/eda/sections/reroute.py`|生成改道意愿标签、高/低信息渠道分组，执行 Mann-Whitney U|`figures/eda/reroute_scenarios.png`；报告章节markdown|
|2.1.3|`src/eda/sections/likert.py`|遍历 Likert 题，输出均值/Std/偏度、Cronbach's α|`figures/eda/likert_violin.png`|
|2.1.4|`src/eda/run_eda.py:build_correlation_matrices()`|构建 Pearson/Spearman 矩阵，导出 `corr_heatmap.png`|`figures/eda/corr_heatmap.png`|
|2.2|`src/models/reroute_factors.py`|计算派生特征、拟合分位数回归 & RandomForest、生成 SHAP|`reports/02_reroute_factors_analysis.pdf`；`results/reroute_factors/importance.json`|
|2.3|`src/models/cluster_driver.py`|标准化偏好指数，GridSearch k∈[4,6]，输出簇画像|`reports/03_driver_segmentation.html`；`results/clustering/cluster_model.pkl`|
|2.4|`src/models/rules_extractor.py`|CART/RuleFit + Apriori，对不同情景生成规则 JSON|`results/decision_rules/*.json`；`rules_report.md`|

### CLI 使用示例

```bash
# 1. 初始化环境
pip install -r requirements.txt

# 2. 运行 EDA（生成报告+图表）
python -m analysis_cn.cli run-eda --config config/eda.yml

# 3. 进行改道因素分析
python -m analysis_cn.cli reroute-factors --config config/reroute_factors.yml

# 4. 聚类 + 画像
python -m analysis_cn.cli cluster --config config/clustering.yml

# 5. 决策规则提取
python -m analysis_cn.cli extract-rules --task accident
```

### 核心代码片段示例

```python
# src/models/reroute_factors.py
def train_quantile_regression(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    features = config["features"]
    target = "delay_threshold"
    qr = QuantReg(df[target], sm.add_constant(df[features]))
    models = {q: qr.fit(q=q) for q in config["quantiles"]}
    return {
        "coefficients": {q: m.params.to_dict() for q, m in models.items()},
        "goodness_of_fit": {q: m.prsquared for q, m in models.items()}
    }
```

```python
# src/models/cluster_driver.py
def build_cluster_model(df: pd.DataFrame, cfg: Dict) -> Tuple[KMeans, pd.DataFrame]:
    X = df[cfg["feature_set"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    grid = {}
    for k in cfg["k_range"]:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        grid[k] = {"model": km, "score": score, "labels": labels}
    best_k = max(grid, key=lambda k: grid[k]["score"])
    best = grid[best_k]
    df["cluster"] = best["labels"]
    save_cluster_summary(df, best_k, cfg["output_dir"])
    joblib.dump({"model": best["model"], "scaler": scaler}, cfg["model_path"])
    return best["model"], df
```

```python
# src/models/rules_extractor.py
def extract_cart_rules(df: pd.DataFrame, cfg: Dict, scenario: str) -> List[Dict]:
    mask = df["scenario"] == scenario
    X = df.loc[mask, cfg["features"]]
    y = (df.loc[mask, cfg["target"]] >= cfg["positive_threshold"]).astype(int)
    tree = DecisionTreeClassifier(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"])
    tree.fit(X, y)
    rules = tree_to_rules(tree, cfg["feature_names"])
    save_rules(rules, cfg["output_dir"] / f"rules_{scenario}.json")
    return rules
```

### 质量保障

- `pytest` 单元测试覆盖 utils & 关键模型函数（`tests/test_stats.py`、`tests/test_clustering.py`）。  
- `pre-commit`：`black`, `ruff`, `mypy`。  
- `makefile` 中定义 `make eda`, `make reroute`, `make cluster`, `make rules`, `make qa`。  
- 自动化检查：`scripts/check_metrics.py` 对生成的统计量、模型性能是否达标（如 silhouette ≥0.45、规则置信度 ≥0.8）进行验收。

> 该代码项目确保成员2能在既定 3 周周期内逐步交付统计分析、聚类与规则提取成果，同时与成员1、3、4 的数据及可视化工作流无缝衔接。
