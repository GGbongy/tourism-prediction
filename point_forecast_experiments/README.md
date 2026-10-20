# 点预测实验工作区（Point Forecasting Experiments）

本目录是一个**完全独立**的点预测实验工作区，不修改仓库中的 Notebook、作业文件、模板或正式提交文件。它只使用训练期（≤ `2022M07`）和验证期（`2022M08`–`2023M02`），**不读取 locked audit 与最终预测期的真实值**。

## 目录

```text
point_forecast_experiments/
├── configs/experiment_config.json   # 模型网格、回测起点、随机种子
├── src/
│   ├── data_loader.py               # 数据加载、SHA256、时间隔离、缺失统计
│   ├── metrics.py                   # 与 Notebook 严格对齐的 MAE/MASE/MAPE
│   ├── baselines.py                 # lag-1、lag-12、移动平均
│   ├── statistical_models.py        # 阻尼趋势、自适应季节
│   ├── ml_models.py                 # 全局 Ridge、HistGradientBoosting、特征构造
│   ├── ensemble.py                  # 加权集成
│   ├── backtesting.py               # 训练期内 rolling-origin 回测
│   ├── run_experiments.py           # 全流程实验入口
│   ├── analyze_results.py           # 按起点/步长拆解、集成回测重建
│   └── make_figures.py              # 图表
├── results/                         # 指标、预测、回测、实验日志
├── figures/                         # 报告图表
├── reports/                         # 中文实验报告
└── run.sh                           # 一键复现
```

## 数据与时间划分

- 数据源优先级：`TOURISM_DATA_PATH` → 项目根目录 `ISF-TDF2023.csv` → 公开 URL。
- 实际数据：427 个月（`1989M01`–`2024M07`）× 20 个目的地；训练历史 403 个月（≤ `2022M07`）。
- SHA256：`8884485b0f5f772bd78f412c56f688251f38c6377b663363503a3fedd613cb1e`。
- 验证预测起点固定为 `2022M07`，目标月份 `2022M08`–`2023M02`（7 个月）。
- `load_data` 只返回训练历史和验证期真实值，locked audit / final 数据不进入实验对象。
- 缺失值：开头缺失按缺失处理；Chile（15 个内部缺失）、Maldives（3 个）、Thailand（6 个）存在内部缺失。所有特征构造只使用预测起点及以前的数据；Ridge 用训练样本中位数填补，HGB 原生处理缺失。
- lag-12 在 Chile 的 `2022M08`、`2022M09` 因 `2021M08`、`2021M09` 缺失而无定义。Notebook 原始基线（含 NaN、指标按成对非缺失样本计算）用于一致性校验；正式实验使用“同月份更早年份 → 最近观测值”的截止安全回退，并记录回退来源。

## 指标

严格复用 Notebook 定义：

- MAE：验证期绝对误差均值。
- MASE：`MAE / mean(|Δy|)`，分母来自该目的地训练历史（≤ 预测起点）。
- MAPE：排除实际值为 0 的样本，记录 `mape_valid_n`。
- 汇总同时报告宏平均（20 个目的地等权）与合并样本 MAE；本实验每个目的地验证样本数相等，因此两者数值相同。

## 模型

- 基线：lag-1、lag-12（递归实现，等同 Notebook）。
- 近期均值：最近 3 / 6 个月均值。
- 阻尼趋势：`log1p` 尺度 OLS 斜率 + 阻尼累积 + 趋势截断/总增长上限。
- 自适应季节：同月份历史值 × 近期水平调整比（带比例上限和回退）。
- 全局 Ridge / HistGradientBoosting：`(destination, origin, horizon)` 监督样本，目标为对数相对变化；特征只来自 origin 及以前。
- 集成：Ridge、阻尼趋势、HGB、自适应季节的少量固定权重组合。

## 回测

回测起点：`2017M07`、`2018M07`、`2019M07`、`2020M12`、`2021M03`，全部在 `≤ 2022M07` 内，且每个起点 20 个目的地都有观测。`2020M07` 和 `2021M07` 因 Chile 缺失被排除，以保证平衡面板；该决定记录在报告中。

每个起点重新拟合模型，预测未来 1–12 个月，只与不晚于 `2022M07` 的真实值比较。MASE 分母使用该起点之前的历史，避免未来信息。

## 运行

```bash
bash point_forecast_experiments/run.sh
```

或分步：

```bash
cd point_forecast_experiments
../.venv/Scripts/python.exe -m src.run_experiments
../.venv/Scripts/python.exe -m src.analyze_results
../.venv/Scripts/python.exe -m src.make_figures
```

## 输出

- `results/metrics_summary.csv`：全部模型的总体指标。
- `results/metrics_by_destination.csv`：目的地级指标。
- `results/metrics_by_horizon.csv`：验证期 1–7 步指标。
- `results/validation_predictions.csv`：每个模型 140 条验证预测及真实值。
- `results/backtest_metrics.csv`：回测逐样本误差（含 origin）。
- `results/backtest_summary.csv`：回测参数汇总。
- `results/backtest_by_origin.csv`、`backtest_by_horizon.csv`：按起点/步长拆分。
- `results/backtest_relative_by_origin.csv`、`backtest_relative_by_horizon.csv`：相对 lag-1 的稳定性。
- `results/backtest_ensemble_*.csv`：集成模型的回测重建。
- `results/runtime_comparison.csv`：验证运行时间。
- `results/experiment_log.json`：数据来源、SHA256、环境、种子、参数、失败记录、基线一致性检查。
- `results/report_facts.json`：报告使用的汇总事实。
- `figures/*.png`：总体、目的地、步长、回测稳定性、失败案例图。

## 环境

Python 3.13.2；numpy 2.5.4、pandas 2.3.3、scikit-learn 1.9.1、scipy 1.18.1、matplotlib 3.11.2（仅绘图，未加入 `requirements.txt`）。无 GPU。固定随机种子 742；HGB 使用该种子，其余模型确定性。完整实验约 5 分钟。

## 已知限制

- 验证集只有 7 个月、单一起点，模型排名对少数高误差目的地敏感。
- 回测跨越 2020 年疫情断层，原始 MASE 在不同起点不可直接比较；报告同时给出按起点拆解和相对 lag-1 的指标。
- `evaluate_point_forecast` 的分母先 `dropna()` 再 `diff()`，内部缺失会形成跨月差分；本实验为与 Notebook 对齐保留该定义，并在报告中说明。
- 本工作区只做点预测，不生成 `Forecast.csv`、`Intervals.csv`。
