# 窗口恢复指数预测（Windowed recovery）

每个目的地独立预测恢复指数，所有目的地使用同一种模型和同一组窗口、阻尼参数；不进行逐目的地选型。

当前保留的三组配置：12 个月普通线性趋势（φ=1）、12 个月中位月增量（φ=0.95）、18 个月中位月增量（φ=0.95）。

r=exp(z)=(y+1)/(b+1)，point=max(0,(b+1)r_hat-1)。窗口为预测起点前 n 个日历月，内部缺失不补零。验证起点 2022M07，测试起点 2023M07，均一次预测整个目标期。

反事实基线及疫情前滚动验证、拟合参数可在本地复现时生成至 outputs/baseline/ 与 metrics/baseline_cv/，不再依赖已删除的方法目录。基线仍保留历史目的地差异及回退。

完整复现：`python experiments/run_retained.py --clean`。分别运行时：先执行 `python experiments/windowed-recovery/run.py`，再执行 `python experiments/comparison/run.py` 与 `python experiments/visual/generate_method_comparisons.py`。

原 73 组固定配置仍可重新计算；仅当前三组列入实验首页。生成结果 CSV、PNG、JSON 日志已按要求清理并忽略，HTML 和指标快照保留。原始输入在 ../data/ 和根目录 testset.csv，不忽略。设计来源见 design.md。

本次整理只读取保存结果，没有重新调参或训练；Notebook 不修改。

## 验证集（完整 140 条）

| 配置 | MAE | MASE | MAPE |
| --- | ---: | ---: | ---: |
| [12 个月线性恢复趋势，φ=1](../visual/html/recovery_linear_12.html) | 32,568.30 | 0.7398 | 42.35% |
| [12 个月中位月增量，φ=0.95](../visual/html/recovery_median_12.html) | 44,475.77 | 0.8226 | 39.88% |
| [18 个月中位月增量，φ=0.95](../visual/html/recovery_median_18.html) | 44,335.10 | 0.8160 | 39.31% |

## 测试集（164 条有效真值）

| 配置 | MAE | MASE | MAPE |
| --- | ---: | ---: | ---: |
| [12 个月线性恢复趋势，φ=1](../visual/html/recovery_linear_12.html) | 232,238.79 | 2.9005 | 29.50% |
| [12 个月中位月增量，φ=0.95](../visual/html/recovery_median_12.html) | 89,979.55 | 1.6581 | 17.62% |
| [18 个月中位月增量，φ=0.95](../visual/html/recovery_median_18.html) | 58,915.11 | 1.6758 | 15.21% |

## 文件职责

- `retained/<配置>/<时段>/outputs/`：当前配置的长表、宽表预测及逐条误差。
- `retained/<配置>/<时段>/metrics/`：汇总、按目的地、按预测步长指标，以及与基线共同有效样本指标。
- `outputs/baseline/`、`metrics/baseline_cv/`：冻结基线和疫情前证据。
- `metrics/<时段>/candidate_*.csv`：所有固定配置的历史搜索结果；两个时段始终分开。
- `compare_validation_test.py`：匹配样本诊断，不拟合模型。
- `models.py`：近期水平、线性趋势、中位月增量及对照模型。
- `test_models.py`：窗口、缺失、数值及边界检查。
