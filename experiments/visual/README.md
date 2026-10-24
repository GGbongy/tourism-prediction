# 可视化图表索引

每个 HTML 只包含一张图，页面仅保留标题、坐标轴和图例。沿用衬线字体、配色、数据集背景与垂直虚线，不添加按钮、工具栏或说明文字。

| HTML | 内容 |
| --- | --- |
| [all_destinations.html](html/all_destinations.html) | 原始游客趋势图，展示 13 个测试期完整目的地的全部可用历史与本地测试真值。 |
| [recovery_linear_12.html](html/recovery_linear_12.html) | 最近 12 个月恢复指数的普通线性趋势，无阻尼（φ=1）：真实值、验证预测和测试预测。 |
| [recovery_median_12.html](html/recovery_median_12.html) | 最近 12 个月恢复指数的中位月增量趋势，φ=0.95：真实值、验证预测和测试预测。 |
| [recovery_median_18.html](html/recovery_median_18.html) | 最近 18 个月恢复指数的中位月增量趋势，φ=0.95：真实值、验证预测和测试预测。 |
| [sarima_fixed.html](html/sarima_fixed.html) | 固定 SARIMA(0,1,1)×(1,0,0,12)、无漂移：真实值、验证预测和测试预测。全部目的地统一阶数，系数各自拟合。 |

## 四张方法对比图的数据口径

显示 2022M01–2024M07，包含全部 20 个目的地。实线为真实值，长虚线为从 2022M07 起点一次预测的 2022M08–2023M02 验证结果，点线为从 2023M07 起点一次预测的 2023M08–2024M07 测试结果。审核期只画真实值，没有将不同起点的预测拼成一条线。缺失真实值留空，不补零；即使没有测试真值，仍显示该目的地的预测。

三条垂直虚线分别分隔训练、验证、锁定审核和测试。方法图使用原始游客人数的线性刻度，不做归一化。恢复方法沿用已冻结的反事实基线，本图只画最终游客量和真实值，不叠加反事实基线或恢复指数。

SARIMA 从原有八组固定配置中按验证集宏平均 MASE 选择一组全局配置；不再使用逐目的地混合选型。三个恢复配置则是用户明确指定的固定方法。生成图表只读取保存的预测，不拟合模型。

## 交互

- 点击目的地图例：切换该目的地的所有曲线。
- 点击线型图例：切换全部目的地的真实值、验证预测或测试预测。
- 双击图例：单独显示同组中的该项；再次双击恢复整组。
- 切换显隐时，方法图纵轴适配当前可见曲线。原始趋势图维持原有显隐行为。
- 滚轮缩放，左键拖动平移，双击绘图区复位。支持键盘聚焦图例后用 Enter 或空格切换。

## 代码与结果来源

- [generate_trends.py](generate_trends.py)：原始趋势图生成及共用 SVG/交互渲染器。
- [generate_method_comparisons.py](generate_method_comparisons.py)：四张固定方法对比图。
- `method_comparisons_manifest.json`：生成时记录预测来源、哈希、数量；作为本地 JSON 日志，已清理并忽略。
- `manifest.json`：原始趋势图生成时的来源记录；作为本地 JSON 日志，已清理并忽略。

方法预测来自 `../windowed-recovery/retained/<配置>/<时段>/outputs/` 和 `../sarima/fixed_global/<时段>/outputs/`。历史真实值来自 `../data/ISF-TDF2023.csv`，测试真实值来自相应方法保存的评分结果。

HTML 内嵌全部图形与脚本，CSV 清理后仍可离线打开。PNG、生成结果 CSV 与 JSON 日志已清理并加入忽略；原始 CSV 数据保留且不忽略。完整重建本地结果和图表：

```powershell
python experiments/run_retained.py --clean
```

仅在已生成本地预测 CSV 时，可单独画图：

```powershell
python experiments/visual/generate_method_comparisons.py
```

原版 MA、MB 及其图表、生成脚本已删除。新增或修改图表时同步维护本索引。