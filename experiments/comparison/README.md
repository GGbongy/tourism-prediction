# 固定配置比较汇总

完整复现并清理中间文件：`python experiments/run_retained.py --clean`。已有本地预测 CSV 时，可单独运行 `python experiments/comparison/run.py`，提取三个窗口恢复配置、按验证指标选择固定 SARIMA，并重建实验首页。

`metrics/validation.csv` 与 `metrics/test.csv` 分开保存六种方法在共同有效样本上的指标。生成结果 CSV 已清理且被忽略，重新运行时会本地生成。`outputs/manifest.json` 在本地复现时记录配置、来源与哈希，完成后随日志一起清理并忽略。本目录不拟合模型。
