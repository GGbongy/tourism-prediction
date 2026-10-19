# 旅游需求预测项目

请从以下材料开始：

1. [作业说明](specification.md) — 数据、时间划分规则、必须进行的比较、输出格式及应提交的成果。
2. [起步 Notebook](Tourism-Forecasting-Starter.ipynb) — 用于加载公开数据并生成所需文件的安全起点。
3. [点预测模板](templates/Forecast.csv)和[区间预测模板](templates/Intervals.csv) — 填写时不要改变行或列。
4. [虚构的点预测示例](examples/Forecast.example.csv)和[虚构的区间预测示例](examples/Intervals.example.csv) — 对照查看两个文件的配对方式与行结构；其中数值均为编造，不是可用的预测结果。
5. [预测方案模板](templates/Forecast-Protocol.md) — 查看锁定的公开审核数据之前，先固定模型选择规则。
6. [报告提纲](templates/Report-Outline.md)和[模型卡模板](templates/Model-Card.md) — 用于组织证据、描述最终确定的系统。
7. [小组贡献声明](templates/Contribution-Declaration.md)和[提交检查清单](Submission-Checklist.md) — 打包作业前完成。
8. [输出验证脚本](validate_submission.py) — 检查文件结构与数值约束，不计算成绩，也不使用隐藏真实值。
9. 本次开课的作业页面 — 以其中的小组规则、权重、日期、文件名、提交方式和适用政策为准。

快速验证命令：

```bash
python validate_submission.py A2-Group-NAME-Forecast.csv A2-Group-NAME-Intervals.csv
```

本仓库中的说明文档供中文阅读；需要提交的 Notebook、报告、预测文件和各类已填写模板仍须使用英文。代码、CSV 列名、文件名及规定的输出格式保持原样。
