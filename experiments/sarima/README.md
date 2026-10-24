# SARIMA 点预测实验

每个目的地独立拟合月度 SARIMA；季节周期固定为 12，无外生变量。

## 运行

```powershell
python -m pip install --target experiments/sarima/.deps -r experiments/sarima/requirements.txt
python experiments/sarima/run.py
```

依赖仅安装在本方法的 `.deps/` 中；也可在自行配置的 Python 环境安装依赖。准确版本见 `requirements-lock.txt`。

已有验证与选中模型结果时，可用 `python experiments/sarima/run.py --complete-candidate-reports` 补齐全参数测试报告，保留原选型配置文件。默认命令也会生成完整的两个时段全参数报告。

## 方法与时间划分

- 训练截止 2022M07；每个目的地比较预先指定的 8 组 SARIMA 参数，用 2022M08–2023M02 七步预测的 MAE 选型。对同一目的地，此排序也等价于 MASE 排序。
- 仅收敛且七个验证预测都有效的候选可入选；MAE 相同先选阶数总和更小的模型，再按候选 ID 排序。未用 AIC 或测试结果选型。
- 去除前导空白，保留完整月度索引和内部缺失，由状态空间滤波处理，不补零、不按行删除内部缺失。仅用拟合期观测均值做数值缩放，预测后还原。
- 不做对数变换；原始尺度预测小于零时截为零，并保存 raw_point 与 clipped_negative。
- 固定平稳性、可逆性约束。先用 L-BFGS（200 次迭代），未收敛则用 Powell（100 次迭代）重试。所有警告和收敛情况保留。
- 验证选型保存到 `outputs/validation/selected_configs.json` 后，使用截至 2023M07 的公开历史重拟合相同模型，生成 2023M08–2024M07 的 12 步预测。
- 测试真实值只在所有最终预测已生成并保存后读取，用于本地探索性评分；不参与本轮参数、缩放或模型选择。未评估锁定审核期。
- 所有 8 组参数都分别报告验证和测试表现，每个时段保留 20×8=160 条目的地—参数指标记录；无测试真实值的目的地保留记录，指标留空。
- 全参数测试报告是探索性比较，不按测试结果重新选择模型；原验证选型及其独立报告保留。
- 本地数据趋势已在先前的探索图中展示，包括审核期，因此本实验不能作为尚未查看审核数据的预注册结果。
- MAE、MASE、MAPE 与基线共享计算代码。MASE 分母仅用相应拟合截止之前历史的 lag-1 平均绝对差（先去缺失再求差）；MAPE 排除真实值为零的样本。
- 缺失测试真实值不计分；测试有效 164 条。SARIMA 验证完整 140 条；和 lag-1、lag-12 公平比较时只使用共同有效的 138 条。
- micro 是有效目的地—月份等权；macro_destination 是各目的地指标等权。验证和测试单独保存，不能直接视为同条件比较。

## 候选参数

| candidate_id | order | seasonal_order | trend |
| --- | --- | --- | --- |
| ar1_sar1 | (1, 0, 0) | (1, 0, 0, 12) | c |
| ar1_d1_sar1 | (1, 1, 0) | (1, 0, 0, 12) | n |
| ma1_d1_sar1 | (0, 1, 1) | (1, 0, 0, 12) | n |
| arma11_d1_sar1 | (1, 1, 1) | (1, 0, 0, 12) | n |
| ma1_d1_sma1_D1 | (0, 1, 1) | (0, 1, 1, 12) | n |
| ar1_d1_sma1_D1 | (1, 1, 0) | (0, 1, 1, 12) | n |
| arma11_d1_sma1_D1 | (1, 1, 1) | (0, 1, 1, 12) | n |
| ma1_d1_sar1_drift | (0, 1, 1) | (1, 0, 0, 12) | c |

trend=c 表示常数项，trend=n 表示无常数项。

## 各目的地选型

| destination | candidate_id | mae | mase | mape_percent |
| --- | --- | --- | --- | --- |
| Canada | ar1_sar1 | 1,858.2429 | 0.2911 | 27.8820 |
| Chile | ar1_sar1 | 222.1737 | 0.7574 | 21.3124 |
| Mexico | ma1_d1_sma1_D1 | 2,462.7811 | 1.6042 | 21.1164 |
| Taiwan China | ar1_d1_sar1 | 3,089.8323 | 0.1211 | 46.1353 |
| Hong Kong China | ar1_sar1 | 193,207.5694 | 0.8116 | 133.6141 |
| Japan | ar1_sar1 | 8,738.0501 | 0.3090 | 39.7524 |
| South Korea | ma1_d1_sar1_drift | 9,710.3614 | 0.3515 | 30.2224 |
| Macao China | ar1_sar1 | 393,164.7005 | 3.0744 | 66.1178 |
| Maldives | ma1_d1_sar1_drift | 1,275.1392 | 0.4736 | 30.2912 |
| Cambodia | ar1_sar1 | 8,924.6691 | 1.4770 | 26.9518 |
| Indonesia | ar1_sar1 | 6,532.4883 | 0.3876 | 33.5522 |
| Singapore | ar1_d1_sar1 | 7,992.3694 | 0.3030 | 30.7798 |
| New Zealand | ar1_d1_sar1 | 1,998.2772 | 0.5339 | 54.4234 |
| USA | ar1_d1_sar1 | 10,559.6269 | 0.4105 | 20.0644 |
| Thailand | ar1_sar1 | 22,184.1335 | 0.6140 | 25.7351 |
| Turkey | ar1_sar1 | 1,155.2694 | 0.8441 | 13.2808 |
| Australia | ma1_d1_sar1_drift | 5,545.8119 | 0.6383 | 22.9127 |
| Hawaii | ar1_d1_sar1 | 273.0852 | 0.2167 | 43.8973 |
| Austria | ma1_d1_sar1 | 965.3860 | 0.1440 | 19.7065 |
| Czech | ma1_d1_sar1 | 2,773.7419 | 0.3506 | 36.7039 |

## 文件组织

| 路径 | 内容 |
| --- | --- |
| `run.py` | 数据准备、候选搜索、预测、评分和本报告生成。 |
| `outputs/validation/` | 验证期预测、逐条误差、缩放分母、全部候选预测、拟合日志、固定选型配置。 |
| `outputs/test/` | 测试期选中模型及全部候选的预测、误差和拟合日志。 |
| `metrics/validation/`、`metrics/test/` | 两个时段各自独立保存选中模型及全部候选指标，不混在同一个文件中。 |
| `metrics/<时段>/candidate_metrics.csv` | 全部 20×8 个目的地—参数组合的 MAE、MASE、MAPE、样本数、收敛状态和失败原因。 |
| `metrics/<时段>/candidate_summary.csv` | 每组参数跨目的地的 micro 与 macro 汇总；列出参数阶数和收敛目的地数。 |
| `metrics/<时段>/candidate_by_horizon.csv` | 每组参数按预测步长的全部指标。 |
| `outputs/<时段>/candidate_predictions_long.csv` | 每组参数的原始点预测，含未截断预测值；可供可视化读取。 |
| `outputs/<时段>/candidate_scored_predictions.csv` | 全部候选预测与真实值配对后的逐条误差。 |
| `outputs/<时段>/candidate_fits.json` | 所有候选的拟合参数、警告、优化重试及收敛记录。 |
| `outputs/<时段>/candidate_report_metadata.json` | 全参数报告的数据与代码哈希、样本范围及用途。 |
| `metrics/<时段>/common_support/` | 三种方法共同有效样本上的 SARIMA 指标。 |
| `metrics/<时段>/comparison_common_support.csv` | 同一时段、相同样本上的三种方法汇总指标。 |
| `metrics/<时段>/comparison_by_destination.csv` | 同一时段、相同样本上的三种方法按目的地指标。 |
| `metrics/<时段>/improvement_common_support.csv` | 相对每个基线的指标降幅（正值改善，负值变差）。 |
| `outputs/<时段>/run_metadata.json` | 数据与代码哈希、运行环境、时间与用途。 |

可视化应直接读取以上 CSV，不重复拟合模型。

## 验证集

共同有效样本、micro 口径下，相对 lag-1：mae 下降 22.24%，mase 下降 20.88%，mape_percent 下降 1.08%。

### SARIMA 全部有效样本

| aggregation | n_predictions | n_scored | mape_valid_n | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| micro | 140 | 140 | 140 | 34,131.6855 | 0.6857 | 37.2226 |
| macro_destination | 140 | 140 | 140 | 34,131.6855 | 0.6857 | 37.2226 |

### 与基线比较（共同有效样本）

| method | aggregation | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- |
| lag-1 | micro | 138 | 44,529.9203 | 0.8757 | 38.0403 |
| lag-1 | macro_destination | 138 | 43,900.7443 | 0.8869 | 38.1526 |
| lag-12 | micro | 138 | 37,217.2971 | 1.3947 | 71.0176 |
| lag-12 | macro_destination | 138 | 36,697.1100 | 1.4139 | 71.1226 |
| sarima | micro | 138 | 34,625.5283 | 0.6928 | 37.6298 |
| sarima | macro_destination | 138 | 34,134.9984 | 0.6970 | 37.4664 |

### 按目的地

| destination | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- |
| Canada | 7 | 1,858.2429 | 0.2911 | 27.8820 |
| Chile | 7 | 222.1737 | 0.7574 | 21.3124 |
| Mexico | 7 | 2,462.7811 | 1.6042 | 21.1164 |
| Taiwan China | 7 | 3,089.8323 | 0.1211 | 46.1353 |
| Hong Kong China | 7 | 193,207.5694 | 0.8116 | 133.6141 |
| Japan | 7 | 8,738.0501 | 0.3090 | 39.7524 |
| South Korea | 7 | 9,710.3614 | 0.3515 | 30.2224 |
| Macao China | 7 | 393,164.7005 | 3.0744 | 66.1178 |
| Maldives | 7 | 1,275.1392 | 0.4736 | 30.2912 |
| Cambodia | 7 | 8,924.6691 | 1.4770 | 26.9518 |
| Indonesia | 7 | 6,532.4883 | 0.3876 | 33.5522 |
| Singapore | 7 | 7,992.3694 | 0.3030 | 30.7798 |
| New Zealand | 7 | 1,998.2772 | 0.5339 | 54.4234 |
| USA | 7 | 10,559.6269 | 0.4105 | 20.0644 |
| Thailand | 7 | 22,184.1335 | 0.6140 | 25.7351 |
| Turkey | 7 | 1,155.2694 | 0.8441 | 13.2808 |
| Australia | 7 | 5,545.8119 | 0.6383 | 22.9127 |
| Hawaii | 7 | 273.0852 | 0.2167 | 43.8973 |
| Austria | 7 | 965.3860 | 0.1440 | 19.7065 |
| Czech | 7 | 2,773.7419 | 0.3506 | 36.7039 |

### 按预测步长

| horizon | Date | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- |
| 1 | 2022M08 | 20 | 18,815.2426 | 0.3596 | 26.9139 |
| 2 | 2022M09 | 20 | 27,344.3164 | 0.4070 | 30.3101 |
| 3 | 2022M10 | 20 | 32,743.2859 | 0.4791 | 39.8221 |
| 4 | 2022M11 | 20 | 13,603.6475 | 0.4394 | 37.4116 |
| 5 | 2022M12 | 20 | 12,834.4097 | 0.6721 | 37.2185 |
| 6 | 2023M01 | 20 | 44,175.9785 | 0.9228 | 38.1023 |
| 7 | 2023M02 | 20 | 89,404.9178 | 1.5199 | 50.7797 |

MASE 较差的目的地：Macao China（3.0744）、Mexico（1.6042）、Cambodia（1.4770）。

### 全部参数：跨目的地汇总

每一行对应将同一组参数用于所有目的地；这些结果与前面的“每个目的地按验证集单独选型”系统不同。验证期每组最多 140 条配对，测试期每组最多 164 条；缺失或失败预测不计分，收敛情况单列。

| candidate_id | aggregation | converged_destinations | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| ar1_sar1 | micro | 20 | 140 | 35,395.6234 | 0.8421 | 72.3112 |
| ar1_sar1 | macro_destination | 20 | 140 | 35,395.6234 | 0.8421 | 72.3112 |
| ar1_d1_sar1 | micro | 20 | 140 | 41,424.8623 | 0.8180 | 36.7908 |
| ar1_d1_sar1 | macro_destination | 20 | 140 | 41,424.8623 | 0.8180 | 36.7908 |
| ma1_d1_sar1 | micro | 20 | 140 | 40,577.0044 | 0.8141 | 36.8679 |
| ma1_d1_sar1 | macro_destination | 20 | 140 | 40,577.0044 | 0.8141 | 36.8679 |
| arma11_d1_sar1 | micro | 20 | 140 | 41,468.4867 | 0.8929 | 40.5628 |
| arma11_d1_sar1 | macro_destination | 20 | 140 | 41,468.4867 | 0.8929 | 40.5628 |
| ma1_d1_sma1_D1 | micro | 20 | 140 | 43,298.6973 | 1.1004 | 68.7364 |
| ma1_d1_sma1_D1 | macro_destination | 20 | 140 | 43,298.6973 | 1.1004 | 68.7364 |
| ar1_d1_sma1_D1 | micro | 20 | 140 | 44,113.6041 | 1.1301 | 71.4821 |
| ar1_d1_sma1_D1 | macro_destination | 20 | 140 | 44,113.6041 | 1.1301 | 71.4821 |
| arma11_d1_sma1_D1 | micro | 20 | 140 | 43,589.5740 | 1.0749 | 67.3894 |
| arma11_d1_sma1_D1 | macro_destination | 20 | 140 | 43,589.5740 | 1.0749 | 67.3894 |
| ma1_d1_sar1_drift | micro | 20 | 140 | 40,781.2553 | 0.8195 | 37.5690 |
| ma1_d1_sar1_drift | macro_destination | 20 | 140 | 40,781.2553 | 0.8195 | 37.5690 |

### 全部参数：按目的地完整报告

| destination | candidate_id | converged | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Canada | ar1_sar1 | True | 7 | 1,858.2429 | 0.2911 | 27.8820 |
| Chile | ar1_sar1 | True | 7 | 222.1737 | 0.7574 | 21.3124 |
| Mexico | ar1_sar1 | True | 7 | 4,125.0325 | 2.6870 | 36.7644 |
| Taiwan China | ar1_sar1 | True | 7 | 17,284.9627 | 0.6776 | 388.7609 |
| Hong Kong China | ar1_sar1 | True | 7 | 193,207.5694 | 0.8116 | 133.6141 |
| Japan | ar1_sar1 | True | 7 | 8,738.0501 | 0.3090 | 39.7524 |
| South Korea | ar1_sar1 | True | 7 | 10,418.4851 | 0.3771 | 39.6390 |
| Macao China | ar1_sar1 | True | 7 | 393,164.7005 | 3.0744 | 66.1178 |
| Maldives | ar1_sar1 | True | 7 | 1,364.6537 | 0.5068 | 84.4688 |
| Cambodia | ar1_sar1 | True | 7 | 8,924.6691 | 1.4770 | 26.9518 |
| Indonesia | ar1_sar1 | True | 7 | 6,532.4883 | 0.3876 | 33.5522 |
| Singapore | ar1_sar1 | True | 7 | 12,343.2234 | 0.4680 | 69.3191 |
| New Zealand | ar1_sar1 | True | 7 | 2,920.6545 | 0.7804 | 103.0503 |
| USA | ar1_sar1 | True | 7 | 11,497.1926 | 0.4469 | 22.0221 |
| Thailand | ar1_sar1 | True | 7 | 22,184.1335 | 0.6140 | 25.7351 |
| Turkey | ar1_sar1 | True | 7 | 1,155.2694 | 0.8441 | 13.2808 |
| Australia | ar1_sar1 | True | 7 | 6,401.0293 | 0.7367 | 46.7806 |
| Hawaii | ar1_sar1 | True | 7 | 1,287.6625 | 1.0218 | 198.8929 |
| Austria | ar1_sar1 | True | 7 | 1,382.6232 | 0.2063 | 29.9524 |
| Czech | ar1_sar1 | True | 7 | 2,899.6508 | 0.3665 | 38.3744 |
| Canada | ar1_d1_sar1 | True | 7 | 2,201.4139 | 0.3448 | 31.3982 |
| Chile | ar1_d1_sar1 | True | 7 | 335.6798 | 1.1444 | 32.7305 |
| Mexico | ar1_d1_sar1 | True | 7 | 2,650.2489 | 1.7263 | 22.1525 |
| Taiwan China | ar1_d1_sar1 | True | 7 | 3,089.8323 | 0.1211 | 46.1353 |
| Hong Kong China | ar1_d1_sar1 | True | 7 | 198,637.3520 | 0.8344 | 44.9693 |
| Japan | ar1_d1_sar1 | True | 7 | 10,872.0789 | 0.3845 | 38.0399 |
| South Korea | ar1_d1_sar1 | True | 7 | 9,808.3719 | 0.3550 | 30.5265 |
| Macao China | ar1_d1_sar1 | True | 7 | 513,176.0994 | 4.0128 | 88.0498 |
| Maldives | ar1_d1_sar1 | True | 7 | 1,297.0192 | 0.4817 | 38.8763 |
| Cambodia | ar1_d1_sar1 | True | 7 | 10,136.3197 | 1.6775 | 32.4952 |
| Indonesia | ar1_d1_sar1 | True | 7 | 9,482.0555 | 0.5626 | 34.9211 |
| Singapore | ar1_d1_sar1 | True | 7 | 7,992.3694 | 0.3030 | 30.7798 |
| New Zealand | ar1_d1_sar1 | True | 7 | 1,998.2772 | 0.5339 | 54.4234 |
| USA | ar1_d1_sar1 | True | 7 | 10,559.6269 | 0.4105 | 20.0644 |
| Thailand | ar1_d1_sar1 | True | 7 | 35,019.5318 | 0.9693 | 39.8034 |
| Turkey | ar1_d1_sar1 | True | 7 | 1,551.1976 | 1.1334 | 18.1149 |
| Australia | ar1_d1_sar1 | True | 7 | 5,620.4267 | 0.6469 | 31.2275 |
| Hawaii | ar1_d1_sar1 | True | 7 | 273.0852 | 0.2167 | 43.8973 |
| Austria | ar1_d1_sar1 | True | 7 | 972.9149 | 0.1451 | 19.8847 |
| Czech | ar1_d1_sar1 | True | 7 | 2,823.3440 | 0.3568 | 37.3257 |
| Canada | ma1_d1_sar1 | True | 7 | 2,201.5918 | 0.3448 | 31.3993 |
| Chile | ma1_d1_sar1 | True | 7 | 335.7385 | 1.1446 | 32.7362 |
| Mexico | ma1_d1_sar1 | True | 7 | 2,652.1074 | 1.7275 | 22.1705 |
| Taiwan China | ma1_d1_sar1 | True | 7 | 3,163.4190 | 0.1240 | 48.0951 |
| Hong Kong China | ma1_d1_sar1 | True | 7 | 201,037.4582 | 0.8444 | 48.3240 |
| Japan | ma1_d1_sar1 | True | 7 | 10,837.2144 | 0.3833 | 37.9275 |
| South Korea | ma1_d1_sar1 | True | 7 | 9,778.0468 | 0.3539 | 30.4571 |
| Macao China | ma1_d1_sar1 | True | 7 | 492,291.8249 | 3.8495 | 83.7353 |
| Maldives | ma1_d1_sar1 | True | 7 | 1,284.3122 | 0.4770 | 30.5112 |
| Cambodia | ma1_d1_sar1 | True | 7 | 10,147.0825 | 1.6793 | 32.5525 |
| Indonesia | ma1_d1_sar1 | True | 7 | 9,918.7521 | 0.5885 | 37.0455 |
| Singapore | ma1_d1_sar1 | True | 7 | 8,716.6206 | 0.3305 | 34.7454 |
| New Zealand | ma1_d1_sar1 | True | 7 | 2,181.1681 | 0.5828 | 61.1674 |
| USA | ma1_d1_sar1 | True | 7 | 10,762.4502 | 0.4184 | 20.4986 |
| Thailand | ma1_d1_sar1 | True | 7 | 35,105.9268 | 0.9717 | 39.9939 |
| Turkey | ma1_d1_sar1 | True | 7 | 1,488.4435 | 1.0875 | 17.3580 |
| Australia | ma1_d1_sar1 | True | 7 | 5,602.4657 | 0.6448 | 23.1621 |
| Hawaii | ma1_d1_sar1 | True | 7 | 296.3373 | 0.2352 | 49.0689 |
| Austria | ma1_d1_sar1 | True | 7 | 965.3860 | 0.1440 | 19.7065 |
| Czech | ma1_d1_sar1 | True | 7 | 2,773.7419 | 0.3506 | 36.7039 |
| Canada | arma11_d1_sar1 | True | 7 | 1,886.2934 | 0.2955 | 28.1553 |
| Chile | arma11_d1_sar1 | True | 7 | 336.5967 | 1.1475 | 32.8334 |
| Mexico | arma11_d1_sar1 | True | 7 | 4,048.9751 | 2.6374 | 35.9678 |
| Taiwan China | arma11_d1_sar1 | True | 7 | 3,185.3231 | 0.1249 | 48.6478 |
| Hong Kong China | arma11_d1_sar1 | True | 7 | 199,532.9836 | 0.8381 | 46.2380 |
| Japan | arma11_d1_sar1 | True | 7 | 15,009.2130 | 0.5308 | 63.8436 |
| South Korea | arma11_d1_sar1 | True | 7 | 10,005.6308 | 0.3621 | 31.2786 |
| Macao China | arma11_d1_sar1 | True | 7 | 500,560.3520 | 3.9141 | 85.3149 |
| Maldives | arma11_d1_sar1 | True | 7 | 1,299.8553 | 0.4828 | 30.8288 |
| Cambodia | arma11_d1_sar1 | True | 7 | 9,231.9601 | 1.5278 | 28.2132 |
| Indonesia | arma11_d1_sar1 | True | 7 | 12,252.9376 | 0.7270 | 47.4994 |
| Singapore | arma11_d1_sar1 | True | 7 | 9,554.1653 | 0.3622 | 39.1377 |
| New Zealand | arma11_d1_sar1 | True | 7 | 2,106.7366 | 0.5629 | 58.4627 |
| USA | arma11_d1_sar1 | True | 7 | 11,032.3618 | 0.4289 | 21.3194 |
| Thailand | arma11_d1_sar1 | True | 7 | 36,660.8525 | 1.0148 | 42.7758 |
| Turkey | arma11_d1_sar1 | True | 7 | 1,876.3179 | 1.3709 | 21.9056 |
| Australia | arma11_d1_sar1 | True | 7 | 5,567.8958 | 0.6409 | 23.1873 |
| Hawaii | arma11_d1_sar1 | True | 7 | 298.8125 | 0.2371 | 50.0828 |
| Austria | arma11_d1_sar1 | True | 7 | 1,315.4032 | 0.1962 | 28.2124 |
| Czech | arma11_d1_sar1 | True | 7 | 3,607.0679 | 0.4559 | 47.3517 |
| Canada | ma1_d1_sma1_D1 | True | 7 | 6,028.3556 | 0.9443 | 76.0777 |
| Chile | ma1_d1_sma1_D1 | True | 7 | 282.6414 | 0.9636 | 30.4450 |
| Mexico | ma1_d1_sma1_D1 | True | 7 | 2,462.7811 | 1.6042 | 21.1164 |
| Taiwan China | ma1_d1_sma1_D1 | True | 7 | 4,948.5176 | 0.1940 | 94.9155 |
| Hong Kong China | ma1_d1_sma1_D1 | True | 7 | 230,517.8609 | 0.9683 | 96.0877 |
| Japan | ma1_d1_sma1_D1 | True | 7 | 23,334.4996 | 0.8253 | 88.3445 |
| South Korea | ma1_d1_sma1_D1 | True | 7 | 24,300.1986 | 0.8795 | 81.5075 |
| Macao China | ma1_d1_sma1_D1 | True | 7 | 421,004.1677 | 3.2921 | 62.7525 |
| Maldives | ma1_d1_sma1_D1 | True | 7 | 2,206.9929 | 0.8197 | 89.8301 |
| Cambodia | ma1_d1_sma1_D1 | True | 7 | 12,632.6957 | 2.0906 | 48.5297 |
| Indonesia | ma1_d1_sma1_D1 | True | 7 | 20,089.1687 | 1.1920 | 82.1828 |
| Singapore | ma1_d1_sma1_D1 | True | 7 | 19,797.3484 | 0.7506 | 99.6411 |
| New Zealand | ma1_d1_sma1_D1 | True | 7 | 2,929.5997 | 0.7828 | 81.7634 |
| USA | ma1_d1_sma1_D1 | True | 7 | 22,741.8103 | 0.8840 | 53.3903 |
| Thailand | ma1_d1_sma1_D1 | True | 7 | 53,496.7348 | 1.4808 | 85.6906 |
| Turkey | ma1_d1_sma1_D1 | True | 7 | 2,529.5017 | 1.8482 | 27.2124 |
| Australia | ma1_d1_sma1_D1 | True | 7 | 8,180.5133 | 0.9416 | 62.5109 |
| Hawaii | ma1_d1_sma1_D1 | True | 7 | 583.9194 | 0.4634 | 64.8772 |
| Austria | ma1_d1_sma1_D1 | True | 7 | 3,695.0930 | 0.5513 | 74.3927 |
| Czech | ma1_d1_sma1_D1 | True | 7 | 4,211.5453 | 0.5323 | 53.4605 |
| Canada | ar1_d1_sma1_D1 | True | 7 | 6,032.1023 | 0.9448 | 76.1067 |
| Chile | ar1_d1_sma1_D1 | True | 7 | 282.7099 | 0.9638 | 30.4534 |
| Mexico | ar1_d1_sma1_D1 | True | 7 | 2,471.9249 | 1.6102 | 21.1577 |
| Taiwan China | ar1_d1_sma1_D1 | True | 7 | 4,867.3127 | 0.1908 | 91.3926 |
| Hong Kong China | ar1_d1_sma1_D1 | True | 7 | 230,906.9653 | 0.9699 | 94.1566 |
| Japan | ar1_d1_sma1_D1 | True | 7 | 23,249.8932 | 0.8223 | 87.6584 |
| South Korea | ar1_d1_sma1_D1 | True | 7 | 24,631.8250 | 0.8915 | 82.5498 |
| Macao China | ar1_d1_sma1_D1 | True | 7 | 433,610.8200 | 3.3906 | 65.3847 |
| Maldives | ar1_d1_sma1_D1 | True | 7 | 2,341.1321 | 0.8695 | 98.9936 |
| Cambodia | ar1_d1_sma1_D1 | True | 7 | 12,628.1374 | 2.0899 | 48.5089 |
| Indonesia | ar1_d1_sma1_D1 | True | 7 | 20,251.4336 | 1.2016 | 82.7406 |
| Singapore | ar1_d1_sma1_D1 | True | 7 | 20,022.8424 | 0.7591 | 98.9804 |
| New Zealand | ar1_d1_sma1_D1 | True | 7 | 4,454.6517 | 1.1903 | 139.7858 |
| USA | ar1_d1_sma1_D1 | True | 7 | 23,575.6284 | 0.9164 | 55.6613 |
| Thailand | ar1_d1_sma1_D1 | True | 7 | 53,495.0712 | 1.4807 | 85.5789 |
| Turkey | ar1_d1_sma1_D1 | True | 7 | 2,526.1463 | 1.8457 | 27.1760 |
| Australia | ar1_d1_sma1_D1 | True | 7 | 8,369.8531 | 0.9634 | 63.2335 |
| Hawaii | ar1_d1_sma1_D1 | True | 7 | 502.7542 | 0.3990 | 50.3975 |
| Austria | ar1_d1_sma1_D1 | True | 7 | 3,711.5864 | 0.5537 | 74.7483 |
| Czech | ar1_d1_sma1_D1 | True | 7 | 4,339.2914 | 0.5484 | 54.9769 |
| Canada | arma11_d1_sma1_D1 | True | 7 | 4,840.0058 | 0.7581 | 63.6727 |
| Chile | arma11_d1_sma1_D1 | True | 7 | 284.4391 | 0.9697 | 30.6460 |
| Mexico | arma11_d1_sma1_D1 | True | 7 | 2,469.7766 | 1.6088 | 21.0428 |
| Taiwan China | arma11_d1_sma1_D1 | True | 7 | 4,996.0305 | 0.1959 | 96.9768 |
| Hong Kong China | arma11_d1_sma1_D1 | True | 7 | 229,883.3198 | 0.9656 | 93.6097 |
| Japan | arma11_d1_sma1_D1 | True | 7 | 23,120.6529 | 0.8177 | 86.6104 |
| South Korea | arma11_d1_sma1_D1 | True | 7 | 23,711.0972 | 0.8582 | 79.1715 |
| Macao China | arma11_d1_sma1_D1 | True | 7 | 429,467.7683 | 3.3582 | 64.3869 |
| Maldives | arma11_d1_sma1_D1 | True | 7 | 2,242.9263 | 0.8330 | 91.8395 |
| Cambodia | arma11_d1_sma1_D1 | True | 7 | 10,763.2742 | 1.7812 | 39.8292 |
| Indonesia | arma11_d1_sma1_D1 | True | 7 | 19,983.1321 | 1.1857 | 81.3995 |
| Singapore | arma11_d1_sma1_D1 | True | 7 | 19,225.1854 | 0.7289 | 96.1157 |
| New Zealand | arma11_d1_sma1_D1 | True | 7 | 2,278.7042 | 0.6089 | 68.6785 |
| USA | arma11_d1_sma1_D1 | True | 7 | 24,876.4700 | 0.9670 | 59.4579 |
| Thailand | arma11_d1_sma1_D1 | True | 7 | 53,992.9552 | 1.4945 | 85.1651 |
| Turkey | arma11_d1_sma1_D1 | True | 7 | 2,488.4016 | 1.8181 | 26.8499 |
| Australia | arma11_d1_sma1_D1 | True | 7 | 8,200.2376 | 0.9438 | 62.7400 |
| Hawaii | arma11_d1_sma1_D1 | True | 7 | 569.2755 | 0.4517 | 63.3524 |
| Austria | arma11_d1_sma1_D1 | True | 7 | 3,970.6970 | 0.5924 | 80.2658 |
| Czech | arma11_d1_sma1_D1 | True | 7 | 4,427.1303 | 0.5595 | 55.9770 |
| Canada | ma1_d1_sar1_drift | True | 7 | 2,177.0665 | 0.3410 | 31.1306 |
| Chile | ma1_d1_sar1_drift | True | 7 | 339.4425 | 1.1572 | 33.1312 |
| Mexico | ma1_d1_sar1_drift | True | 7 | 2,607.1108 | 1.6982 | 21.8299 |
| Taiwan China | ma1_d1_sar1_drift | True | 7 | 3,334.5609 | 0.1307 | 51.7125 |
| Hong Kong China | ma1_d1_sar1_drift | True | 7 | 202,161.8674 | 0.8492 | 49.5386 |
| Japan | ma1_d1_sar1_drift | True | 7 | 10,869.5347 | 0.3844 | 38.0388 |
| South Korea | ma1_d1_sar1_drift | True | 7 | 9,710.3614 | 0.3515 | 30.2224 |
| Macao China | ma1_d1_sar1_drift | True | 7 | 494,472.9412 | 3.8665 | 84.1587 |
| Maldives | ma1_d1_sar1_drift | True | 7 | 1,275.1392 | 0.4736 | 30.2912 |
| Cambodia | ma1_d1_sar1_drift | True | 7 | 10,019.7961 | 1.6582 | 31.9409 |
| Indonesia | ma1_d1_sar1_drift | True | 7 | 10,504.8758 | 0.6233 | 39.4919 |
| Singapore | ma1_d1_sar1_drift | True | 7 | 8,964.2886 | 0.3399 | 35.9336 |
| New Zealand | ma1_d1_sar1_drift | True | 7 | 2,234.2156 | 0.5970 | 62.8419 |
| USA | ma1_d1_sar1_drift | True | 7 | 10,630.2698 | 0.4132 | 20.2391 |
| Thailand | ma1_d1_sar1_drift | True | 7 | 34,948.8084 | 0.9674 | 39.7178 |
| Turkey | ma1_d1_sar1_drift | True | 7 | 1,558.5522 | 1.1388 | 18.2181 |
| Australia | ma1_d1_sar1_drift | True | 7 | 5,545.8119 | 0.6383 | 22.9127 |
| Hawaii | ma1_d1_sar1_drift | True | 7 | 297.1057 | 0.2358 | 49.3080 |
| Austria | ma1_d1_sar1_drift | True | 7 | 1,074.9311 | 0.1604 | 22.3969 |
| Czech | ma1_d1_sar1_drift | True | 7 | 2,898.4258 | 0.3663 | 38.3244 |

### 全部参数：按预测步长完整报告

| candidate_id | horizon | Date | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| ar1_sar1 | 1 | 2022M08 | 20 | 19,483.9121 | 0.4668 | 52.4811 |
| ar1_sar1 | 2 | 2022M09 | 20 | 28,435.7453 | 0.5428 | 73.2230 |
| ar1_sar1 | 3 | 2022M10 | 20 | 35,133.6845 | 0.7115 | 105.3393 |
| ar1_sar1 | 4 | 2022M11 | 20 | 16,075.4275 | 0.7205 | 90.3024 |
| ar1_sar1 | 5 | 2022M12 | 20 | 14,618.8913 | 0.8252 | 57.5236 |
| ar1_sar1 | 6 | 2023M01 | 20 | 45,427.1806 | 1.0416 | 51.5589 |
| ar1_sar1 | 7 | 2023M02 | 20 | 88,594.5224 | 1.5861 | 75.7501 |
| ar1_d1_sar1 | 1 | 2022M08 | 20 | 17,272.8320 | 0.3611 | 22.7479 |
| ar1_d1_sar1 | 2 | 2022M09 | 20 | 25,862.9635 | 0.4123 | 25.5551 |
| ar1_d1_sar1 | 3 | 2022M10 | 20 | 28,530.7060 | 0.4971 | 31.3898 |
| ar1_d1_sar1 | 4 | 2022M11 | 20 | 14,699.5131 | 0.5280 | 31.3390 |
| ar1_d1_sar1 | 5 | 2022M12 | 20 | 18,654.6418 | 0.8481 | 37.7577 |
| ar1_d1_sar1 | 6 | 2023M01 | 20 | 67,585.3689 | 1.1997 | 48.1396 |
| ar1_d1_sar1 | 7 | 2023M02 | 20 | 117,368.0105 | 1.8801 | 60.6064 |
| ma1_d1_sar1 | 1 | 2022M08 | 20 | 16,935.9141 | 0.3494 | 22.6385 |
| ma1_d1_sar1 | 2 | 2022M09 | 20 | 24,143.8217 | 0.3803 | 23.5648 |
| ma1_d1_sar1 | 3 | 2022M10 | 20 | 28,667.8038 | 0.4865 | 30.5388 |
| ma1_d1_sar1 | 4 | 2022M11 | 20 | 13,446.6910 | 0.5066 | 30.0754 |
| ma1_d1_sar1 | 5 | 2022M12 | 20 | 17,808.3950 | 0.8686 | 39.6417 |
| ma1_d1_sar1 | 6 | 2023M01 | 20 | 66,640.1057 | 1.2116 | 49.3980 |
| ma1_d1_sar1 | 7 | 2023M02 | 20 | 116,396.2995 | 1.8958 | 62.2183 |
| arma11_d1_sar1 | 1 | 2022M08 | 20 | 17,147.3706 | 0.3957 | 26.8238 |
| arma11_d1_sar1 | 2 | 2022M09 | 20 | 25,432.1939 | 0.4319 | 27.9708 |
| arma11_d1_sar1 | 3 | 2022M10 | 20 | 29,298.5977 | 0.5623 | 34.9186 |
| arma11_d1_sar1 | 4 | 2022M11 | 20 | 15,017.4622 | 0.6194 | 36.5608 |
| arma11_d1_sar1 | 5 | 2022M12 | 20 | 18,293.6462 | 0.9154 | 39.4101 |
| arma11_d1_sar1 | 6 | 2023M01 | 20 | 67,769.7944 | 1.3210 | 52.8257 |
| arma11_d1_sar1 | 7 | 2023M02 | 20 | 117,320.3421 | 2.0045 | 65.4300 |
| ma1_d1_sma1_D1 | 1 | 2022M08 | 20 | 8,675.5715 | 0.3449 | 41.7580 |
| ma1_d1_sma1_D1 | 2 | 2022M09 | 20 | 33,853.2995 | 0.7191 | 59.5044 |
| ma1_d1_sma1_D1 | 3 | 2022M10 | 20 | 27,786.8813 | 0.7117 | 65.4767 |
| ma1_d1_sma1_D1 | 4 | 2022M11 | 20 | 18,836.8848 | 0.8335 | 83.0811 |
| ma1_d1_sma1_D1 | 5 | 2022M12 | 20 | 21,881.1732 | 1.2710 | 73.4905 |
| ma1_d1_sma1_D1 | 6 | 2023M01 | 20 | 63,211.4154 | 1.5051 | 72.0129 |
| ma1_d1_sma1_D1 | 7 | 2023M02 | 20 | 128,845.6551 | 2.3175 | 85.8313 |
| ar1_d1_sma1_D1 | 1 | 2022M08 | 20 | 7,742.2687 | 0.3663 | 45.2763 |
| ar1_d1_sma1_D1 | 2 | 2022M09 | 20 | 33,970.3314 | 0.7383 | 63.2001 |
| ar1_d1_sma1_D1 | 3 | 2022M10 | 20 | 29,075.5887 | 0.7404 | 68.8108 |
| ar1_d1_sma1_D1 | 4 | 2022M11 | 20 | 20,312.3190 | 0.8657 | 86.3149 |
| ar1_d1_sma1_D1 | 5 | 2022M12 | 20 | 23,210.1878 | 1.2977 | 75.3128 |
| ar1_d1_sma1_D1 | 6 | 2023M01 | 20 | 65,491.2892 | 1.5624 | 76.0624 |
| ar1_d1_sma1_D1 | 7 | 2023M02 | 20 | 128,993.2437 | 2.3397 | 85.3973 |
| arma11_d1_sma1_D1 | 1 | 2022M08 | 20 | 7,009.0101 | 0.3304 | 37.8370 |
| arma11_d1_sma1_D1 | 2 | 2022M09 | 20 | 33,952.6148 | 0.7426 | 63.2454 |
| arma11_d1_sma1_D1 | 3 | 2022M10 | 20 | 28,684.9005 | 0.7249 | 67.7311 |
| arma11_d1_sma1_D1 | 4 | 2022M11 | 20 | 19,769.6882 | 0.7947 | 79.6131 |
| arma11_d1_sma1_D1 | 5 | 2022M12 | 20 | 22,868.1281 | 1.2499 | 71.6670 |
| arma11_d1_sma1_D1 | 6 | 2023M01 | 20 | 64,350.8787 | 1.4423 | 68.9026 |
| arma11_d1_sma1_D1 | 7 | 2023M02 | 20 | 128,491.7975 | 2.2391 | 82.7294 |
| ma1_d1_sar1_drift | 1 | 2022M08 | 20 | 17,015.1015 | 0.3519 | 23.0069 |
| ma1_d1_sar1_drift | 2 | 2022M09 | 20 | 24,286.3097 | 0.3876 | 24.3447 |
| ma1_d1_sar1_drift | 3 | 2022M10 | 20 | 28,749.9335 | 0.4950 | 31.5052 |
| ma1_d1_sar1_drift | 4 | 2022M11 | 20 | 13,686.0648 | 0.5136 | 30.9420 |
| ma1_d1_sar1_drift | 5 | 2022M12 | 20 | 18,031.7738 | 0.8625 | 39.8884 |
| ma1_d1_sar1_drift | 6 | 2023M01 | 20 | 66,947.2773 | 1.2202 | 50.1541 |
| ma1_d1_sar1_drift | 7 | 2023M02 | 20 | 116,752.3263 | 1.9061 | 63.1414 |

## 测试集（本地探索性）

共同有效样本、micro 口径下，相对 lag-1：mae 上升 11.30%，mase 下降 8.24%，mape_percent 下降 15.90%。

### SARIMA 全部有效样本

| aggregation | n_predictions | n_scored | mape_valid_n | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| micro | 240 | 164 | 164 | 88,831.5593 | 2.4492 | 21.9912 |
| macro_destination | 240 | 164 | 164 | 90,425.9653 | 2.4932 | 22.1730 |

### 与基线比较（共同有效样本）

| method | aggregation | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- |
| lag-1 | micro | 164 | 79,810.2378 | 2.6692 | 26.1497 |
| lag-1 | macro_destination | 164 | 81,356.7292 | 2.7008 | 26.2012 |
| lag-12 | micro | 164 | 277,658.8659 | 6.2913 | 62.1947 |
| lag-12 | macro_destination | 164 | 278,709.2827 | 6.3530 | 62.2817 |
| sarima | micro | 164 | 88,831.5593 | 2.4492 | 21.9912 |
| sarima | macro_destination | 164 | 90,425.9653 | 2.4932 | 22.1730 |

### 按目的地

| destination | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- |
| Canada | 0 | — | — | — |
| Chile | 0 | — | — | — |
| Mexico | 12 | 2,341.3638 | 1.4681 | 12.9650 |
| Taiwan China | 12 | 3,237.2120 | 0.1351 | 10.6033 |
| Hong Kong China | 12 | 517,471.8779 | 2.1615 | 19.6385 |
| Japan | 12 | 177,396.1623 | 6.2950 | 33.4119 |
| South Korea | 12 | 75,606.6799 | 2.7710 | 20.1269 |
| Macao China | 12 | 221,595.2121 | 1.6956 | 10.7451 |
| Maldives | 12 | 8,406.2823 | 3.1254 | 50.4712 |
| Cambodia | 12 | 7,324.8107 | 1.2148 | 11.8380 |
| Indonesia | 12 | 11,282.2357 | 0.7105 | 11.0259 |
| Singapore | 12 | 56,913.3175 | 2.1889 | 34.0928 |
| New Zealand | 12 | 5,567.5017 | 1.5247 | 24.2583 |
| USA | 0 | — | — | — |
| Thailand | 8 | 155,796.6120 | 4.2999 | 29.6301 |
| Turkey | 12 | 7,730.8223 | 5.5550 | 22.4039 |
| Australia | 12 | 15,293.4240 | 1.7601 | 19.2116 |
| Hawaii | 0 | — | — | — |
| Austria | 0 | — | — | — |
| Czech | 0 | — | — | — |

### 按预测步长

| horizon | Date | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- |
| 1 | 2023M08 | 14 | 79,924.3777 | 1.0697 | 10.2986 |
| 2 | 2023M09 | 14 | 86,290.5437 | 1.7882 | 20.5267 |
| 3 | 2023M10 | 13 | 22,236.4821 | 1.1434 | 15.5940 |
| 4 | 2023M11 | 13 | 30,473.7405 | 1.3963 | 24.6743 |
| 5 | 2023M12 | 13 | 71,369.1403 | 1.7142 | 23.7996 |
| 6 | 2024M01 | 14 | 76,623.9777 | 1.5861 | 15.6872 |
| 7 | 2024M02 | 14 | 135,930.9118 | 4.0237 | 30.0216 |
| 8 | 2024M03 | 14 | 99,460.1630 | 2.7402 | 24.6985 |
| 9 | 2024M04 | 14 | 117,166.0139 | 3.1312 | 24.9891 |
| 10 | 2024M05 | 14 | 100,962.7968 | 3.2544 | 25.5650 |
| 11 | 2024M06 | 14 | 118,685.7049 | 3.3424 | 25.4200 |
| 12 | 2024M07 | 13 | 118,824.7039 | 4.0971 | 22.5212 |

MASE 较差的目的地：Japan（6.2950）、Turkey（5.5550）、Thailand（4.2999）。

### 全部参数：跨目的地汇总

每一行对应将同一组参数用于所有目的地；这些结果与前面的“每个目的地按验证集单独选型”系统不同。验证期每组最多 140 条配对，测试期每组最多 164 条；缺失或失败预测不计分，收敛情况单列。

| candidate_id | aggregation | converged_destinations | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| ar1_sar1 | micro | 20 | 164 | 95,104.9811 | 2.7798 | 27.1372 |
| ar1_sar1 | macro_destination | 20 | 164 | 96,550.0199 | 2.8160 | 27.1966 |
| ar1_d1_sar1 | micro | 20 | 164 | 105,821.6440 | 2.2884 | 22.8102 |
| ar1_d1_sar1 | macro_destination | 20 | 164 | 105,628.2146 | 2.2981 | 22.7430 |
| ma1_d1_sar1 | micro | 20 | 164 | 101,864.8253 | 2.2126 | 21.1049 |
| ma1_d1_sar1 | macro_destination | 20 | 164 | 101,773.3935 | 2.2244 | 21.0791 |
| arma11_d1_sar1 | micro | 20 | 164 | 108,142.3192 | 2.4130 | 22.5977 |
| arma11_d1_sar1 | macro_destination | 20 | 164 | 108,819.1956 | 2.4453 | 22.6893 |
| ma1_d1_sma1_D1 | micro | 20 | 164 | 85,866.4085 | 2.6759 | 27.7822 |
| ma1_d1_sma1_D1 | macro_destination | 20 | 164 | 87,686.3796 | 2.7188 | 27.8299 |
| ar1_d1_sma1_D1 | micro | 20 | 164 | 86,516.7796 | 2.6201 | 27.4958 |
| ar1_d1_sma1_D1 | macro_destination | 20 | 164 | 88,320.4476 | 2.6643 | 27.5503 |
| arma11_d1_sma1_D1 | micro | 20 | 164 | 87,690.3252 | 2.6540 | 28.3773 |
| arma11_d1_sma1_D1 | macro_destination | 20 | 164 | 89,389.4261 | 2.6953 | 28.4005 |
| ma1_d1_sar1_drift | micro | 20 | 164 | 105,078.1059 | 2.1911 | 21.2080 |
| ma1_d1_sar1_drift | macro_destination | 20 | 164 | 104,753.8009 | 2.1991 | 21.1537 |

### 全部参数：按目的地完整报告

| destination | candidate_id | converged | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Canada | ar1_sar1 | True | 0 | — | — | — |
| Chile | ar1_sar1 | True | 0 | — | — | — |
| Mexico | ar1_sar1 | True | 12 | 4,761.2426 | 2.9855 | 27.6704 |
| Taiwan China | ar1_sar1 | True | 12 | 20,507.8201 | 0.8557 | 70.1548 |
| Hong Kong China | ar1_sar1 | True | 12 | 517,471.8779 | 2.1615 | 19.6385 |
| Japan | ar1_sar1 | True | 12 | 177,396.1623 | 6.2950 | 33.4119 |
| South Korea | ar1_sar1 | True | 12 | 111,272.4641 | 4.0782 | 30.2993 |
| Macao China | ar1_sar1 | True | 12 | 221,595.2121 | 1.6956 | 10.7451 |
| Maldives | ar1_sar1 | True | 12 | 5,116.8410 | 1.9024 | 22.0406 |
| Cambodia | ar1_sar1 | True | 12 | 7,324.8107 | 1.2148 | 11.8380 |
| Indonesia | ar1_sar1 | True | 12 | 11,282.2357 | 0.7105 | 11.0259 |
| Singapore | ar1_sar1 | True | 12 | 79,952.9071 | 3.0751 | 35.7724 |
| New Zealand | ar1_sar1 | True | 12 | 6,110.5733 | 1.6734 | 25.1201 |
| USA | ar1_sar1 | True | 0 | — | — | — |
| Thailand | ar1_sar1 | True | 8 | 155,796.6120 | 4.2999 | 29.6301 |
| Turkey | ar1_sar1 | True | 12 | 7,730.8223 | 5.5550 | 22.4039 |
| Australia | ar1_sar1 | True | 12 | 25,380.6974 | 2.9211 | 31.0007 |
| Hawaii | ar1_sar1 | True | 0 | — | — | — |
| Austria | ar1_sar1 | True | 0 | — | — | — |
| Czech | ar1_sar1 | True | 0 | — | — | — |
| Canada | ar1_d1_sar1 | True | 0 | — | — | — |
| Chile | ar1_d1_sar1 | True | 0 | — | — | — |
| Mexico | ar1_d1_sar1 | True | 12 | 2,449.7747 | 1.5361 | 13.6081 |
| Taiwan China | ar1_d1_sar1 | True | 12 | 3,237.2120 | 0.1351 | 10.6033 |
| Hong Kong China | ar1_d1_sar1 | True | 12 | 774,657.8106 | 3.2357 | 29.9412 |
| Japan | ar1_d1_sar1 | True | 12 | 138,320.5063 | 4.9083 | 26.5695 |
| South Korea | ar1_d1_sar1 | True | 12 | 78,746.5018 | 2.8861 | 20.9181 |
| Macao China | ar1_d1_sar1 | True | 12 | 274,444.1906 | 2.1000 | 14.6269 |
| Maldives | ar1_d1_sar1 | True | 12 | 10,220.2179 | 3.7998 | 63.1976 |
| Cambodia | ar1_d1_sar1 | True | 12 | 6,049.0520 | 1.0032 | 11.0748 |
| Indonesia | ar1_d1_sar1 | True | 12 | 8,154.4196 | 0.5135 | 8.8400 |
| Singapore | ar1_d1_sar1 | True | 12 | 56,913.3175 | 2.1889 | 34.0928 |
| New Zealand | ar1_d1_sar1 | True | 12 | 5,567.5017 | 1.5247 | 24.2583 |
| USA | ar1_d1_sar1 | True | 0 | — | — | — |
| Thailand | ar1_d1_sar1 | True | 8 | 97,697.6103 | 2.6964 | 19.9900 |
| Turkey | ar1_d1_sar1 | True | 12 | 5,095.4301 | 3.6613 | 15.0137 |
| Australia | ar1_d1_sar1 | True | 12 | 17,241.4592 | 1.9843 | 25.6682 |
| Hawaii | ar1_d1_sar1 | True | 0 | — | — | — |
| Austria | ar1_d1_sar1 | True | 0 | — | — | — |
| Czech | ar1_d1_sar1 | True | 0 | — | — | — |
| Canada | ma1_d1_sar1 | True | 0 | — | — | — |
| Chile | ma1_d1_sar1 | True | 0 | — | — | — |
| Mexico | ma1_d1_sar1 | True | 12 | 2,462.5071 | 1.5441 | 13.6635 |
| Taiwan China | ma1_d1_sar1 | True | 12 | 3,543.1432 | 0.1478 | 11.2954 |
| Hong Kong China | ma1_d1_sar1 | True | 12 | 742,470.6435 | 3.1013 | 28.7133 |
| Japan | ma1_d1_sar1 | True | 12 | 139,650.8804 | 4.9556 | 26.8038 |
| South Korea | ma1_d1_sar1 | True | 12 | 79,634.8549 | 2.9187 | 21.2067 |
| Macao China | ma1_d1_sar1 | True | 12 | 257,182.3600 | 1.9679 | 13.5787 |
| Maldives | ma1_d1_sar1 | True | 12 | 8,202.5897 | 3.0497 | 48.7992 |
| Cambodia | ma1_d1_sar1 | True | 12 | 6,053.1535 | 1.0039 | 11.0584 |
| Indonesia | ma1_d1_sar1 | True | 12 | 7,859.4298 | 0.4949 | 8.3472 |
| Singapore | ma1_d1_sar1 | True | 12 | 53,034.4299 | 2.0397 | 30.0659 |
| New Zealand | ma1_d1_sar1 | True | 12 | 6,330.2732 | 1.7336 | 27.5737 |
| USA | ma1_d1_sar1 | True | 0 | — | — | — |
| Thailand | ma1_d1_sar1 | True | 8 | 98,024.6896 | 2.7054 | 20.0219 |
| Turkey | ma1_d1_sar1 | True | 12 | 5,192.4995 | 3.7311 | 15.2156 |
| Australia | ma1_d1_sar1 | True | 12 | 15,186.0546 | 1.7478 | 18.7647 |
| Hawaii | ma1_d1_sar1 | True | 0 | — | — | — |
| Austria | ma1_d1_sar1 | True | 0 | — | — | — |
| Czech | ma1_d1_sar1 | True | 0 | — | — | — |
| Canada | arma11_d1_sar1 | True | 0 | — | — | — |
| Chile | arma11_d1_sar1 | True | 0 | — | — | — |
| Mexico | arma11_d1_sar1 | True | 12 | 4,521.5754 | 2.8352 | 26.2312 |
| Taiwan China | arma11_d1_sar1 | True | 12 | 3,671.7797 | 0.1532 | 11.6294 |
| Hong Kong China | arma11_d1_sar1 | True | 12 | 768,734.5245 | 3.2110 | 29.7070 |
| Japan | arma11_d1_sar1 | True | 12 | 165,054.0045 | 5.8570 | 31.2409 |
| South Korea | arma11_d1_sar1 | True | 12 | 81,976.7665 | 3.0045 | 21.9009 |
| Macao China | arma11_d1_sar1 | True | 12 | 261,812.3436 | 2.0033 | 13.8640 |
| Maldives | arma11_d1_sar1 | True | 12 | 8,185.8984 | 3.0435 | 48.6546 |
| Cambodia | arma11_d1_sar1 | True | 12 | 6,005.5783 | 0.9960 | 11.4707 |
| Indonesia | arma11_d1_sar1 | True | 12 | 10,555.4884 | 0.6647 | 10.2748 |
| Singapore | arma11_d1_sar1 | True | 12 | 50,423.7113 | 1.9393 | 26.6638 |
| New Zealand | arma11_d1_sar1 | True | 12 | 6,032.4525 | 1.6520 | 26.3226 |
| USA | arma11_d1_sar1 | True | 0 | — | — | — |
| Thailand | arma11_d1_sar1 | True | 8 | 136,571.1295 | 3.7693 | 26.4476 |
| Turkey | arma11_d1_sar1 | True | 12 | 4,659.6018 | 3.3482 | 14.0585 |
| Australia | arma11_d1_sar1 | True | 12 | 15,263.8842 | 1.7567 | 19.1850 |
| Hawaii | arma11_d1_sar1 | True | 0 | — | — | — |
| Austria | arma11_d1_sar1 | True | 0 | — | — | — |
| Czech | arma11_d1_sar1 | True | 0 | — | — | — |
| Canada | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Chile | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Mexico | ma1_d1_sma1_D1 | True | 12 | 2,341.3638 | 1.4681 | 12.9650 |
| Taiwan China | ma1_d1_sma1_D1 | True | 12 | 24,912.9512 | 1.0395 | 84.4835 |
| Hong Kong China | ma1_d1_sma1_D1 | True | 12 | 413,891.9954 | 1.7288 | 15.5951 |
| Japan | ma1_d1_sma1_D1 | True | 12 | 216,752.9944 | 7.6915 | 42.5663 |
| South Korea | ma1_d1_sma1_D1 | True | 12 | 125,220.7312 | 4.5894 | 34.0527 |
| Macao China | ma1_d1_sma1_D1 | True | 12 | 157,888.0920 | 1.2081 | 7.4622 |
| Maldives | ma1_d1_sma1_D1 | True | 12 | 7,136.1446 | 2.6532 | 40.7718 |
| Cambodia | ma1_d1_sma1_D1 | True | 12 | 11,893.1796 | 1.9724 | 18.7858 |
| Indonesia | ma1_d1_sma1_D1 | True | 12 | 17,786.2989 | 1.1200 | 18.1563 |
| Singapore | ma1_d1_sma1_D1 | True | 12 | 60,953.0665 | 2.3443 | 28.4673 |
| New Zealand | ma1_d1_sma1_D1 | True | 12 | 4,421.0952 | 1.2108 | 18.9782 |
| USA | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Thailand | ma1_d1_sma1_D1 | True | 8 | 162,305.1956 | 4.4795 | 29.7858 |
| Turkey | ma1_d1_sma1_D1 | True | 12 | 6,651.6066 | 4.7795 | 19.5335 |
| Australia | ma1_d1_sma1_D1 | True | 12 | 15,454.5990 | 1.7787 | 18.0158 |
| Hawaii | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Austria | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Czech | ma1_d1_sma1_D1 | True | 0 | — | — | — |
| Canada | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Chile | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Mexico | ar1_d1_sma1_D1 | True | 12 | 2,297.6836 | 1.4407 | 12.7679 |
| Taiwan China | ar1_d1_sma1_D1 | True | 12 | 24,397.1265 | 1.0180 | 82.6194 |
| Hong Kong China | ar1_d1_sma1_D1 | True | 12 | 434,194.6760 | 1.8136 | 16.4091 |
| Japan | ar1_d1_sma1_D1 | True | 12 | 216,113.3325 | 7.6688 | 42.4053 |
| South Korea | ar1_d1_sma1_D1 | True | 12 | 121,848.7940 | 4.4658 | 33.0930 |
| Macao China | ar1_d1_sma1_D1 | True | 12 | 158,258.8532 | 1.2110 | 7.5856 |
| Maldives | ar1_d1_sma1_D1 | True | 12 | 8,133.0558 | 3.0238 | 50.0710 |
| Cambodia | ar1_d1_sma1_D1 | True | 12 | 11,896.9306 | 1.9730 | 18.7903 |
| Indonesia | ar1_d1_sma1_D1 | True | 12 | 17,148.0164 | 1.0798 | 17.4397 |
| Singapore | ar1_d1_sma1_D1 | True | 12 | 58,915.8466 | 2.2660 | 29.0025 |
| New Zealand | ar1_d1_sma1_D1 | True | 12 | 3,337.8196 | 0.9141 | 13.4619 |
| USA | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Thailand | ar1_d1_sma1_D1 | True | 8 | 162,270.8373 | 4.4786 | 29.7823 |
| Turkey | ar1_d1_sma1_D1 | True | 12 | 6,485.4376 | 4.6601 | 18.9846 |
| Australia | ar1_d1_sma1_D1 | True | 12 | 11,187.8572 | 1.2876 | 13.2910 |
| Hawaii | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Austria | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Czech | ar1_d1_sma1_D1 | True | 0 | — | — | — |
| Canada | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Chile | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Mexico | arma11_d1_sma1_D1 | True | 12 | 2,125.7021 | 1.3329 | 11.7469 |
| Taiwan China | arma11_d1_sma1_D1 | True | 12 | 25,059.8429 | 1.0456 | 85.0138 |
| Hong Kong China | arma11_d1_sma1_D1 | True | 12 | 435,550.2167 | 1.8193 | 16.4662 |
| Japan | arma11_d1_sma1_D1 | True | 12 | 213,615.0133 | 7.5802 | 41.7718 |
| South Korea | arma11_d1_sma1_D1 | True | 12 | 126,520.0428 | 4.6370 | 34.4452 |
| Macao China | arma11_d1_sma1_D1 | True | 12 | 157,778.4440 | 1.2073 | 7.5169 |
| Maldives | arma11_d1_sma1_D1 | True | 12 | 7,117.7154 | 2.6463 | 40.6150 |
| Cambodia | arma11_d1_sma1_D1 | True | 12 | 10,066.6616 | 1.6695 | 17.6145 |
| Indonesia | arma11_d1_sma1_D1 | True | 12 | 22,454.2660 | 1.4140 | 23.3686 |
| Singapore | arma11_d1_sma1_D1 | True | 12 | 65,209.0602 | 2.5080 | 28.2566 |
| New Zealand | arma11_d1_sma1_D1 | True | 12 | 5,918.3248 | 1.6208 | 27.1092 |
| USA | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Thailand | arma11_d1_sma1_D1 | True | 8 | 159,052.5639 | 4.3897 | 29.3505 |
| Turkey | arma11_d1_sma1_D1 | True | 12 | 5,715.0322 | 4.1065 | 16.5689 |
| Australia | arma11_d1_sma1_D1 | True | 12 | 15,269.0793 | 1.7573 | 17.7627 |
| Hawaii | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Austria | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Czech | arma11_d1_sma1_D1 | True | 0 | — | — | — |
| Canada | ma1_d1_sar1_drift | True | 0 | — | — | — |
| Chile | ma1_d1_sar1_drift | True | 0 | — | — | — |
| Mexico | ma1_d1_sar1_drift | True | 12 | 2,372.8079 | 1.4878 | 13.1428 |
| Taiwan China | ma1_d1_sar1_drift | True | 12 | 3,230.8819 | 0.1348 | 10.4286 |
| Hong Kong China | ma1_d1_sar1_drift | True | 12 | 780,652.0385 | 3.2608 | 30.2046 |
| Japan | ma1_d1_sar1_drift | True | 12 | 135,018.2645 | 4.7912 | 25.9912 |
| South Korea | ma1_d1_sar1_drift | True | 12 | 75,606.6799 | 2.7710 | 20.1269 |
| Macao China | ma1_d1_sar1_drift | True | 12 | 275,356.6846 | 2.1070 | 14.6492 |
| Maldives | ma1_d1_sar1_drift | True | 12 | 8,406.2823 | 3.1254 | 50.4712 |
| Cambodia | ma1_d1_sar1_drift | True | 12 | 5,831.8435 | 0.9672 | 10.9900 |
| Indonesia | ma1_d1_sar1_drift | True | 12 | 8,604.9939 | 0.5419 | 9.3960 |
| Singapore | ma1_d1_sar1_drift | True | 12 | 53,557.5544 | 2.0599 | 30.7801 |
| New Zealand | ma1_d1_sar1_drift | True | 12 | 6,221.0030 | 1.7037 | 27.2212 |
| USA | ma1_d1_sar1_drift | True | 0 | — | — | — |
| Thailand | ma1_d1_sar1_drift | True | 8 | 91,457.2956 | 2.5242 | 18.9264 |
| Turkey | ma1_d1_sar1_drift | True | 12 | 4,943.4583 | 3.5521 | 14.6115 |
| Australia | ma1_d1_sar1_drift | True | 12 | 15,293.4240 | 1.7601 | 19.2116 |
| Hawaii | ma1_d1_sar1_drift | True | 0 | — | — | — |
| Austria | ma1_d1_sar1_drift | True | 0 | — | — | — |
| Czech | ma1_d1_sar1_drift | True | 0 | — | — | — |

### 全部参数：按预测步长完整报告

| candidate_id | horizon | Date | n_scored | mae | mase | mape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| ar1_sar1 | 1 | 2023M08 | 14 | 80,970.8911 | 0.9748 | 9.7971 |
| ar1_sar1 | 2 | 2023M09 | 14 | 86,930.2021 | 1.8491 | 23.2768 |
| ar1_sar1 | 3 | 2023M10 | 13 | 21,794.8277 | 1.1036 | 17.1081 |
| ar1_sar1 | 4 | 2023M11 | 13 | 26,420.5668 | 1.2166 | 23.6632 |
| ar1_sar1 | 5 | 2023M12 | 13 | 70,607.6547 | 1.7394 | 21.2433 |
| ar1_sar1 | 6 | 2024M01 | 14 | 86,252.2543 | 2.1635 | 25.4075 |
| ar1_sar1 | 7 | 2024M02 | 14 | 147,803.4375 | 4.9852 | 38.2323 |
| ar1_sar1 | 8 | 2024M03 | 14 | 111,435.2100 | 3.1945 | 31.4334 |
| ar1_sar1 | 9 | 2024M04 | 14 | 125,838.4145 | 3.4125 | 32.8754 |
| ar1_sar1 | 10 | 2024M05 | 14 | 109,458.5134 | 3.4200 | 30.3934 |
| ar1_sar1 | 11 | 2024M06 | 14 | 125,970.3810 | 3.4998 | 31.2257 |
| ar1_sar1 | 12 | 2024M07 | 13 | 139,022.0771 | 5.7012 | 40.5637 |
| ar1_d1_sar1 | 1 | 2023M08 | 14 | 93,511.7089 | 1.1316 | 10.2998 |
| ar1_d1_sar1 | 2 | 2023M09 | 14 | 92,387.0161 | 1.7051 | 20.3138 |
| ar1_d1_sar1 | 3 | 2023M10 | 13 | 23,266.8696 | 1.2980 | 19.7513 |
| ar1_d1_sar1 | 4 | 2023M11 | 13 | 57,089.9704 | 1.8097 | 31.7807 |
| ar1_d1_sar1 | 5 | 2023M12 | 13 | 26,722.8630 | 1.3834 | 23.1585 |
| ar1_d1_sar1 | 6 | 2024M01 | 14 | 24,790.9716 | 1.3362 | 14.2336 |
| ar1_d1_sar1 | 7 | 2024M02 | 14 | 86,567.6966 | 3.1073 | 24.3114 |
| ar1_d1_sar1 | 8 | 2024M03 | 14 | 155,470.3758 | 2.6583 | 26.7883 |
| ar1_d1_sar1 | 9 | 2024M04 | 14 | 181,516.3298 | 3.1038 | 28.4194 |
| ar1_d1_sar1 | 10 | 2024M05 | 14 | 160,529.4583 | 3.0613 | 26.9377 |
| ar1_d1_sar1 | 11 | 2024M06 | 14 | 194,545.3109 | 3.1748 | 27.1905 |
| ar1_d1_sar1 | 12 | 2024M07 | 13 | 162,480.7169 | 3.6166 | 20.8438 |
| ma1_d1_sar1 | 1 | 2023M08 | 14 | 93,144.1535 | 1.2697 | 12.2556 |
| ma1_d1_sar1 | 2 | 2023M09 | 14 | 86,588.1342 | 1.7521 | 20.5236 |
| ma1_d1_sar1 | 3 | 2023M10 | 13 | 22,581.7215 | 1.1083 | 15.7882 |
| ma1_d1_sar1 | 4 | 2023M11 | 13 | 50,194.4541 | 1.5158 | 25.8743 |
| ma1_d1_sar1 | 5 | 2023M12 | 13 | 30,450.7532 | 1.3715 | 21.7728 |
| ma1_d1_sar1 | 6 | 2024M01 | 14 | 27,457.7196 | 1.3646 | 14.6832 |
| ma1_d1_sar1 | 7 | 2024M02 | 14 | 89,789.2687 | 3.4192 | 26.6892 |
| ma1_d1_sar1 | 8 | 2024M03 | 14 | 148,727.3684 | 2.5011 | 24.0899 |
| ma1_d1_sar1 | 9 | 2024M04 | 14 | 171,894.6796 | 2.8480 | 24.5056 |
| ma1_d1_sar1 | 10 | 2024M05 | 14 | 150,879.8370 | 2.7737 | 22.5892 |
| ma1_d1_sar1 | 11 | 2024M06 | 14 | 185,025.7022 | 2.8842 | 23.0104 |
| ma1_d1_sar1 | 12 | 2024M07 | 13 | 154,983.4763 | 3.6580 | 21.5151 |
| arma11_d1_sar1 | 1 | 2023M08 | 14 | 95,549.2378 | 1.2838 | 12.2810 |
| arma11_d1_sar1 | 2 | 2023M09 | 14 | 88,378.8082 | 1.7493 | 20.5516 |
| arma11_d1_sar1 | 3 | 2023M10 | 13 | 20,435.5264 | 1.0276 | 14.5294 |
| arma11_d1_sar1 | 4 | 2023M11 | 13 | 49,127.3972 | 1.5494 | 24.6861 |
| arma11_d1_sar1 | 5 | 2023M12 | 13 | 29,148.2212 | 1.5083 | 21.4883 |
| arma11_d1_sar1 | 6 | 2024M01 | 14 | 32,253.1872 | 1.7418 | 18.0061 |
| arma11_d1_sar1 | 7 | 2024M02 | 14 | 99,007.0269 | 3.7520 | 29.8303 |
| arma11_d1_sar1 | 8 | 2024M03 | 14 | 161,281.0938 | 2.8325 | 27.0141 |
| arma11_d1_sar1 | 9 | 2024M04 | 14 | 182,242.3381 | 3.0530 | 25.6874 |
| arma11_d1_sar1 | 10 | 2024M05 | 14 | 163,277.2967 | 3.1112 | 25.8152 |
| arma11_d1_sar1 | 11 | 2024M06 | 14 | 197,802.1429 | 3.2583 | 26.6248 |
| arma11_d1_sar1 | 12 | 2024M07 | 13 | 167,309.2016 | 3.9748 | 24.2711 |
| ma1_d1_sma1_D1 | 1 | 2023M08 | 14 | 85,417.0592 | 1.0597 | 10.1967 |
| ma1_d1_sma1_D1 | 2 | 2023M09 | 14 | 58,921.0228 | 1.4551 | 22.4091 |
| ma1_d1_sma1_D1 | 3 | 2023M10 | 13 | 14,373.0574 | 0.8742 | 16.4112 |
| ma1_d1_sma1_D1 | 4 | 2023M11 | 13 | 38,382.7593 | 1.3634 | 27.4023 |
| ma1_d1_sma1_D1 | 5 | 2023M12 | 13 | 42,247.2787 | 1.8345 | 28.3013 |
| ma1_d1_sma1_D1 | 6 | 2024M01 | 14 | 40,399.6443 | 1.3272 | 17.9696 |
| ma1_d1_sma1_D1 | 7 | 2024M02 | 14 | 151,622.4844 | 4.5554 | 38.0233 |
| ma1_d1_sma1_D1 | 8 | 2024M03 | 14 | 105,036.9370 | 3.3622 | 34.4904 |
| ma1_d1_sma1_D1 | 9 | 2024M04 | 14 | 124,412.5038 | 3.7289 | 33.9912 |
| ma1_d1_sma1_D1 | 10 | 2024M05 | 14 | 108,456.6060 | 3.7394 | 34.5786 |
| ma1_d1_sma1_D1 | 11 | 2024M06 | 14 | 127,548.9262 | 3.7720 | 34.5109 |
| ma1_d1_sma1_D1 | 12 | 2024M07 | 13 | 124,741.3981 | 4.9163 | 34.8013 |
| ar1_d1_sma1_D1 | 1 | 2023M08 | 14 | 86,460.7610 | 0.9860 | 8.7384 |
| ar1_d1_sma1_D1 | 2 | 2023M09 | 14 | 61,565.3104 | 1.2765 | 19.8012 |
| ar1_d1_sma1_D1 | 3 | 2023M10 | 13 | 13,533.1025 | 0.9844 | 18.4359 |
| ar1_d1_sma1_D1 | 4 | 2023M11 | 13 | 42,498.3395 | 1.5497 | 31.2243 |
| ar1_d1_sma1_D1 | 5 | 2023M12 | 13 | 39,822.2787 | 1.8026 | 28.9563 |
| ar1_d1_sma1_D1 | 6 | 2024M01 | 14 | 37,270.1278 | 1.3567 | 18.6776 |
| ar1_d1_sma1_D1 | 7 | 2024M02 | 14 | 145,898.0384 | 4.2717 | 35.7998 |
| ar1_d1_sma1_D1 | 8 | 2024M03 | 14 | 108,676.3147 | 3.4105 | 35.8783 |
| ar1_d1_sma1_D1 | 9 | 2024M04 | 14 | 127,445.9932 | 3.6441 | 32.8589 |
| ar1_d1_sma1_D1 | 10 | 2024M05 | 14 | 110,750.9710 | 3.6964 | 33.9861 |
| ar1_d1_sma1_D1 | 11 | 2024M06 | 14 | 130,885.5539 | 3.7214 | 33.2955 |
| ar1_d1_sma1_D1 | 12 | 2024M07 | 13 | 124,408.4997 | 4.6334 | 32.3694 |
| arma11_d1_sma1_D1 | 1 | 2023M08 | 14 | 87,643.4320 | 1.1468 | 11.4227 |
| arma11_d1_sma1_D1 | 2 | 2023M09 | 14 | 61,330.4795 | 1.4662 | 23.0738 |
| arma11_d1_sma1_D1 | 3 | 2023M10 | 13 | 12,570.5380 | 0.8637 | 17.0358 |
| arma11_d1_sma1_D1 | 4 | 2023M11 | 13 | 40,175.1221 | 1.3742 | 27.9476 |
| arma11_d1_sma1_D1 | 5 | 2023M12 | 13 | 40,083.3583 | 1.8191 | 28.9034 |
| arma11_d1_sma1_D1 | 6 | 2024M01 | 14 | 39,969.9618 | 1.4194 | 19.4828 |
| arma11_d1_sma1_D1 | 7 | 2024M02 | 14 | 149,128.1393 | 4.5205 | 38.1217 |
| arma11_d1_sma1_D1 | 8 | 2024M03 | 14 | 109,827.4774 | 3.2939 | 34.9512 |
| arma11_d1_sma1_D1 | 9 | 2024M04 | 14 | 128,912.3825 | 3.6375 | 34.1923 |
| arma11_d1_sma1_D1 | 10 | 2024M05 | 14 | 112,468.8169 | 3.6536 | 35.0366 |
| arma11_d1_sma1_D1 | 11 | 2024M06 | 14 | 132,281.7248 | 3.7095 | 35.3287 |
| arma11_d1_sma1_D1 | 12 | 2024M07 | 13 | 128,658.6378 | 4.8192 | 34.6780 |
| ma1_d1_sar1_drift | 1 | 2023M08 | 14 | 91,807.3647 | 1.2571 | 12.0875 |
| ma1_d1_sar1_drift | 2 | 2023M09 | 14 | 88,749.2039 | 1.7485 | 20.5735 |
| ma1_d1_sar1_drift | 3 | 2023M10 | 13 | 23,639.2384 | 1.1214 | 16.2581 |
| ma1_d1_sar1_drift | 4 | 2023M11 | 13 | 55,641.6953 | 1.6031 | 27.1358 |
| ma1_d1_sar1_drift | 5 | 2023M12 | 13 | 24,662.6011 | 1.3601 | 21.8990 |
| ma1_d1_sar1_drift | 6 | 2024M01 | 14 | 21,924.3722 | 1.3331 | 14.4218 |
| ma1_d1_sar1_drift | 7 | 2024M02 | 14 | 89,262.8290 | 3.3072 | 25.8206 |
| ma1_d1_sar1_drift | 8 | 2024M03 | 14 | 155,092.3511 | 2.4618 | 24.0638 |
| ma1_d1_sar1_drift | 9 | 2024M04 | 14 | 179,895.7960 | 2.8370 | 25.0981 |
| ma1_d1_sar1_drift | 10 | 2024M05 | 14 | 159,594.0030 | 2.7633 | 23.0253 |
| ma1_d1_sar1_drift | 11 | 2024M06 | 14 | 194,702.6681 | 2.9047 | 23.8348 |
| ma1_d1_sar1_drift | 12 | 2024M07 | 13 | 165,164.8601 | 3.5130 | 20.3343 |

## 稳定性、开销与局限

- 验证期 160 个候选拟合中 0 个未成功收敛；测试期 160 个候选拟合中 0 个未成功收敛。两个时段共 1 次触发优化器重试。
- 本次全流程约 24.38 秒；Python 与库版本、平台、CPU 信息和分阶段耗时见运行记录。无 GPU、无付费服务，无随机采样。
- 仅七个月的验证数据用于每个目的地的八选一，验证表现存在选择偏差。不能据此证明长期泛化能力。
- 全历史拟合假设相对稳定的动态结构，疫情与恢复期的结构变化可能破坏这一假设；本轮没有专门建模干预变量，也没有使用预测区间。
- 本轮结果是预先限定候选范围的一次 SARIMA 实验，不代表所有 SARIMA 配置的最佳效果；不会根据测试成绩回头修改参数。

## 来源与复现记录

- 历史数据：[TULIP Lab ISF-TDF2023](https://github.com/tulip-lab/open-data/tree/main/ISF-TDF2023)，本地缓存 `../data/ISF-TDF2023.csv`。
- 实现依据：[statsmodels SARIMAX 官方文档](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)。
- 代码检查：`python -m unittest discover -s experiments/sarima -p "test_*.py"`，覆盖未来真实值隔离、内部缺失、候选收敛与选型规则。
- Codex 辅助实现与核查。未修改提交 notebook；测试真值只用于本地评分。

本次在已有结果基础上补齐全参数测试报告约耗时 17.66 秒；复用了此前验证拟合和 20 个选中模型的测试拟合。上述全流程耗时指原先验证选型与选中模型预测的运行。
