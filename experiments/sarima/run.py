"""SARIMA point forecasting: validation-only selection, then final refit/evaluation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if (HERE / '.deps').is_dir():
    sys.path.insert(0, str(HERE / '.deps'))
sys.path.insert(0, str(HERE.parent / '_shared'))
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

from point_baselines import (aggregate, dates, forecast, md_table, mean, read_data,
                             scales, score, tables, write_csv)

# Fixed before this experiment's outcomes are inspected. No test-derived choices.
CANDIDATES = [
    {'id': 'ar1_sar1', 'order': [1, 0, 0], 'seasonal_order': [1, 0, 0, 12], 'trend': 'c'},
    {'id': 'ar1_d1_sar1', 'order': [1, 1, 0], 'seasonal_order': [1, 0, 0, 12], 'trend': 'n'},
    {'id': 'ma1_d1_sar1', 'order': [0, 1, 1], 'seasonal_order': [1, 0, 0, 12], 'trend': 'n'},
    {'id': 'arma11_d1_sar1', 'order': [1, 1, 1], 'seasonal_order': [1, 0, 0, 12], 'trend': 'n'},
    {'id': 'ma1_d1_sma1_D1', 'order': [0, 1, 1], 'seasonal_order': [0, 1, 1, 12], 'trend': 'n'},
    {'id': 'ar1_d1_sma1_D1', 'order': [1, 1, 0], 'seasonal_order': [0, 1, 1, 12], 'trend': 'n'},
    {'id': 'arma11_d1_sma1_D1', 'order': [1, 1, 1], 'seasonal_order': [0, 1, 1, 12], 'trend': 'n'},
    {'id': 'ma1_d1_sar1_drift', 'order': [0, 1, 1], 'seasonal_order': [1, 0, 0, 12], 'trend': 'c'},
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare(history, country, cutoff):
    available = sorted(d for d in history if d <= cutoff)
    observed = [d for d in available if history[d][country] is not None]
    if not observed:
        raise ValueError(f'No history for {country}')
    labels = dates(observed[0], cutoff)
    values = np.array([history.get(d, {}).get(country)
                       if history.get(d, {}).get(country) is not None else np.nan
                       for d in labels], dtype=float)
    factor = max(float(np.nanmean(np.abs(values))), 1.0)
    return values/factor, factor, labels


def fit_forecast(history, country, cutoff, targets, config):
    if targets != dates(dates(cutoff, targets[-1])[1], targets[-1]):
        raise ValueError('Targets must immediately follow cutoff and be contiguous.')
    started = time.perf_counter()
    values, factor, labels = prepare(history, country, cutoff)
    record = {'destination': country, 'candidate_id': config['id'],
              'order': config['order'], 'seasonal_order': config['seasonal_order'],
              'trend': config['trend'], 'fit_start': labels[0], 'fit_cutoff': cutoff,
              'normalization_factor': factor, 'history_months': len(values),
              'missing_history_months': int(np.isnan(values).sum()), 'attempts': []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = SARIMAX(values, order=tuple(config['order']),
                        seasonal_order=tuple(config['seasonal_order']), trend=config['trend'],
                        enforce_stationarity=True, enforce_invertibility=True,
                        simple_differencing=False)
        result = model.fit(method='lbfgs', maxiter=200, disp=False, cov_type='none')
        record['attempts'].append({'optimizer': 'lbfgs', 'maxiter': 200,
                                   'converged': bool(result.mle_retvals.get('converged', False))})
        if not record['attempts'][-1]['converged']:
            result = model.fit(start_params=result.params, method='powell', maxiter=100,
                               disp=False, cov_type='none')
            record['attempts'].append({'optimizer': 'powell', 'maxiter': 100,
                                       'converged': bool(result.mle_retvals.get('converged', False))})
        raw = np.asarray(result.forecast(steps=len(targets)), dtype=float)*factor
    converged = bool(result.mle_retvals.get('converged', False))
    finite = bool(np.isfinite(raw).all() and np.isfinite(result.params).all())
    record.update({'converged': converged, 'finite_forecast': finite,
                   'params': {n: float(v) for n, v in zip(result.param_names, result.params)
                              if math.isfinite(float(v))},
                   'aic': float(result.aic) if math.isfinite(result.aic) else None,
                   'warnings': sorted({str(w.message) for w in caught}),
                   'fit_seconds': time.perf_counter()-started})
    if not finite:
        raise ValueError(f'Nonfinite SARIMA result: {country}, {config["id"]}')
    rows = [{'Date': d, 'destination': country, 'horizon': h, 'fit_cutoff': cutoff,
             'candidate_id': config['id'], 'raw_point': float(value), 'point': max(0.0, float(value)),
             'clipped_negative': bool(value < 0)}
            for h, (d, value) in enumerate(zip(targets, raw), 1)]
    return rows, record


def select_candidate(candidates):
    eligible = [c for c in candidates if c['converged'] and c['mae'] is not None
                and math.isfinite(c['mae']) and c['n_scored'] == 7]
    if not eligible:
        raise RuntimeError('No converged candidate with all seven validation predictions.')
    return min(eligible, key=lambda c: (c['mae'], c['complexity'], c['candidate_id']))


def select_global_candidate(metrics, countries):
    eligible=[]
    for candidate in CANDIDATES:
        subset=[r for r in metrics if r['candidate_id']==candidate['id']]
        if (len(subset)==len(countries) and {r['destination'] for r in subset}==set(countries)
                and all(r['converged'] and r['n_scored']==7 and r['mase'] is not None for r in subset)):
            eligible.append((mean([r['mase'] for r in subset]),mean([r['mape_percent'] for r in subset]),candidate['id']))
    if not eligible:raise RuntimeError('No globally converged full-coverage SARIMA configuration')
    return min(eligible)[2]


def all_test_candidates(history, countries, targets, selected_predictions, selected_fits):
    """Reuse the saved selected models; fit every remaining configuration."""
    cached = {(r['destination'], r['candidate_id']): r for r in selected_fits}
    predictions, fits = [], []
    for index, country in enumerate(countries, 1):
        for config in CANDIDATES:
            key = (country, config['id'])
            if key in cached:
                fit = {**cached[key], 'fit_source': 'reused_selected_model'}
                rows = [r.copy() for r in selected_predictions
                        if (r['destination'], r['candidate_id']) == key]
                if len(rows) != len(targets):
                    raise ValueError(f'Incomplete cached predictions: {key}')
            else:
                started = time.perf_counter()
                try:
                    rows, fit = fit_forecast(history, country, '2023M07', targets, config)
                    fit['fit_source'] = 'additional_candidate_refit'
                except Exception as error:
                    fit = {'destination': country, 'candidate_id': config['id'],
                           'converged': False, 'fit_source': 'additional_candidate_refit',
                           'error': f'{type(error).__name__}: {error}', 'attempts': [],
                           'fit_seconds': time.perf_counter()-started}
                    rows = []
            predictions.extend(rows)
            fits.append(fit)
        print(f'All test configurations {index}/20: {country}', flush=True)
    return predictions, fits


def save_candidate_reports(split, history, countries, targets, predictions, actuals, fits, metadata):
    """Save every country/configuration, including missing or failed forecasts."""
    cutoff = '2022M07' if split == 'validation' else '2023M07'
    lookup = {(r['candidate_id'], r['destination'], r['Date']): r for r in predictions}
    if len(lookup) != len(predictions):
        raise ValueError('Duplicate candidate prediction keys.')
    fit_lookup = {(r['candidate_id'], r['destination']): r for r in fits}
    dense = []
    for config in CANDIDATES:
        for d in targets:
            for country in countries:
                row = lookup.get((config['id'], country, d))
                if row is None:
                    row = {'Date': d, 'destination': country, 'horizon': targets.index(d)+1,
                           'fit_cutoff': cutoff, 'candidate_id': config['id'],
                           'raw_point': None, 'point': None, 'clipped_negative': False}
                dense.append(row)
    scored = score(dense, actuals, scales(history, countries, cutoff))
    summaries, destination_metrics, horizon_metrics = [], [], []
    for config in CANDIDATES:
        subset = [r for r in scored if r['candidate_id'] == config['id']]
        summary, by_destination, by_horizon = tables(subset, countries)
        annotations = {'candidate_id': config['id'], 'order': str(tuple(config['order'])),
                       'seasonal_order': str(tuple(config['seasonal_order'])), 'trend': config['trend']}
        n_converged = sum(fit_lookup[(config['id'], c)].get('converged', False) for c in countries)
        summaries += [{**annotations, 'converged_destinations': n_converged, **r} for r in summary]
        for row in by_destination:
            fit = fit_lookup[(config['id'], row['destination'])]
            destination_metrics.append({**annotations, 'destination': row['destination'],
                'complexity': sum(config['order'])+sum(config['seasonal_order'][:3]),
                'converged': fit.get('converged', False), 'fit_seconds': fit['fit_seconds'],
                'error': fit.get('error', ''), **row})
        horizon_metrics += [{**annotations, **r} for r in by_horizon]
    output, metrics = HERE/'outputs'/split, HERE/'metrics'/split
    write_csv(output/'candidate_predictions_long.csv', dense)
    write_csv(output/'candidate_scored_predictions.csv', scored)
    write_json(output/'candidate_fits.json', fits)
    write_csv(metrics/'candidate_metrics.csv', destination_metrics)
    write_csv(metrics/'candidate_summary.csv', summaries)
    write_csv(metrics/'candidate_by_horizon.csv', horizon_metrics)
    write_json(output/'candidate_report_metadata.json', {
        **metadata, 'split': split, 'fit_cutoff': cutoff, 'target_months': targets,
        'candidate_count': len(CANDIDATES), 'destination_count': len(countries),
        'fit_records': len(fits), 'prediction_rows': len(dense),
        'test_results_used_to_change_selected_configs': False,
        'purpose': 'All-parameter exploratory reporting; original validation selection retained.'})
    return {'summary': summaries, 'by_destination': destination_metrics, 'by_horizon': horizon_metrics}


def save_partition(split, history, countries, cutoff, targets, predictions, actuals, metadata):
    predictions.sort(key=lambda r: (r['Date'], countries.index(r['destination'])))
    output, metrics = HERE/'outputs'/split, HERE/'metrics'/split
    scale = scales(history, countries, cutoff)
    scored = score(predictions, actuals, scale)
    write_csv(output/'predictions_long.csv', predictions)
    write_csv(output/'predictions_wide.csv', [
        {'Date': d, **{r['destination']: r['point'] for r in predictions if r['Date']==d}}
        for d in targets])
    write_csv(output/'scored_predictions.csv', scored)
    write_csv(output/'mase_scales.csv', list(scale.values()))
    metric_tables = tables(scored, countries)
    for filename, data in zip(('summary.csv', 'by_destination.csv', 'by_horizon.csv'), metric_tables):
        write_csv(metrics/filename, data)
    baseline_rows = {}
    for lag in (1, 12):
        baseline_rows[f'lag-{lag}'] = score(forecast(history, countries, cutoff, targets, lag), actuals, scale)
    baseline_rows['sarima'] = scored
    support = set.intersection(*[
        {(r['Date'], r['destination']) for r in rows if r['absolute_error'] is not None}
        for rows in baseline_rows.values()])
    common = [r for r in scored if (r['Date'], r['destination']) in support]
    write_csv(output/'common_support_scored_predictions.csv', common)
    common_tables = tables(common, countries)
    for filename, data in zip(('summary.csv', 'by_destination.csv', 'by_horizon.csv'), common_tables):
        write_csv(metrics/'common_support'/filename, data)
    comparisons = []
    destination_comparison = []
    for method, rows in baseline_rows.items():
        shared = [r for r in rows if (r['Date'], r['destination']) in support]
        summary, by_destination, _ = tables(shared, countries)
        comparisons += [{'method': method, **r} for r in summary]
        destination_comparison += [{'method': method, **r} for r in by_destination]
    write_csv(metrics/'comparison_common_support.csv', comparisons)
    write_csv(metrics/'comparison_by_destination.csv', destination_comparison)
    point = next(r for r in comparisons if r['method']=='sarima' and r['aggregation']=='micro')
    reductions = []
    for baseline in ('lag-1', 'lag-12'):
        reference = next(r for r in comparisons if r['method']==baseline and r['aggregation']=='micro')
        for metric in ('mae', 'mase', 'mape_percent'):
            reductions.append({'baseline': baseline, 'metric': metric,
                               'baseline_value': reference[metric], 'sarima_value': point[metric],
                               'reduction_percent': (1-point[metric]/reference[metric])*100
                               if reference[metric] else None})
    write_csv(metrics/'improvement_common_support.csv', reductions)
    write_json(output/'run_metadata.json', {**metadata, 'split': split,
               'fit_cutoff': cutoff, 'target_months': targets,
               'scoring_actuals_used_in_fit': False, 'common_support_n': len(common)})
    return {'tables': metric_tables, 'common_tables': common_tables, 'comparisons': comparisons,
            'destination_comparison': destination_comparison}


def report(reports, selected, validation_fits, final_fits, elapsed, candidate_extension_seconds=None):
    content = ['# SARIMA 点预测实验', '',
               '每个目的地独立拟合月度 SARIMA；季节周期固定为 12，无外生变量。', '',
               '## 运行', '', '```powershell',
               'python -m pip install --target experiments/sarima/.deps -r experiments/sarima/requirements.txt',
               'python experiments/sarima/run.py', '```', '',
               '依赖仅安装在本方法的 `.deps/` 中；也可在自行配置的 Python 环境安装依赖。准确版本见 `requirements-lock.txt`。', '',
               '已有验证与选中模型结果时，可用 `python experiments/sarima/run.py --complete-candidate-reports` 补齐全参数测试报告，保留原选型配置文件。默认命令也会生成完整的两个时段全参数报告。', '',
               '## 方法与时间划分', '',
               '- 训练截止 2022M07；比较预先指定的 8 组 SARIMA 参数，以 2022M08–2023M02 的宏平均 MASE 选择一组全局配置，全部目的地采用相同阶数与趋势设置，系数各自拟合。',
               '- 候选必须在全部目的地收敛、验证预测完整；宏平均 MASE 并列时按宏平均 MAPE、候选 ID 排序。未用 AIC 或测试结果选型。',
               '- 去除前导空白，保留完整月度索引和内部缺失，由状态空间滤波处理，不补零、不按行删除内部缺失。仅用拟合期观测均值做数值缩放，预测后还原。',
               '- 不做对数变换；原始尺度预测小于零时截为零，并保存 raw_point 与 clipped_negative。',
               '- 固定平稳性、可逆性约束。先用 L-BFGS（200 次迭代），未收敛则用 Powell（100 次迭代）重试。所有警告和收敛情况保留。',
               '- 验证选型保存到 `outputs/validation/selected_configs.json` 后，使用截至 2023M07 的公开历史重拟合相同模型，生成 2023M08–2024M07 的 12 步预测。',
               '- 测试真实值只在所有最终预测已生成并保存后读取，用于本地探索性评分；不参与本轮参数、缩放或模型选择。未评估锁定审核期。',
               '- 所有 8 组参数都分别报告验证和测试表现，每个时段保留 20×8=160 条目的地—参数指标记录；无测试真实值的目的地保留记录，指标留空。',
               '- 全参数测试报告是探索性比较，不按测试结果重新选择模型；原验证选型及其独立报告保留。',
               '- 本地数据趋势已在先前的探索图中展示，包括审核期，因此本实验不能作为尚未查看审核数据的预注册结果。',
               '- MAE、MASE、MAPE 与基线共享计算代码。MASE 分母仅用相应拟合截止之前历史的 lag-1 平均绝对差（先去缺失再求差）；MAPE 排除真实值为零的样本。',
               '- 缺失测试真实值不计分；测试有效 164 条。SARIMA 验证完整 140 条；和 lag-1、lag-12 公平比较时只使用共同有效的 138 条。',
               '- micro 是有效目的地—月份等权；macro_destination 是各目的地指标等权。验证和测试单独保存，不能直接视为同条件比较。', '',
               '## 候选参数', '', md_table([
                   {'candidate_id': c['id'], 'order': str(tuple(c['order'])),
                    'seasonal_order': str(tuple(c['seasonal_order'])), 'trend': c['trend']}
                   for c in CANDIDATES], ['candidate_id', 'order', 'seasonal_order', 'trend']), '',
               'trend=c 表示常数项，trend=n 表示无常数项。', '',
               '## 全局固定配置的目的地指标', '', md_table(selected, ['destination', 'candidate_id', 'mae', 'mase', 'mape_percent']), '',
               '## 文件组织', '', '| 路径 | 内容 |', '| --- | --- |',
               '| `run.py` | 数据准备、候选搜索、预测、评分和本报告生成。 |',
               '| `outputs/validation/` | 验证期预测、逐条误差、缩放分母、全部候选预测、拟合日志、固定选型配置。 |',
               '| `outputs/test/` | 测试期选中模型及全部候选的预测、误差和拟合日志。 |',
               '| `metrics/validation/`、`metrics/test/` | 两个时段各自独立保存选中模型及全部候选指标，不混在同一个文件中。 |',
               '| `metrics/<时段>/candidate_metrics.csv` | 全部 20×8 个目的地—参数组合的 MAE、MASE、MAPE、样本数、收敛状态和失败原因。 |',
               '| `metrics/<时段>/candidate_summary.csv` | 每组参数跨目的地的 micro 与 macro 汇总；列出参数阶数和收敛目的地数。 |',
               '| `metrics/<时段>/candidate_by_horizon.csv` | 每组参数按预测步长的全部指标。 |',
               '| `outputs/<时段>/candidate_predictions_long.csv` | 每组参数的原始点预测，含未截断预测值；可供可视化读取。 |',
               '| `outputs/<时段>/candidate_scored_predictions.csv` | 全部候选预测与真实值配对后的逐条误差。 |',
               '| `outputs/<时段>/candidate_fits.json` | 所有候选的拟合参数、警告、优化重试及收敛记录。 |',
               '| `outputs/<时段>/candidate_report_metadata.json` | 全参数报告的数据与代码哈希、样本范围及用途。 |',
               '| `metrics/<时段>/common_support/` | 三种方法共同有效样本上的 SARIMA 指标。 |',
               '| `metrics/<时段>/comparison_common_support.csv` | 同一时段、相同样本上的三种方法汇总指标。 |',
               '| `metrics/<时段>/comparison_by_destination.csv` | 同一时段、相同样本上的三种方法按目的地指标。 |',
               '| `metrics/<时段>/improvement_common_support.csv` | 相对每个基线的指标降幅（正值改善，负值变差）。 |',
               '| `outputs/<时段>/run_metadata.json` | 数据与代码哈希、运行环境、时间与用途。 |', '',
               '可视化应直接读取以上 CSV，不重复拟合模型。']
    for split, title in [('validation', '验证集'), ('test', '测试集（本地探索性）')]:
        data = reports[split]
        summary, by_destination, by_horizon = data['tables']
        point = next(r for r in data['comparisons'] if r['method']=='sarima' and r['aggregation']=='micro')
        base = next(r for r in data['comparisons'] if r['method']=='lag-1' and r['aggregation']=='micro')
        changes = []
        for metric in ('mae', 'mase', 'mape_percent'):
            change = (point[metric]/base[metric]-1)*100
            changes.append(f'{metric} {"下降" if change < 0 else "上升"} {abs(change):.2f}%')
        content += ['', f'## {title}', '',
                    '共同有效样本、micro 口径下，相对 lag-1：' + '，'.join(changes) + '。', '',
                    '### SARIMA 全部有效样本', '',
                    md_table(summary, ['aggregation', 'n_predictions', 'n_scored', 'mape_valid_n', 'mae', 'mase', 'mape_percent']), '',
                    '### 与基线比较（共同有效样本）', '',
                    md_table(data['comparisons'], ['method', 'aggregation', 'n_scored', 'mae', 'mase', 'mape_percent']), '',
                    '### 按目的地', '', md_table(by_destination, ['destination', 'n_scored', 'mae', 'mase', 'mape_percent']), '',
                    '### 按预测步长', '', md_table(by_horizon, ['horizon', 'Date', 'n_scored', 'mae', 'mase', 'mape_percent'])]
        worst = sorted([r for r in by_destination if r['mase'] is not None], key=lambda r: r['mase'], reverse=True)[:3]
        content += ['', 'MASE 较差的目的地：' + '、'.join(f'{r["destination"]}（{r["mase"]:.4f}）' for r in worst) + '。']
        candidates = data['candidate_report']
        content += ['', '### 全部参数：跨目的地汇总', '',
                    '每一行对应将同一组参数用于所有目的地；这些结果与前面的“每个目的地按验证集单独选型”系统不同。验证期每组最多 140 条配对，测试期每组最多 164 条；缺失或失败预测不计分，收敛情况单列。', '',
                    md_table(candidates['summary'], ['candidate_id', 'aggregation', 'converged_destinations', 'n_scored', 'mae', 'mase', 'mape_percent']), '',
                    '### 全部参数：按目的地完整报告', '',
                    md_table(candidates['by_destination'], ['destination', 'candidate_id', 'converged', 'n_scored', 'mae', 'mase', 'mape_percent']), '',
                    '### 全部参数：按预测步长完整报告', '',
                    md_table(candidates['by_horizon'], ['candidate_id', 'horizon', 'Date', 'n_scored', 'mae', 'mase', 'mape_percent'])]
    total_candidates = len(validation_fits)
    failed = sum(not r.get('converged', False) for r in validation_fits)
    retries = sum(len(r.get('attempts', [])) > 1 for r in validation_fits + final_fits)
    content += ['', '## 稳定性、开销与局限', '',
                f'- 验证期 {total_candidates} 个候选拟合中 {failed} 个未成功收敛；测试期 {len(final_fits)} 个候选拟合中 {sum(not r.get("converged", False) for r in final_fits)} 个未成功收敛。两个时段共 {retries} 次触发优化器重试。',
                f'- 本次全流程约 {elapsed:.2f} 秒；Python 与库版本、平台、CPU 信息和分阶段耗时见运行记录。无 GPU、无付费服务，无随机采样。',
                '- 仅七个月的验证数据用于八组配置的全局选择，验证表现存在选择偏差。不能据此证明长期泛化能力。',
                '- 全历史拟合假设相对稳定的动态结构，疫情与恢复期的结构变化可能破坏这一假设；本轮没有专门建模干预变量，也没有使用预测区间。',
                '- 本轮结果是预先限定候选范围的一次 SARIMA 实验，不代表所有 SARIMA 配置的最佳效果；不会根据测试成绩回头修改参数。', '',
                '## 来源与复现记录', '',
                '- 历史数据：[TULIP Lab ISF-TDF2023](https://github.com/tulip-lab/open-data/tree/main/ISF-TDF2023)，本地缓存 `../data/ISF-TDF2023.csv`。',
                '- 实现依据：[statsmodels SARIMAX 官方文档](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)。',
                '- 代码检查：`python -m unittest discover -s experiments/sarima -p "test_*.py"`，覆盖未来真实值隔离、内部缺失、候选收敛与选型规则。',
                '- Codex 辅助实现与核查。未修改提交 notebook；测试真值只用于本地评分。']
    if candidate_extension_seconds is not None:
        content += ['', f'本次在已有结果基础上补齐全参数测试报告约耗时 {candidate_extension_seconds:.2f} 秒；复用了此前验证拟合和 20 个选中模型的测试拟合。上述全流程耗时指原先验证选型与选中模型预测的运行。']
    (HERE/'README.md').write_text('\n'.join(content)+'\n', encoding='utf-8')


def load_csv(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def complete_candidate_reports(args):
    """Extend the existing run without replacing its selected configurations."""
    started = time.perf_counter()
    notebook = ROOT/'Tourism-Forecasting-Starter.ipynb'
    selection_path = HERE/'outputs/validation/selected_configs.json'
    notebook_hash, selection_hash = digest(notebook), digest(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    countries, history = read_data(args.history)
    if digest(args.history) != selection['history_sha256'] or selection['candidates'] != CANDIDATES:
        raise ValueError('History or candidate grid differs from the saved validation selection.')
    history = {d: r for d, r in history.items() if d <= '2023M07'}
    reports = {}
    for split in ('validation', 'test'):
        base = HERE/'metrics'/split
        reports[split] = {'tables': tuple(load_csv(base/f) for f in
            ('summary.csv', 'by_destination.csv', 'by_horizon.csv')),
            'comparisons': load_csv(base/'comparison_common_support.csv')}
    validation_predictions = load_csv(HERE/'outputs/validation/candidate_predictions_long.csv')
    validation_fits = json.loads((HERE/'outputs/validation/candidate_fits.json').read_text())
    selected_predictions = load_csv(HERE/'outputs/test/predictions_long.csv')
    selected_fits = json.loads((HERE/'outputs/test/final_fits.json').read_text())
    test_targets = dates('2023M08', '2024M07')
    test_predictions, test_fits = all_test_candidates(history, countries, test_targets,
                                                     selected_predictions, selected_fits)
    write_csv(HERE/'outputs/test/candidate_predictions_long.csv', test_predictions)
    # Only after all candidate forecasts have been saved do we read test actuals.
    test_countries, test = read_data(args.test)
    if test_countries != countries or sorted(test) != test_targets:
        raise ValueError('Unexpected test destinations or months.')
    metadata = {'history_sha256': digest(args.history),
                'report_implementation_sha256': digest(Path(__file__)),
                'reused_fitting_implementation_sha256': selection['implementation_sha256'],
                'selected_configs_sha256': selection_hash,
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'selected_validation_fit_reused_count': 160,
                'selected_test_fit_reused_count': 20, 'additional_test_fit_count': 140}
    reports['validation']['candidate_report'] = save_candidate_reports('validation', history,
        countries, dates('2022M08', '2023M02'), validation_predictions, history, validation_fits, metadata)
    reports['test']['candidate_report'] = save_candidate_reports('test', history, countries,
        test_targets, test_predictions, test, test_fits, {**metadata, 'actual_sha256': digest(args.test)})
    candidate_metrics = reports['validation']['candidate_report']['by_destination']
    selected = [next(r for r in candidate_metrics if r['destination']==country
                     and r['candidate_id']==selection['selected'][country]['id']) for country in countries]
    original_elapsed = sum(json.loads((HERE/f'outputs/{s}/run_metadata.json').read_text())['elapsed_seconds']
                           for s in ('validation', 'test'))
    report(reports, selected, validation_fits, test_fits, original_elapsed,
           candidate_extension_seconds=time.perf_counter()-started)
    if digest(selection_path) != selection_hash or digest(notebook) != notebook_hash:
        raise RuntimeError('Frozen selection or notebook changed during report extension.')
    print('Completed: 160 country/configuration metrics per partition; frozen selection and notebook unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=ROOT/'experiments/data/ISF-TDF2023.csv')
    parser.add_argument('--test', type=Path, default=ROOT/'testset.csv')
    parser.add_argument('--complete-candidate-reports', action='store_true',
                        help='Reuse saved validation and selected-model results; add all test candidates.')
    args = parser.parse_args()
    if args.complete_candidate_reports:
        complete_candidate_reports(args)
        return
    started = time.perf_counter()
    notebook = ROOT/'Tourism-Forecasting-Starter.ipynb'
    notebook_hash = digest(notebook)
    countries, history = read_data(args.history)
    if len(countries) != 20:
        raise ValueError('Expected 20 destinations.')
    history = {d: r for d, r in history.items() if d <= '2023M07'}
    validation_targets, test_targets = dates('2022M08', '2023M02'), dates('2023M08', '2024M07')
    scale = scales(history, countries, '2022M07')
    candidate_predictions, candidate_metrics, validation_fits, validation_predictions = [], [], [], []
    selected, configs = [], {}
    for country in countries:
        local = []
        predictions_by_id = {}
        for config in CANDIDATES:
            t = time.perf_counter()
            try:
                predictions, fit = fit_forecast(history, country, '2022M07', validation_targets, config)
                metrics = aggregate(score(predictions, history, scale))
                predictions_by_id[config['id']] = predictions
                candidate_predictions += predictions
            except Exception as error:
                fit = {'destination': country, 'candidate_id': config['id'], 'converged': False,
                       'error': f'{type(error).__name__}: {error}', 'attempts': [],
                       'fit_seconds': time.perf_counter()-t}
                metrics = aggregate([])
            validation_fits.append(fit)
            row = {'destination': country, 'candidate_id': config['id'],
                   'complexity': sum(config['order'])+sum(config['seasonal_order'][:3]),
                   'converged': fit['converged'], 'fit_seconds': fit['fit_seconds'], **metrics}
            candidate_metrics.append(row)
            local.append(row)
        print(f'Validation candidates complete: {country}', flush=True)
    global_id=select_global_candidate(candidate_metrics,countries)
    global_config=next(c for c in CANDIDATES if c['id']==global_id)
    configs={country:global_config for country in countries}
    selected=[r for r in candidate_metrics if r['candidate_id']==global_id]
    validation_predictions=[r for r in candidate_predictions if r['candidate_id']==global_id]
    print(f'Fixed global SARIMA configuration: {global_id}',flush=True)
    versions = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                'statsmodels': statsmodels.__version__}
    metadata = {'method': 'sarima', 'history_sha256': digest(args.history),
                'implementation_sha256': digest(Path(__file__)), 'shared_metrics_sha256': digest(HERE.parent/'_shared/point_baselines.py'),
                'history_path': str(args.history.resolve()), 'python': sys.version, 'versions': versions,
                'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
                'seed': None, 'generated_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(HERE/'outputs/validation/selected_configs.json', {
        'selection_rule': 'One global configuration: validation macro MASE, then macro MAPE, then candidate ID; all destinations converged.',
        'fit_cutoff': '2022M07', 'validation_targets': validation_targets,
        'candidates': CANDIDATES, 'selected': configs, 'test_data_read_before_freeze': False, **metadata})
    write_json(HERE/'outputs/validation/candidate_fits.json', validation_fits)
    write_csv(HERE/'outputs/validation/candidate_predictions_long.csv', candidate_predictions)
    write_csv(HERE/'metrics/validation/candidate_metrics.csv', candidate_metrics)
    reports = {'validation': save_partition('validation', history, countries, '2022M07', validation_targets,
                validation_predictions, history, {**metadata, 'elapsed_seconds': time.perf_counter()-started})}
    reports['validation']['candidate_report'] = save_candidate_reports('validation', history, countries,
        validation_targets, candidate_predictions, history, validation_fits, metadata)
    final_started = time.perf_counter()
    final_predictions, final_fits = [], []
    for country in countries:
        predictions, fit = fit_forecast(history, country, '2023M07', test_targets, configs[country])
        if not fit['converged']:
            raise RuntimeError(f'Final selected model did not converge: {country}')
        final_predictions += predictions
        final_fits.append(fit)
        print(f'Final refit {len(final_fits)}/20: {country}', flush=True)
    final_predictions.sort(key=lambda r: (r['Date'], countries.index(r['destination'])))
    write_csv(HERE/'outputs/test/predictions_long.csv', final_predictions)
    write_json(HERE/'outputs/test/final_fits.json', final_fits)
    test_candidate_predictions, test_candidate_fits = all_test_candidates(history, countries, test_targets,
                                                                         final_predictions, final_fits)
    write_csv(HERE/'outputs/test/candidate_predictions_long.csv', test_candidate_predictions)
    # The first access to test targets occurs AFTER the entire final forecast is saved.
    test_countries, test = read_data(args.test)
    if test_countries != countries or sorted(test) != test_targets:
        raise ValueError('Unexpected test destinations or months.')
    reports['test'] = save_partition('test', history, countries, '2023M07', test_targets, final_predictions,
        test, {**metadata, 'actual_source': str(args.test.resolve()), 'actual_sha256': digest(args.test),
               'selected_configs_sha256': digest(HERE/'outputs/validation/selected_configs.json'),
               'elapsed_seconds': time.perf_counter()-final_started,
               'test_usage': 'Local exploratory evaluation after final predictions saved; no tuning.'})
    reports['test']['candidate_report'] = save_candidate_reports('test', history, countries,
        test_targets, test_candidate_predictions, test, test_candidate_fits,
        {**metadata, 'actual_sha256': digest(args.test)})
    import importlib.metadata
    packages = ['numpy', 'pandas', 'scipy', 'statsmodels', 'patsy', 'packaging', 'python-dateutil', 'pytz', 'tzdata', 'six']
    (HERE/'requirements-lock.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages)+'\n', encoding='utf-8')
    report(reports, selected, validation_fits, test_candidate_fits, time.perf_counter()-started)
    if digest(notebook) != notebook_hash:
        raise RuntimeError('Submission notebook changed during this run.')
    print(json.dumps({'elapsed_seconds': time.perf_counter()-started,
                      'common_support': {s: r['comparisons'] for s, r in reports.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
