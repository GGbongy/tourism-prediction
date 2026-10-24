"""Deterministic cutoff-safe point baselines; standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_URL = 'https://raw.githubusercontent.com/tulip-lab/open-data/main/ISF-TDF2023/ISF-TDF2023.csv'


def month(label):
    year, number = label.split('M')
    if len(year) != 4 or not 1 <= int(number) <= 12:
        raise ValueError(f'Invalid month: {label}')
    return int(year) * 12 + int(number) - 1


def date(number):
    return f'{number // 12}M{number % 12 + 1:02d}'


def dates(start, end):
    return [date(m) for m in range(month(start), month(end) + 1)]


def read_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or columns[0] != 'Date' or len(columns) != len(set(columns)):
            raise ValueError(f'Invalid header: {path}')
        countries = columns[1:]
        rows = {}
        for row in reader:
            key = row['Date']
            month(key)
            if key in rows:
                raise ValueError(f'Duplicate date: {key}')
            rows[key] = {}
            for country in countries:
                value = float(row[country]) if row[country].strip() else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f'Invalid value: {key}, {country}')
                rows[key][country] = value
    return countries, rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def scales(history, countries, cutoff):
    # Match the starter notebook: drop missing values, then lag-1 differences.
    output = {}
    for country in countries:
        values = [row[country] for d, row in sorted(history.items())
                  if d <= cutoff and row[country] is not None]
        differences = [abs(b-a) for a, b in zip(values, values[1:])]
        output[country] = {'destination': country, 'fit_cutoff': cutoff,
                           'history_valid_n': len(values), 'difference_n': len(differences),
                           'mase_denominator': mean(differences)}
    return output


def forecast(history, countries, cutoff, targets, lag):
    if lag not in (1, 12):
        raise ValueError('Expected lag 1 or 12.')
    if targets != dates(date(month(cutoff)+1), targets[-1]):
        raise ValueError('Forecast targets must be contiguous and immediately follow cutoff.')
    values = {d: row.copy() for d, row in history.items() if d <= cutoff}
    roots = {}
    result = []
    for horizon, target in enumerate(targets, 1):
        source = date(month(target)-lag)
        if source not in values:
            raise ValueError(f'Missing lag source {source}')
        prediction = values[source].copy()
        values[target] = prediction
        roots[target] = roots.get(source, source)
        for country in countries:
            result.append({'Date': target, 'destination': country, 'horizon': horizon,
                           'fit_cutoff': cutoff, 'lag': lag, 'source_month': source,
                           'observed_source_month': roots[target], 'point': prediction[country]})
    return result


def score(predictions, actuals, scale):
    result = []
    for prediction in predictions:
        country = prediction['destination']
        actual = actuals.get(prediction['Date'], {}).get(country)
        denominator = scale[country]['mase_denominator']
        error = abs(actual-prediction['point']) if actual is not None and prediction['point'] is not None else None
        result.append({**prediction, 'actual': actual, 'absolute_error': error,
                       'mase_denominator': denominator,
                       'scaled_absolute_error': error/denominator
                       if error is not None and denominator is not None and denominator > 0 else None,
                       'absolute_percentage_error': error/abs(actual)*100
                       if error is not None and actual != 0 else None})
    return result


def aggregate(rows):
    return {'n_predictions': len(rows),
            'n_actual': sum(r['actual'] is not None for r in rows),
            'n_scored': sum(r['absolute_error'] is not None for r in rows),
            'n_missing_prediction': sum(r['point'] is None for r in rows),
            'n_missing_actual': sum(r['actual'] is None for r in rows),
            'mase_valid_n': sum(r['scaled_absolute_error'] is not None for r in rows),
            'mape_valid_n': sum(r['absolute_percentage_error'] is not None for r in rows),
            'mae': mean([r['absolute_error'] for r in rows]),
            'mase': mean([r['scaled_absolute_error'] for r in rows]),
            'mape_percent': mean([r['absolute_percentage_error'] for r in rows])}


def tables(rows, countries):
    by_destination = [{'destination': c, **aggregate([r for r in rows if r['destination']==c])}
                      for c in countries]
    by_horizon = [{'horizon': h, 'Date': next(r['Date'] for r in rows if r['horizon']==h),
                   **aggregate([r for r in rows if r['horizon']==h])}
                  for h in sorted({r['horizon'] for r in rows})]
    micro = {'aggregation': 'micro', **aggregate(rows)}
    macro = {**micro, 'aggregation': 'macro_destination',
             **{key: mean([r[key] for r in by_destination])
                for key in ('mae', 'mase', 'mape_percent')}}
    return [micro, macro], by_destination, by_horizon


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows, columns):
    def fmt(v):
        return '—' if v is None else (f'{v:,.4f}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                       '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                      ['| ' + ' | '.join(fmt(r[c]) for c in columns) + ' |' for r in rows])


def run(lag, method_dir):
    parser = argparse.ArgumentParser(description=f'Run recursive lag-{lag} baseline.')
    parser.add_argument('--history', type=Path, default=ROOT/'experiments/data/ISF-TDF2023.csv')
    parser.add_argument('--test', type=Path, default=ROOT/'testset.csv')
    args = parser.parse_args()
    started = time.perf_counter()
    countries, history = read_data(args.history)
    if len(countries) != 20:
        raise ValueError('Expected 20 destinations.')
    test_countries, test = read_data(args.test)
    if countries != test_countries or sorted(test) != dates('2023M08', '2024M07'):
        raise ValueError('Unexpected test columns or months.')
    # Never fit on test actuals; future public placeholder rows are also excluded.
    history = {d: row for d, row in history.items() if d <= '2023M07'}
    reports = {}
    for split, cutoff, targets, actuals in [
        ('validation', '2022M07', dates('2022M08', '2023M02'), history),
        ('test', '2023M07', dates('2023M08', '2024M07'), test),
    ]:
        scale = scales(history, countries, cutoff)
        predictions = forecast(history, countries, cutoff, targets, lag)
        scored = score(predictions, actuals, scale)
        output = method_dir/'outputs'/split
        metrics = method_dir/'metrics'/split
        write_csv(output/'predictions_long.csv', predictions)
        write_csv(output/'predictions_wide.csv', [
            {'Date': d, **{r['destination']: r['point'] for r in predictions if r['Date']==d}}
            for d in targets])
        write_csv(output/'scored_predictions.csv', scored)
        write_csv(output/'mase_scales.csv', list(scale.values()))
        summary, by_destination, by_horizon = tables(scored, countries)
        for filename, data in [('summary.csv', summary), ('by_destination.csv', by_destination),
                               ('by_horizon.csv', by_horizon)]:
            write_csv(metrics/filename, data)
        # Both baselines are compared on the intersection of finite predictions.
        other = forecast(history, countries, cutoff, targets, 12 if lag == 1 else 1)
        other_valid = {(r['Date'], r['destination']) for r in other if r['point'] is not None}
        common = [r for r in scored if r['absolute_error'] is not None
                  and (r['Date'], r['destination']) in other_valid]
        common_tables = tables(common, countries)
        for filename, data in zip(('summary.csv', 'by_destination.csv', 'by_horizon.csv'), common_tables):
            write_csv(metrics/'common_support'/filename, data)
        write_csv(output/'common_support_scored_predictions.csv', common)
        reports[split] = (summary, by_destination, by_horizon, common_tables[0])
        metadata = {'method': f'lag-{lag}', 'split': split, 'fit_cutoff': cutoff,
                    'target_months': targets, 'destinations': countries,
                    'history_path': str(args.history.resolve()),
                    'history_sha256': hashlib.sha256(args.history.read_bytes()).hexdigest(),
                    'actual_source': str((args.test if split=='test' else args.history).resolve()),
                    'actual_sha256': hashlib.sha256((args.test if split=='test' else args.history).read_bytes()).hexdigest(),
                    'test_usage': 'Local exploratory evaluation only; not used in predictions or scale fitting.' if split=='test' else 'None in this partition.',
                    'python': sys.version, 'platform': platform.platform(), 'seed': None,
                    'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'elapsed_seconds_since_start': time.perf_counter()-started,
                    'audit': 'Not scored; no frozen selection protocol yet.'}
        (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    description = ('所有预测步均使用截止月份的最后一个观测值；后续步递归使用自身预测，不读取后续真实值。'
                   if lag==1 else
                   '每个月使用前 12 个月对应月份的观测值；超过 12 步时递归使用自身预测。本次最长预测 12 步。')
    content = [f'# lag-{lag} 点预测基线', '', description, '',
               f'运行：`python experiments/lag-{lag}/run.py`。仅依赖 Python 标准库，无随机过程。', '',
               '## 评估口径', '',
               '- 验证：训练截止 2022M07，一次性预测 2022M08–2023M02，共 7 步、20 个目的地。',
               '- 测试：历史截止 2023M07，一次性预测 2023M08–2024M07，共 12 步、20 个目的地。只在预测生成后对照本地测试真实值评分，属于探索性结果。',
               '- 两个时段的预测、逐条误差、缩放分母和指标分别存放，不混用。预测期间不使用真实值滚动更新。',
               '- 锁定审核期尚未评分，待固定方案后单独运行；最终拟合允许使用截至 2023M07 的历史。',
               '- MAE 为绝对误差均值；MAPE 为百分比，排除零真实值；MASE 使用各目的地拟合截止时点之前历史的 lag-1 绝对差均值缩放，与 starter notebook 一致。',
               '- MASE 历史处理与 notebook 一致：先去除缺失再求差，可能跨越历史缺失月份；分母为零或不可用时不计算该项。',
               '- 缺失真实值不补零、不参与指标；保留所有预测。测试包含 13 个完整目的地及 Thailand 的 8 个已知月份，其他 6 个目的地无测试真实值。',
               '- lag-12 验证来源中 Chile 的 2021M08、2021M09 缺失，对应两个预测留空，不做补值；lag-1 验证有效 140 条，lag-12 有效 138 条。',
               '- `common_support/` 只使用两种基线均有预测且真实值存在的样本：验证 138 条，测试 164 条；比较两种方法时优先使用该口径。n_actual 是真实值数，n_scored 才是有效预测配对数。',
               '- micro 对有效目的地—月份样本等权；macro_destination 先对每个目的地求指标，再对有有效指标的目的地等权平均。样本数为底层有效配对计数。',
               '- 验证与测试的真实值覆盖、预测长度和 MASE 缩放历史不同，不能将两组数值视为同条件比较。', '',
               '## 文件组织', '',
               '| 路径 | 内容 |', '| --- | --- |',
               '| `outputs/validation/`、`outputs/test/` | 分时段保存下列原始计算结果。 |',
               '| `predictions_long.csv` | 日期、目的地、预测步长、截止月份、lag 来源月份、原始观测来源月份、点预测。 |',
               '| `predictions_wide.csv` | Date 加 20 个目的地列，便于可视化读取。 |',
               '| `scored_predictions.csv` | 预测、真实值、绝对误差、缩放绝对误差、绝对百分比误差；缺失项留空。 |',
               '| `mase_scales.csv` | 各目的地的 MASE 分母、有效历史数与差分数。 |',
               '| `run_metadata.json` | 该时段的数据来源、文件哈希、环境、耗时及评估用途。 |',
               '| `metrics/validation/`、`metrics/test/` | 分时段保存 summary.csv、by_destination.csv、by_horizon.csv。 |',
               '| `metrics/<时段>/common_support/` | 两种基线共同有效样本上的汇总、按目的地与按预测步长指标。 |',
               '| `outputs/<时段>/common_support_scored_predictions.csv` | 共同有效样本对应的逐条预测与误差。 |',
               '| `README.md` | 由运行脚本同步生成的指标报告。 |', '',
               '原始历史数据来自 [TULIP Lab ISF-TDF2023](https://github.com/tulip-lab/open-data/tree/main/ISF-TDF2023)，缓存于 `../data/ISF-TDF2023.csv`；测试数据为根目录 `testset.csv`。', '',
               '共享实现位于 `../_shared/point_baselines.py`。本地实验未修改提交 notebook。代码由 Codex 辅助编写，并通过递归预测和时间截止检查。']
    for split, heading in [('validation', '验证集'), ('test', '测试集（本地探索性评估）')]:
        summary, by_destination, by_horizon, common_summary = reports[split]
        content += ['', f'## {heading}', '', f'指标文件：`metrics/{split}/`；原始结果：`outputs/{split}/`。', '',
                    '### 汇总（各方法自身有效样本）', '', md_table(summary, ['aggregation', 'n_predictions', 'n_actual', 'n_scored', 'mase_valid_n', 'mape_valid_n', 'mae', 'mase', 'mape_percent']), '',
                    '### 汇总（两种基线共同有效样本）', '', md_table(common_summary, ['aggregation', 'n_scored', 'mae', 'mase', 'mape_percent']), '',
                    '### 按目的地（各方法自身有效样本）', '', md_table(by_destination, ['destination', 'n_actual', 'n_scored', 'mape_valid_n', 'mae', 'mase', 'mape_percent']), '',
                    '### 按预测步长（各方法自身有效样本）', '', md_table(by_horizon, ['horizon', 'Date', 'n_actual', 'n_scored', 'mape_valid_n', 'mae', 'mase', 'mape_percent'])]
        worst_destinations = sorted([r for r in by_destination if r['mase'] is not None],
                                    key=lambda r: r['mase'], reverse=True)[:3]
        worst_horizon = max((r for r in by_horizon if r['mase'] is not None), key=lambda r: r['mase'])
        content += ['', '### 失败案例与局限', '',
                    '按 MASE 较差的三个目的地：' + '、'.join(
                        f'{r["destination"]}（{r["mase"]:.4f}）' for r in worst_destinations) + '。', '',
                    f'按 MASE 最差的预测步长为 h={worst_horizon["horizon"]}（{worst_horizon["Date"]}，MASE={worst_horizon["mase"]:.4f}）。', '',
                    '该方法确定且无待调超参数；相同输入重复运行预测一致。它不显式建模趋势变化或结构突变，不能把这些基线结果当作未来表现保证；本次仅实现点预测，未构造预测区间。']
    (method_dir/'README.md').write_text('\n'.join(content)+'\n', encoding='utf-8')
    print(json.dumps({'method': f'lag-{lag}', 'summary': {s: r[0] for s, r in reports.items()}}, indent=2))
