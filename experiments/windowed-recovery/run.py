"""Evaluate fixed-configuration, recent-window recovery extrapolation methods."""
from __future__ import annotations

import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for deps in (HERE/'.deps', HERE.parent/'.deps', HERE.parent/'sarima/.deps'):
    if deps.is_dir():
        sys.path.insert(0, str(deps))
        break
sys.path.insert(0, str(HERE.parent/'_shared'))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from point_baselines import aggregate, dates, date, month, scales, score, tables, write_csv, md_table, mean, forecast
from experiment_io import csv_read, json_write
from models import CONFIGS, predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(rows, columns):
    return md_table(rows, columns)


def prepare(history, baseline, countries, cutoff):
    labels = dates('2020M01', cutoff)
    z = np.empty((len(labels), len(countries)))
    for j,c in enumerate(countries):
        for t,d in enumerate(labels):
            value = history.get(d,{}).get(c)
            z[t,j] = np.log(value+1)-np.log(baseline[(c,d)]+1) if value is not None else np.nan
    return labels,z


def generate(history, baseline, countries, cutoff, targets):
    labels,z = prepare(history,baseline,countries,cutoff)
    rows,logs = [],[]
    for j,c in enumerate(countries):
        for config in CONFIGS:
            started = time.perf_counter()
            r,info = predict(z[:,j],len(targets),config)
            logs.append({'destination':c, **config, 'fit_cutoff':cutoff,
                'fit_start':labels[-config['window']] if config['window'] and config['window']<=len(labels) else labels[0],
                'fit_seconds':time.perf_counter()-started, **info})
            for h,d in enumerate(targets,1):
                b = baseline[(c,d)]
                point = max(0.,(b+1)*float(r[h-1])-1)
                if not math.isfinite(point):
                    raise ValueError(f'Nonfinite point: {config["candidate_id"]}, {c}, {d}')
                rows.append({'Date':d,'destination':c,'horizon':h,'fit_cutoff':cutoff,
                    'candidate_id':config['candidate_id'],'family':config['family'],
                    'window':config['window'] if config['window'] is not None else 'all',
                    'phi':config['phi'],'baseline':b,'r_hat':float(r[h-1]),'point':point})
        if (j+1)%5==0:
            print(f'{cutoff}: {j+1}/{len(countries)} destinations, {len(CONFIGS)} configurations',flush=True)
    return rows,logs


def extended(rows):
    out=aggregate(rows)
    paired=[r for r in rows if r['actual'] is not None and r['point'] is not None]
    signed=[r['point']-r['actual'] for r in paired]
    out.update({'mean_signed_error':mean(signed),
                'underforecast_fraction':sum(v<0 for v in signed)/len(signed) if signed else None})
    return out


def bias_fields(rows,countries,aggregation):
    if aggregation=='micro':
        result=extended(rows)
        return {k:result[k] for k in ('mean_signed_error','underforecast_fraction')}
    groups=[extended([r for r in rows if r['destination']==c]) for c in countries]
    return {k:mean([r[k] for r in groups]) for k in ('mean_signed_error','underforecast_fraction')}


def candidate_tables(scored,countries):
    summaries,destinations,horizons=[],[],[]
    for config in CONFIGS:
        subset=[r for r in scored if r['candidate_id']==config['candidate_id']]
        summary,dest,horizon=tables(subset,countries)
        annotations={'candidate_id':config['candidate_id'],'family':config['family'],
                     'window':config['window'] if config['window'] is not None else 'all','phi':config['phi']}
        summaries += [{**annotations,**r,**bias_fields(subset,countries,r['aggregation'])} for r in summary]
        destinations += [{**annotations,**r} for r in dest]
        horizons += [{**annotations,**r} for r in horizon]
    return summaries,destinations,horizons


def save_candidates(split,history,actual,countries,cutoff,rows,logs):
    scored=score(rows,actual,scales(history,countries,cutoff))
    output,metrics=HERE/'outputs'/split,HERE/'metrics'/split
    write_csv(output/'candidate_predictions_long.csv',rows)
    write_csv(output/'candidate_scored_predictions.csv',scored)
    json_write(output/'candidate_fits.json',logs)
    write_csv(output/'mase_scales.csv',list(scales(history,countries,cutoff).values()))
    summary,dest,horizon=candidate_tables(scored,countries)
    for filename,data in [('candidate_summary.csv',summary),('candidate_metrics.csv',dest),('candidate_by_horizon.csv',horizon)]:
        write_csv(metrics/filename,data)
    # Fixed-form comparisons isolate the window effect instead of reselecting a method.
    ablation=[r for r in summary if (r['family'] in ('mean_r','median_r','weighted_mean_r','ar1_z')
              or (r['family'] in ('linear_r','median_slope_r','linear_z') and r['phi']==.95))]
    write_csv(metrics/'window_ablation.csv',ablation)
    return {'scored':scored,'summary':summary,'dest':dest,'horizon':horizon}


def choose(validation,countries):
    ranked=sorted([r for r in validation['summary'] if r['aggregation']=='macro_destination' and r['n_scored']==140],
                  key=lambda r:(r['mase'],r['mape_percent'],r['candidate_id']))
    if not ranked:
        raise ValueError('No full-coverage validation candidate')
    global_id=ranked[0]['candidate_id']
    family={}
    window={}
    for r in ranked:
        family.setdefault(r['family'],r['candidate_id'])
        window.setdefault(str(r['window']),r['candidate_id'])
    return {'global':global_id,'by_family':family,'by_window':window}


def save_selections(split,data,selected,countries,history,cutoff,actual):
    scored=data['scored']
    for name in ('global',):
        subset=[r for r in scored if r['candidate_id']==selected['global']]
        summary,dest,horizon=tables(subset,countries)
        for filename,rows in [('summary.csv',summary),('by_destination.csv',dest),('by_horizon.csv',horizon)]:
            write_csv(HERE/f'metrics/{split}/selected_{name}'/filename,rows)
        write_csv(HERE/f'outputs/{split}/selected_{name}_scored_predictions.csv',subset)
    for grouping in ('family','window'):
        rows=[]
        for group,candidate in selected['by_'+grouping].items():
            rows += [{'group':group,**r} for r in data['summary'] if r['candidate_id']==candidate]
        write_csv(HERE/f'metrics/{split}/validation_selected_by_{grouping}.csv',rows)
    pool={}
    for name in ('global',):
        pool['window_'+name]=[r for r in scored if r['candidate_id']==selected['global']]
    fixed_sarima=ROOT/f'experiments/sarima/fixed_global/{split}/outputs/predictions_long.csv'
    if fixed_sarima.exists():
        pool['SARIMA_fixed']=score(csv_read(fixed_sarima),actual,scales(history,countries,cutoff))
    for lag in (1,12):
        targets=sorted({r['Date'] for r in scored})
        pool[f'lag-{lag}']=score(forecast(history,countries,cutoff,targets,lag),actual,scales(history,countries,cutoff))
    keys=set.intersection(*[{(r['Date'],r['destination']) for r in rows if r['absolute_error'] is not None} for rows in pool.values()])
    comparison=[]
    for name,rows in pool.items():
        subset=[r for r in rows if (r['Date'],r['destination']) in keys]
        summary,_,_=tables(subset,countries)
        comparison += [{'method':name,**r,**bias_fields(subset,countries,r['aggregation'])} for r in summary]
    write_csv(HERE/f'metrics/{split}/comparison_common_support.csv',comparison)
    return comparison


def training_cv(history,baseline,countries):
    combined,logs,metrics=[],[],[]
    for cutoff,end in [('2020M12','2021M12'),('2021M07','2022M07'),('2021M12','2022M07')]:
        targets=dates(date(month(cutoff)+1),end)
        rows,fit=generate(history,baseline,countries,cutoff,targets)
        scored=score(rows,history,scales(history,countries,cutoff))
        combined+=scored;logs+=fit
        summary,_,_=candidate_tables(scored,countries)
        metrics += [{'fold_cutoff':cutoff,**r} for r in summary]
    write_csv(HERE/'outputs/training_cv/candidate_scored_predictions.csv',combined)
    json_write(HERE/'outputs/training_cv/candidate_fits.json',logs)
    write_csv(HERE/'metrics/training_cv/fold_summary.csv',metrics)
    # Scaling denominators are specific to each original forecast origin.
    summary,_,_=candidate_tables(combined,countries)
    write_csv(HERE/'metrics/training_cv/candidate_summary.csv',summary)


def report(data,selected,comparisons,elapsed):
    lookup={split:{r['candidate_id']:r for r in result['summary'] if r['aggregation']=='micro'}
            for split,result in data.items()}
    test_cases=[('完整恢复期均值','mean_r_nall_p1'),('最近 3 个月均值','mean_r_n3_p1'),
                ('最近 3 个月中位增量，phi=.95','median_slope_r_n3_p0.95'),
                ('最近 6 个月中位增量，phi=.95','median_slope_r_n6_p0.95'),
                ('最近 9 个月中位增量，phi=.95','median_slope_r_n9_p0.95'),
                ('最近 12 个月中位增量，phi=.95','median_slope_r_n12_p0.95'),
                ('最近 18 个月中位增量，phi=.95','median_slope_r_n18_p0.95'),
                ('完整恢复期中位增量，phi=.95','median_slope_r_nall_p0.95'),
                ('仅保持最近水平','last_r_n1')]
    content=['# Windowed recovery：近期窗口恢复指数实验','',
        '使用已冻结的疫情前反事实基线，比较单目的地恢复指数的近期窗口外推。所有目的地使用统一的恢复模型与超参数；Notebook 不修改。','',
        '## 结论先读','',
        '缩短窗口显著改善了“全历史平均水平”的旧低谷拖累，但不是窗口越短越好。3–6 个月的斜率容易受最近几次跳升影响；在当前本地测试诊断中，12–18 个月的中位月增量配合温和阻尼更值得继续研究，普通线性和对数线性外推存在过冲。', '',
        '保持最后一个恢复水平也是重要对照：其测试表现已经优于第一版的混合选型。全历史中位增量也有较好结果，说明问题不仅是历史长度，还在于模型如何使用旧数据以及如何估计当前水平和增量。', '',
        f'严格按本轮验证规则选出的全局配置是 {selected["global"]}，它在测试期仍出现较大过估；不再按目的地分别挑选模型。不能把下面测试诊断中的优胜参数写成由验证集选出的参数。验证期仅七个月、恢复阶段也与最终测试不同。', '',
        '### 验证集：固定模型下的窗口效应（全部 140 条）','',
        table([{'comparison':name,**lookup['validation'][candidate]} for name,candidate in test_cases],
              ['comparison','n_scored','mae','mase','mape_percent']), '',
        '### 测试集：同一组对照（164 条，仅作本地诊断）','',
        table([{'comparison':name,**lookup['test'][candidate]} for name,candidate in test_cases],
              ['comparison','n_scored','mae','mase','mape_percent','underforecast_fraction']), '',
        '下一步建议收窄到末值对照和 n=12、18 的中位增量模型（小范围阻尼），优先检验恢复阶段的稳定性，不继续扩大复杂模型或逐目的地大范围调参。当前数据不足以把测试诊断里的 n=18 宣称为未见测试时的最优窗口。', '',
        '## 复现','', '```powershell',
        'python -m pip install --target experiments/windowed-recovery/.deps -r experiments/windowed-recovery/requirements.txt',
        'python experiments/windowed-recovery/run.py','```','',
        '可以复用 SARIMA 已安装的依赖，实际版本见 requirements-lock.txt。','',
        '## 实验设计','',
        '- 共 73 个预先固定的配置。窗口 n=3、6、9、12、18 个日历月；all 为截至起点的完整恢复期。另有 n=1 的末值对照。','- 每条预测只依赖起点之前窗口内的恢复观测，窗口内缺失不补零，也不向更早月份扩展凑够 n 条。反事实基线仍使用既有疫情前历史；“窗口限制”专指恢复模型。',
        '- 预测起点固定：验证 T=2022M07，预测随后 7 个月；测试 T=2023M07，预测随后 12 个月。最终预测一次性生成，不读取测试月份真实值滚动更新。',
        '- 近期均值、中位数、指数加权均值（固定半衰期 3 个月）在 r=exp(z) 上计算。','- 线性 r 趋势按实际月份位置做 OLS，末端水平为回归线在起点处的值；未来增量按 phi=0.8、0.95、1.0 累积，phi=1 为不阻尼。',
        '- 稳健趋势用相邻有效观测间的每月增量中位数，phi=0.95 或 1.0；缺失间隔使用真实月距离。对数线性趋势在 z 上拟合，再取 exp，同样比较 phi=0.95 或 1.0。',
        '- 原有 ETS 水平、阻尼 ETS、平稳 AR(1)、无漂移 ARIMA(0,1,1) 在 n=12、18、all 上作对照。它们的不足 12 条有效观测等回退规则沿用第一版，逐条记录，不把回退误报为成功拟合。',
        '- 无观测时固定回退 r=1；趋势不足 3 条时回退窗口内最近水平；数值外推非有限时也回退窗口内最近水平并记录。不把 r 的上界强行设为 1；线性负预测截为 0。','- point=max(0,(baseline+1)*r_hat-1)，所有配置使用相同基线和同一时段的 MASE 分母。',
        '- 全局配置依据验证集 macro_destination MASE 最小选择，依次以 MAPE 和 ID 打破并列；所有目的地使用同一配置，不进行逐目的地选型。配置在测试拟合与评分前固定。',
        '- 所有“按窗口最佳”“按方法最佳”均指验证集最佳，测试时沿用相同 ID；没有根据测试成绩重新挑最佳。','- 原训练期的三个滚动起点用于稳定性诊断，目标不晚于 2022M07，不决定本轮最终配置。官方验证期只有 7 个月，无法充分检验 12 步预测；大量候选也带来选型乐观性。',
        '- 测试真值已在此前实验中查看，本轮只是本地探索性比较，不宣称盲测。锁定审核期不调参，2023M07 前历史只用于最终重拟合。',
        '- MAE、MASE、MAPE 与之前方法共用函数；MAPE 排除零真值，缺失不评分。另存平均有符号误差和低估比例，负的有符号误差表示整体偏低。','',
        '## 验证集固定的配置','',f'全局：`{selected["global"]}`。','',
        '## 文件组织','', '| 路径 | 内容 |','| --- | --- |',
        '| outputs/baseline/ | 从第一版 MA 冻结复制的反事实基线及来源哈希。 |',
        '| outputs/protocol.json、outputs/selected_configs.json | 搜索规则及在测试前固定的配置。 |',
        '| outputs/validation/、outputs/test/ | 全部候选的预测、r_hat、baseline、逐条误差、拟合日志和 MASE 分母；含全局统一选型的结果。 |',
        '| metrics/validation/、metrics/test/ | 各配置汇总、按目的地和步长的全部指标，窗口/方法比较，以及基线比较。两个时段分文件存放。 |',
        '| metrics/<时段>/window_ablation.csv | 固定方法与阻尼后逐个窗口比较，避免把窗口变化与换方法混淆。 |',
        '| outputs/training_cv/、metrics/training_cv/ | 早期滚动稳定性诊断，与正式验证、测试分开。 |',
        '| outputs/run_manifest.json | 数据、基线、代码哈希，环境、耗时和原文件未修改检查。 |','',
        '## 参数完整清单','',table([{'candidate_id':c['candidate_id'],'family':c['family'],'window':c['window'] or 'all','phi':c['phi']} for c in CONFIGS],['candidate_id','family','window','phi'])]
    for split,title in [('validation','验证集'),('test','测试集：本地探索性')]:
        content += ['',f'## {title}','','### 与固定 SARIMA 及基线比较（共同有效样本）','',
            table(comparisons[split],['method','aggregation','n_scored','mae','mase','mape_percent','mean_signed_error','underforecast_fraction']),
            '','### 每个窗口的验证选中配置（本时段表现）','',
            table(csv_read(HERE/f'metrics/{split}/validation_selected_by_window.csv'),['group','candidate_id','aggregation','n_scored','mae','mase','mape_percent']),
            '','### 每类方法的验证选中配置（本时段表现）','',
            table(csv_read(HERE/f'metrics/{split}/validation_selected_by_family.csv'),['group','candidate_id','aggregation','mae','mase','mape_percent']),
            '','### 全部参数：汇总','',table(data[split]['summary'],['candidate_id','aggregation','n_scored','mae','mase','mape_percent','underforecast_fraction']),
            '','### 全部参数：按目的地','',table(data[split]['dest'],['candidate_id','destination','n_scored','mae','mase','mape_percent']),
            '','### 全部参数：按预测步长','',table(data[split]['horizon'],['candidate_id','horizon','Date','n_scored','mae','mase','mape_percent'])]
    content += ['', '## 稳定性诊断：原训练期滚动验证','',
        '该时期仍以疫情低谷为主，与后续恢复阶段存在分布差异；用于说明窗口表现是否稳定，不替代规定验证集。全部逐折配置指标见 metrics/training_cv/fold_summary.csv。','',
        table(csv_read(HERE/'metrics/training_cv/candidate_summary.csv'),['candidate_id','aggregation','n_scored','mae','mase','mape_percent']),
        '',f'本次全流程约 {elapsed:.2f} 秒，无随机采样、无外部变量。','',
        '原始方案依据：design.md。Codex 辅助实施与数值核查。']
    (HERE/'README.md').write_text('\n'.join(content)+'\n',encoding='utf-8')


def main():
    from point_baselines import read_data
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,default=ROOT/'experiments/data/ISF-TDF2023.csv')
    parser.add_argument('--test',type=Path,default=ROOT/'testset.csv')
    args=parser.parse_args()
    started=time.perf_counter()
    protected=[ROOT/'Tourism-Forecasting-Starter.ipynb']
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    countries,history=read_data(args.history)
    history={d:r for d,r in history.items() if d<='2023M07'}
    target=HERE/'outputs/baseline/baseline_long.csv'
    source_info=target.parent/'source.json'
    if source_info.exists():
        cached=json.loads(source_info.read_text(encoding='utf-8'))
        if cached.get('history_sha256') not in (None,digest(args.history)):
            raise ValueError('History differs from frozen baseline source')
    if not target.exists():
        from counterfactual_baseline import baseline_bundle
        baseline_bundle(history,countries,HERE)
    baseline={(r['destination'],r['Date']):r['baseline'] for r in csv_read(target)}
    json_write(source_info,{'history_sha256':digest(args.history),'sha256':digest(target),
               'source':'Frozen pre-pandemic fits and cross-validation retained in this method directory',
               'refitted':False})
    protocol={'configs':CONFIGS,'selection':'validation macro MASE, then MAPE, then candidate ID',
              'window_is_calendar_months':True,'test_updates_from_actuals':False,
              'baseline_sha256':digest(target),'history_sha256':digest(args.history)}
    json_write(HERE/'outputs/protocol.json',protocol)
    training_cv(history,baseline,countries)
    rows,logs=generate(history,baseline,countries,'2022M07',dates('2022M08','2023M02'))
    data={'validation':save_candidates('validation',history,history,countries,'2022M07',rows,logs)}
    selected=choose(data['validation'],countries)
    json_write(HERE/'outputs/selected_configs.json',{'selected':selected,'test_actuals_read_before_selection':False,
               'protocol_sha256':digest(HERE/'outputs/protocol.json'),'selected_at_utc':datetime.now(timezone.utc).isoformat()})
    rows,logs=generate(history,baseline,countries,'2023M07',dates('2023M08','2024M07'))
    write_csv(HERE/'outputs/test/candidate_predictions_long.csv',rows)
    test_countries,test=read_data(args.test)
    if test_countries!=countries or sorted(test)!=dates('2023M08','2024M07'):
        raise ValueError('Unexpected test schema')
    data['test']=save_candidates('test',history,test,countries,'2023M07',rows,logs)
    comparisons={}
    for split,cutoff,actual in [('validation','2022M07',history),('test','2023M07',test)]:
        comparisons[split]=save_selections(split,data[split],selected,countries,history,cutoff,actual)
    after={str(p.relative_to(ROOT)):digest(p) for p in protected}
    if before!=after:
        raise RuntimeError('Protected original result changed')
    elapsed=time.perf_counter()-started
    versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','statsmodels')}
    json_write(HERE/'outputs/run_manifest.json',{'history_sha256':digest(args.history),'test_sha256':digest(args.test),
        'baseline_sha256':digest(target),'models_sha256':digest(HERE/'models.py'),'runner_sha256':digest(Path(__file__)),
        'versions':versions,'python':sys.version,'platform':platform.platform(),'elapsed_seconds':elapsed,
        'original_files_before':before,'original_files_after':after,'candidate_count':len(CONFIGS)})
    (HERE/'requirements-lock.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in
        ('numpy','pandas','scipy','statsmodels','patsy','packaging','python-dateutil','pytz','tzdata','six'))+'\n',encoding='utf-8')
    report(data,selected,comparisons,elapsed)
    print(json.dumps({'selected':selected,'comparisons':comparisons,'elapsed_seconds':elapsed},indent=2),flush=True)


if __name__=='__main__':
    main()
