"""Rebuild retained fixed-configuration metrics and the experiment index from saved predictions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/sarima/.deps'))
sys.path.insert(0,str(ROOT/'experiments/_shared'))

import hashlib
import json
from point_baselines import read_data,score,scales,tables,write_csv,md_table
from experiment_io import csv_read,json_write

HERE=Path(__file__).resolve().parent
RECOVERY=[
    ('recovery_linear_12','linear_r_n12_p1','12 个月线性恢复趋势，φ=1'),
    ('recovery_median_12','median_slope_r_n12_p0.95','12 个月中位月增量，φ=0.95'),
    ('recovery_median_18','median_slope_r_n18_p0.95','18 个月中位月增量，φ=0.95')]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save_result(folder,rows,actual,history,countries,cutoff):
    rows=sorted(rows,key=lambda r:(r['Date'],countries.index(r['destination'])))
    scored=score(rows,actual,scales(history,countries,cutoff))
    write_csv(folder/'outputs/predictions_long.csv',rows)
    write_csv(folder/'outputs/scored_predictions.csv',scored)
    write_csv(folder/'outputs/predictions_wide.csv',[{'Date':d,**{r['destination']:r['point'] for r in rows if r['Date']==d}}
                                                   for d in sorted({r['Date'] for r in rows})])
    for filename,data in zip(('summary.csv','by_destination.csv','by_horizon.csv'),tables(scored,countries)):
        write_csv(folder/'metrics'/filename,data)
    return scored


def main():
    countries,history=read_data(ROOT/'experiments/data/ISF-TDF2023.csv')
    _,test=read_data(ROOT/'testset.csv')
    sarima=ROOT/'experiments/sarima'
    candidates=csv_read(sarima/'metrics/validation/candidate_summary.csv')
    eligible=[r for r in candidates if r['aggregation']=='macro_destination' and r['converged_destinations']==20 and r['n_scored']==140]
    chosen=min(eligible,key=lambda r:(r['mase'],r['mape_percent'],r['candidate_id']))['candidate_id']
    grid=json.loads((sarima/'outputs/validation/selected_configs.json').read_text(encoding='utf-8'))['candidates']
    config=next(r for r in grid if r['id']==chosen)
    json_write(sarima/'outputs/fixed_global_config.json',{'candidate_id':chosen,'config':config,
        'selection_rule':'One global configuration: validation macro MASE, then MAPE, then candidate ID',
        'validation_candidate_summary_sha256':digest(sarima/'metrics/validation/candidate_summary.csv'),
        'test_results_used_for_selection':False})
    metadata={'models_refitted':False,'sarima_fixed_config':config,'source_hashes':{},'methods':[]}
    all_comparisons={}
    plot_sources={}
    for split,cutoff,actual in [('validation','2022M07',history),('test','2023M07',test)]:
        pool={};folders={}
        window=ROOT/f'experiments/windowed-recovery/outputs/{split}/candidate_predictions_long.csv'
        all_rows=csv_read(window)
        metadata['source_hashes'][str(window.relative_to(ROOT))]=digest(window)
        for method,candidate,label in RECOVERY:
            subset=[r for r in all_rows if r['candidate_id']==candidate]
            folder=ROOT/f'experiments/windowed-recovery/retained/{method}/{split}'
            folders[method]=folder
            pool[method]=save_result(folder,subset,actual,history,countries,cutoff)
            plot_sources.setdefault(method,{})[split]=str((folder/'outputs/predictions_long.csv').relative_to(ROOT))
        source=sarima/f'outputs/{split}/candidate_predictions_long.csv'
        metadata['source_hashes'][str(source.relative_to(ROOT))]=digest(source)
        subset=[r for r in csv_read(source) if r['candidate_id']==chosen]
        folder=sarima/f'fixed_global/{split}'
        folders['sarima_fixed']=folder
        pool['sarima_fixed']=save_result(folder,subset,actual,history,countries,cutoff)
        plot_sources.setdefault('sarima_fixed',{})[split]=str((folder/'outputs/predictions_long.csv').relative_to(ROOT))
        for lag in (1,12):
            path=ROOT/f'experiments/lag-{lag}/outputs/{split}/predictions_long.csv'
            metadata['source_hashes'][str(path.relative_to(ROOT))]=digest(path)
            pool[f'lag-{lag}']=score(csv_read(path),actual,scales(history,countries,cutoff))
        support=set.intersection(*[{(r['Date'],r['destination']) for r in rows if r['absolute_error'] is not None}
                                  for rows in pool.values()])
        comparisons=[]
        for method,rows in pool.items():
            subset=[r for r in rows if (r['Date'],r['destination']) in support]
            summary,dest,horizon=tables(subset,countries)
            comparisons += [{'method':method,**r} for r in summary]
            if method in folders:
                for filename,data in [('common_support_summary.csv',summary),('common_support_by_destination.csv',dest),
                                      ('common_support_by_horizon.csv',horizon)]:
                    write_csv(folders[method]/'metrics'/filename,data)
        write_csv(HERE/f'metrics/{split}.csv',comparisons)
        all_comparisons[split]=comparisons
    labels={method:label for method,_,label in RECOVERY}
    labels.update({'sarima_fixed':f'SARIMA{tuple(config["order"])}×{tuple(config["seasonal_order"])}，无漂移',
                   'lag-1':'lag-1 朴素基线','lag-12':'lag-12 季节性朴素基线'})
    order=['lag-1','lag-12']+[r[0] for r in RECOVERY]+['sarima_fixed']
    lines=['# 点预测指标对比','',
        '各方法在所有目的地采用同一组外推模型和超参数；拟合系数各自估计。以下为 micro（有效样本等权）指标。', '',
        '恢复方法沿用冻结的疫情前反事实基线；基线包含历史上按目的地拟合与回退的结果，本次只固定恢复外推配置。','']
    for split,title in [('validation','验证集：2022M08–2023M02，共同有效样本 138 条'),
                        ('test','测试集：2023M08–2024M07，有效样本 164 条')]:
        lines += [f'## {title}','','| 方法 | MAE | MASE | MAPE |','| --- | ---: | ---: | ---: |']
        by_id={r['method']:r for r in all_comparisons[split] if r['aggregation']=='micro'}
        for method in order:
            label=labels[method]
            link=f'visual/html/{method}.html' if method not in ('lag-1','lag-12') else f'{method}/README.md'
            row=by_id[method]
            lines.append(f'| [{label}]({link}) | {row["mae"]:,.2f} | {row["mase"]:.4f} | {row["mape_percent"]:.2f}% |')
        lines.append('')
    lines += ['验证表统一排除 lag-12 缺少历史来源的两个智利预测；测试缺失真实值不计分。生成结果 CSV、PNG 与 JSON 日志已清理并忽略，此处保留已完成运行的指标快照。',
              'SARIMA 固定阶数由已有验证集的宏平均 MASE 选择，不再逐目的地选阶数；与此前混合选型结果不同。测试为本地诊断。']
    (ROOT/'experiments/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata['methods']=[{'method':method,'label':labels[method],'prediction_files':plot_sources.get(method)} for method in order]
    json_write(HERE/'outputs/manifest.json',metadata)
    write_method_reports(config,chosen)
    print(json.dumps({'fixed_sarima':config,'methods':order,'validation_n':138,'test_n':164,'models_refitted':False}))


def write_method_reports(config,chosen):
    window=ROOT/'experiments/windowed-recovery'
    lines=['# 窗口恢复指数预测（Windowed recovery）','',
        '每个目的地独立预测恢复指数，所有目的地使用同一种模型和同一组窗口、阻尼参数；不进行逐目的地选型。', '',
        '当前保留的三组配置：12 个月普通线性趋势（φ=1）、12 个月中位月增量（φ=0.95）、18 个月中位月增量（φ=0.95）。', '',
        'r=exp(z)=(y+1)/(b+1)，point=max(0,(b+1)r_hat-1)。窗口为预测起点前 n 个日历月，内部缺失不补零。验证起点 2022M07，测试起点 2023M07，均一次预测整个目标期。', '',
        '反事实基线及疫情前滚动验证、拟合参数可从原始数据重新生成至 outputs/baseline/ 与 metrics/baseline_cv/，不再依赖已删除的方法目录。基线仍保留历史目的地差异及回退。生成日志与中间文件已清理。', '',
        '完整复现：`python experiments/run_retained.py`。分别运行时：先执行 `python experiments/windowed-recovery/run.py`，再执行 `python experiments/comparison/run.py` 与 `python experiments/visual/generate_method_comparisons.py`。', '',
        '原 73 组固定配置仍可重新计算；仅当前三组列入实验首页。生成结果 CSV、PNG、JSON 日志已按要求清理并忽略，HTML 和指标快照保留。原始输入在 ../data/ 和根目录 testset.csv，不忽略。设计来源见 design.md。', '',
        '本次整理只读取保存结果，没有重新调参或训练；Notebook 不修改。']
    for split,title in [('validation','验证集（完整 140 条）'),('test','测试集（164 条有效真值）')]:
        lines += ['',f'## {title}','','| 配置 | MAE | MASE | MAPE |','| --- | ---: | ---: | ---: |']
        for method,candidate,label in RECOVERY:
            row=next(r for r in csv_read(window/f'retained/{method}/{split}/metrics/summary.csv') if r['aggregation']=='micro')
            lines.append(f'| [{label}](../visual/html/{method}.html) | {row["mae"]:,.2f} | {row["mase"]:.4f} | {row["mape_percent"]:.2f}% |')
    lines += ['','## 文件职责','','- `retained/<配置>/<时段>/outputs/`：当前配置的长表、宽表预测及逐条误差。',
        '- `retained/<配置>/<时段>/metrics/`：汇总、按目的地、按预测步长指标，以及与基线共同有效样本指标。',
        '- `outputs/baseline/`、`metrics/baseline_cv/`：冻结基线和疫情前证据。',
        '- `metrics/<时段>/candidate_*.csv`：所有固定配置的历史搜索结果；两个时段始终分开。',
        '- `compare_validation_test.py`：匹配样本诊断，不拟合模型。',
        '- `models.py`：近期水平、线性趋势、中位月增量及对照模型。',
        '- `test_models.py`：窗口、缺失、数值及边界检查。']
    (window/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sarima=ROOT/'experiments/sarima'
    lines=['# 固定配置 SARIMA 点预测','',
        f'全部目的地统一采用 SARIMA{tuple(config["order"])}×{tuple(config["seasonal_order"])}，trend={config["trend"]}。系数各自拟合。', '',
        f'配置 ID：`{chosen}`。从已有 8 个候选中按验证集宏平均 MASE 选择；所有目的地均需收敛且验证预测完整。测试不参与选择。', '',
        '当前结果位于 `fixed_global/validation/` 和 `fixed_global/test/`，各自包含 outputs 与 metrics。`outputs/fixed_global_config.json` 记录固定规则。旧的逐目的地混合选型输出仅作为历史记录，不用于当前图表或首页对比。', '',
        '完整复现：`python experiments/run_retained.py --clean`。从头重跑候选：`python experiments/sarima/run.py`；随后运行 comparison/run.py 汇总。结果 CSV、PNG、JSON 日志已清理并忽略，指标快照和自包含 HTML 保留。所有预测从各自截止时点一次生成，内部缺失由状态空间模型处理。']
    for split,title in [('validation','验证集'),('test','测试集：本地诊断')]:
        lines += ['',f'## {title}','','### 当前固定配置','',md_table(csv_read(sarima/f'fixed_global/{split}/metrics/summary.csv'),['aggregation','n_scored','mae','mase','mape_percent']),
                  '','### 全部候选固定配置','',md_table(csv_read(sarima/f'metrics/{split}/candidate_summary.csv'),['candidate_id','aggregation','n_scored','mae','mase','mape_percent']),
                  '','### 全部参数按目的地','',md_table(csv_read(sarima/f'metrics/{split}/candidate_metrics.csv'),['destination','candidate_id','converged','n_scored','mae','mase','mape_percent']),
                  '','### 全部参数按步长','',md_table(csv_read(sarima/f'metrics/{split}/candidate_by_horizon.csv'),['candidate_id','horizon','Date','n_scored','mae','mase','mape_percent'])]
    (sarima/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
