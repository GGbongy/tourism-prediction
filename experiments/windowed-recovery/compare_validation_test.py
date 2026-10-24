"""Compare fixed configurations across periods; do not refit or select models."""
from pathlib import Path
import sys
for deps in (Path(__file__).resolve().parent/'.deps',Path(__file__).resolve().parents[1]/'sarima/.deps'):
    if deps.is_dir():
        sys.path.insert(0,str(deps));break

import hashlib
import json
import re
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    data={s:pd.read_csv(HERE/f'outputs/{s}/candidate_scored_predictions.csv',float_precision='round_trip')
          for s in ('validation','test')}
    reference=data['test'].loc[data['test'].candidate_id==data['test'].candidate_id.iloc[0]]
    counts=reference.groupby('destination').actual.count()
    countries=list(counts[counts==12].index)
    scale=data['validation'][['destination','mase_denominator']].drop_duplicates().set_index('destination').mase_denominator
    summaries={}
    for split,frame in data.items():
        subset=frame.loc[frame.destination.isin(countries)&frame.horizon.le(7)].copy()
        subset['matched_mase']=subset.absolute_error/subset.destination.map(scale)
        result=subset.groupby('candidate_id').agg(n_scored=('actual','count'),mae=('absolute_error','mean'),
            mase_fixed_2022=('matched_mase','mean'),mape_percent=('absolute_percentage_error','mean'))
        result['mase_rank']=result.mase_fixed_2022.rank(method='min').astype(int)
        result['scale_cutoff']='2022M07'
        result['n_destinations']=len(countries)
        result['horizons']='1-7'
        result.to_csv(HERE/f'metrics/{split}/fixed_config_matched_diagnostic.csv')
        summaries[split]=result
    correlation=summaries['validation'].mase_fixed_2022.rank().corr(summaries['test'].mase_fixed_2022.rank())
    overlap=len(set(summaries['validation'].nsmallest(10,'mase_fixed_2022').index)&
                set(summaries['test'].nsmallest(10,'mase_fixed_2022').index))
    metadata={'models_refitted':False,'configuration_changed':False,'countries':countries,'horizons':list(range(1,8)),
        'scale_cutoff':'2022M07','mase_rank_spearman':float(correlation),'top10_overlap':overlap,
        'source_files':{s:{'path':f'outputs/{s}/candidate_scored_predictions.csv',
            'sha256':hashlib.sha256((HERE/f'outputs/{s}/candidate_scored_predictions.csv').read_bytes()).hexdigest()}
            for s in data}}
    out=HERE/'outputs/diagnostics';out.mkdir(parents=True,exist_ok=True)
    (out/'validation_test_matching.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    examples=['last_r_n1','median_slope_r_n12_p0.95','median_slope_r_n18_p0.95',
              'median_slope_r_nall_p0.95','linear_r_n12_p1']
    parts=['## 固定配置：验证与测试是否一致','',
        '后续统一使用同一种模型与同一组超参数；目的地之间只允许拟合系数不同，不再按目的地选型。此前逐目的地结果只作历史诊断。','',
        '原始验证为 20 个目的地、7 步、140 条；测试为 14 个有真值的目的地（其中一个仅 8 个月）、12 步、164 条。两个时期的原始 MASE 分母也分别截至 2022M07 和 2023M07，不能直接把指标绝对值变化解释为同条件下的变化。','',
        '补充诊断统一到 13 个测试完整的目的地、前 7 步（各 91 条），两个时期均使用截至 2022M07 的固定 MASE 分母。所有候选仍是相同的固定配置，没有重新选型或拟合。','',
        f'73 组固定配置的匹配样本 MASE 排名 Spearman 相关系数为 {correlation:.4f}，前十名重合 {overlap} 项。总体优劣存在较强一致性，但具体第一名、窗口和增长强度并不稳定。','']
    for split,title in [('validation','验证：匹配样本'),('test','测试：匹配样本（事后诊断）')]:
        parts += [f'### {title}','','| candidate_id | n_scored | MAE | MASE（固定分母） | MAPE (%) | MASE 排名 |',
                  '| --- | --- | --- | --- | --- | --- |']
        for candidate in examples:
            r=summaries[split].loc[candidate]
            parts.append(f'| {candidate} | {int(r.n_scored)} | {r.mae:,.2f} | {r.mase_fixed_2022:.4f} | {r.mape_percent:.2f} | {int(r.mase_rank)} |')
        parts.append('')
    parts += ['12 个月普通线性趋势在验证中排名第一，匹配测试排名第二十二；18 个月中位增量在验证中排名第十八，匹配测试排名第一。因此“验证选中”与“测试最优”不能等同。','',
        '该诊断根据已知数据可用性匹配样本，是事后分析，不是新的验证方案。恢复阶段、预测长度和样本差异共同影响原始结果；匹配后名次仍变化，说明差异不能仅归因于统计口径。','',
        '运行：`python experiments/windowed-recovery/compare_validation_test.py`。两期完整表分别在 metrics/validation/fixed_config_matched_diagnostic.csv 和 metrics/test/fixed_config_matched_diagnostic.csv。']
    section='\n'.join(parts)
    marker='<!-- fixed-config-comparison:start -->';end='<!-- fixed-config-comparison:end -->'
    readme=HERE/'README.md';text=readme.read_text(encoding='utf-8')
    block=marker+'\n'+section+'\n'+end+'\n\n'
    if marker in text:
        text=re.sub(re.escape(marker)+r'[\s\S]*?'+re.escape(end)+r'\n*',lambda _:block,text)
    else:
        text=text.replace('## 结论先读',block+'## 结论先读',1)
    readme.write_text(text,encoding='utf-8')
    print(json.dumps({'matched_n':len(countries)*7,'ranking_correlation':correlation,'top10_overlap':overlap}))


if __name__=='__main__':main()
