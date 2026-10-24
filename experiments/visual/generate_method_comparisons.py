"""Plot the four retained fixed models using saved predictions only."""
import csv
import hashlib
import json
from generate_trends import HERE,ROOT,COLORS,SPLITS,figure,ordinal,read_csv


def read_long(path,column):
    result={}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['destination'],ordinal(row['Date']))
            if key in result:raise ValueError(f'Duplicate key in {path}: {key}')
            result[key]=float(row[column]) if row[column].strip() else None
    return result


def main():
    source=ROOT/'experiments/comparison/outputs/manifest.json'
    manifest=json.loads(source.read_text(encoding='utf-8'))
    countries,history=read_csv(ROOT/'experiments/data/ISF-TDF2023.csv')
    kinds=[('observed','Observed',''),('validation','Validation forecast','8 5'),('test','Test forecast','3 3')]
    figures=[]
    for method in manifest['methods']:
        if not method.get('prediction_files'):continue
        method_id=method['method']
        paths={s:ROOT/p for s,p in method['prediction_files'].items()}
        forecasts={s:read_long(path,'point') for s,path in paths.items()}
        scored_test=paths['test'].with_name('scored_predictions.csv')
        actual=read_long(scored_test,'actual')
        for split,start,end in [('validation','2022M08','2023M02'),('test','2023M08','2024M07')]:
            expected={(c,m) for c in countries for m in range(ordinal(start),ordinal(end)+1)}
            if set(forecasts[split])!=expected or any(v is None for v in forecasts[split].values()):
                raise ValueError(f'Incomplete {method_id} {split} predictions')
        complete=[c for c in countries if all(actual[(c,m)] is not None for m in range(ordinal('2023M08'),ordinal('2024M07')+1))]
        color_order=complete+[c for c in countries if c not in complete]
        colors={c:COLORS[i] for i,c in enumerate(color_order)}
        keys=[];specs={};legends=[]
        for c in countries:
            group=[]
            for kind,name,dash in kinds:
                key=f'{c}::{kind}';keys.append(key);group.append(key)
                specs[key]={'label':f'{c} | {name}','color':colors[c],'dash':dash}
            legends.append({'label':c,'series_keys':group,'color':colors[c],'family':'destination'})
        for kind,name,dash in kinds:
            legends.append({'label':name,'series_keys':[f'{c}::{kind}' for c in countries],
                            'color':'#545b63','dash':dash,'family':'kind'})
        rows={}
        for m in range(ordinal('2022M01'),ordinal('2024M07')+1):
            row={}
            for c in countries:
                row[f'{c}::observed']=history.get(m,{}).get(c) if m<=ordinal('2023M07') else actual[(c,m)]
                row[f'{c}::validation']=forecasts['validation'].get((c,m))
                row[f'{c}::test']=forecasts['test'].get((c,m))
            rows[m]=row
        titles={
            'recovery_linear_12':'12-month linear recovery trend, no damping',
            'recovery_median_12':'12-month median recovery increment, phi = 0.95',
            'recovery_median_18':'18-month median recovery increment, phi = 0.95',
            'sarima_fixed':'SARIMA (0,1,1) x (1,0,0,12), fixed configuration'}
        filename=method_id+'.html'
        figure(keys,rows,HERE/'html'/filename,titles[method_id],series_specs=specs,
               legend_specs=legends,splits=SPLITS,auto_scale_y=True)
        figures.append({'html':filename,'method':method_id,'series_count':60,
                        'validation_values':len(forecasts['validation']),'test_values':len(forecasts['test']),
                        'source_files':{s:{'path':str(p.relative_to(ROOT)),
                            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for s,p in paths.items()}})
    if len(figures)!=4:raise ValueError('Expected four retained method figures')
    (HERE/'method_comparisons_manifest.json').write_text(json.dumps({'models_refitted':False,'figures':figures},indent=2),encoding='utf-8')
    print(json.dumps({'html_files':[f['html'] for f in figures],'models_refitted':False}))


if __name__=='__main__':main()
