"""Generate one standalone interactive trend HTML using the standard library.

Run: python experiments/visual/generate_trends.py
Public history is cached locally; testset.csv is used only in this experiment.
Selection: every test month is finite and nonnegative, and each public partition
has observations. Leading historical blanks and internal gaps remain unfilled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URL = 'https://raw.githubusercontent.com/tulip-lab/open-data/main/ISF-TDF2023/ISF-TDF2023.csv'
SPLITS = [('Training', None, '2022M07', '#f4f6f8'),
          ('Validation', '2022M08', '2023M02', '#e6eff8'),
          ('Locked audit', '2023M03', '2023M07', '#f5ecdc'),
          ('Test', '2023M08', '2024M07', '#e3f0e9')]
COLORS = ['#21618c', '#b03a2e', '#148f77', '#7d3c98', '#b9770e',
          '#34495e', '#d35400', '#2874a6', '#a9326f', '#558b2f',
          '#6c5b4c', '#00838f', '#5c6bc0', '#c0392b', '#455a64']


def ordinal(label):
    if not re.fullmatch(r'\d{4}M(0[1-9]|1[0-2])', label):
        raise ValueError(f'Invalid month: {label}')
    return int(label[:4]) * 12 + int(label[-2:]) - 1


def label(month):
    return f'{month // 12}M{month % 12 + 1:02d}'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or columns[0] != 'Date' or len(set(columns)) != len(columns):
            raise ValueError(f'Invalid CSV header: {path}')
        rows = {}
        for row in reader:
            month = ordinal(row['Date'])
            if month in rows:
                raise ValueError(f'Duplicate date in {path}: {row["Date"]}')
            values = {}
            for country in columns[1:]:
                raw = row[country]
                value = float(raw) if raw and raw.strip() else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f'Invalid value: {country}, {row["Date"]}')
                values[country] = value
            rows[month] = values
    return columns[1:], rows


def figure(countries, rows, output, title):
    esc = html.escape
    months = sorted(rows)
    start = min(m for m in months if any(rows[m][c] is not None for c in countries))
    end = ordinal('2024M07')
    months = list(range(start, end + 1))
    width, left, right, top, bottom = 1440, 110, 1390, 125, 575
    legend_rows = math.ceil(len(countries) / 4)
    height = 670 + 28 * legend_rows
    maximum = max(rows[m][c] for m in months for c in countries
                  if m in rows and rows[m][c] is not None)
    raw_step = maximum / 5 if maximum else 1
    power = 10 ** math.floor(math.log10(raw_step))
    step = next(n * power for n in (1, 2, 2.5, 5, 10) if n * power >= raw_step)
    ymax = math.ceil(maximum / step) * step or step
    x = lambda m: left + (m - start + .5) / (end - start + 1) * (right - left)
    y = lambda value: bottom - value / ymax * (bottom - top)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="group" aria-label="{esc(title)}">',
           '<style>text{font-family:Georgia,"Times New Roman",serif;fill:#222} .tick{font-size:15px}'
           '.legend-item{cursor:pointer;user-select:none}.legend-item[aria-pressed="false"]{opacity:.3}'
           '.legend-item:focus-visible{outline:1px dotted #545b63}.series-group.is-hidden{display:none}</style>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width/2}" y="38" text-anchor="middle" font-size="25">{esc(title)}</text>']
    for i, (name, a, b, color) in enumerate(SPLITS):
        lo = max(start, ordinal(a)) if a else start
        hi = min(end, ordinal(b))
        svg.append(f'<rect class="partition-band" x="{x(lo)-.5*(right-left)/(end-start+1):.2f}" y="{top}" width="{(hi-lo+1)*(right-left)/(end-start+1):.2f}" height="{bottom-top}" fill="{color}"/>')
        lx = left + i * 322
        svg += [f'<rect x="{lx}" y="64" width="15" height="15" fill="{color}" stroke="#aaa"/>',
                f'<text x="{lx+23}" y="77" font-size="16">{name}</text>',
                f'<text x="{lx+23}" y="99" font-size="13">{label(lo)} – {label(hi)}</text>']
    svg.append('<g class="y-grid">')
    for tick in range(round(ymax / step) + 1):
        value = tick * step
        yy = y(value)
        svg += [f'<line x1="{left}" x2="{right}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#d9dfe4" stroke-width=".7"/>',
                f'<text x="{left-14}" y="{yy+5:.2f}" text-anchor="end" class="tick">{value:,.0f}</text>']
    svg += ['</g>', f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#333"/>',
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" stroke="#333"/>',
            f'<text transform="translate(29 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-size="18">Tourist arrivals</text>',
            f'<text x="{(left+right)/2}" y="641" text-anchor="middle" font-size="18">Month</text>']
    stride = max(1, math.ceil((end-start+1) / 12 / 9))
    ticks = [start] + [m for m in months if m % 12 == 0 and (m//12-start//12) % stride == 0 and m > start+9 and m < end-9] + [end]
    svg.append('<g class="x-ticks">')
    for m in ticks:
        xx = x(m)
        svg += [f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{bottom}" y2="{bottom+6}" stroke="#333"/>',
                f'<text x="{xx:.2f}" y="{bottom+27}" text-anchor="middle" class="tick">{label(m).replace("M", "-")}</text>']
    svg.append('</g>')
    svg.append(f'<defs><clipPath id="plot-clip"><rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}"/></clipPath></defs>')
    for _, a, _, _ in SPLITS[1:]:
        xx = x(ordinal(a)-.5)
        svg.append(f'<line class="split-boundary" x1="{xx:.2f}" x2="{xx:.2f}" y1="{top}" y2="{bottom}" stroke="#545b63" stroke-width="1.3" stroke-dasharray="6 5"><title>{a}</title></line>')
    for i, country in enumerate(countries):
        color = COLORS[i % len(COLORS)]
        svg.append(f'<g id="destination-{i}" class="series-group" clip-path="url(#plot-clip)">')
        segments, segment = [], []
        for m in months:
            value = rows.get(m, {}).get(country)
            if value is None:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append((m, value))
        if segment:
            segments.append(segment)
        for segment in segments:
            points = ' '.join(f'{x(m):.2f},{y(v):.2f}' for m, v in segment)
            svg.append(f'<polyline class="series" points="{points}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
            for m, v in segment:
                svg.append(f'<circle cx="{x(m):.2f}" cy="{y(v):.2f}" r="{1.8 if len(countries)==1 else 1.1}" fill="{color}"><title>{esc(country)} | {label(m)} | {v:,.0f}</title></circle>')
        svg.append('</g>')
        lx, ly = left + (i % 4) * 322, 674 + (i // 4) * 28
        svg += [f'<g class="legend-item" tabindex="0" role="button" aria-label="{esc(country)}" aria-pressed="true" aria-controls="destination-{i}">',
                f'<rect x="{lx-5}" y="{ly-19}" width="300" height="26" fill="transparent"/>',
                f'<line x1="{lx}" x2="{lx+23}" y1="{ly-5}" y2="{ly-5}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{lx+31}" y="{ly}" font-size="16">{esc(country)}</text>', '</g>']
    svg.append('</svg>')
    interaction = '<script>\n(() => {\nconst geometry = ' + json.dumps({
        'left': left, 'right': right, 'top': top, 'bottom': bottom,
        'start': start, 'months': end-start+1, 'ymax': ymax}) + ';\n' + '''
document.querySelectorAll('.legend-item').forEach(item => {
  const toggle = () => {
    const visible = item.getAttribute('aria-pressed') === 'true';
    document.getElementById(item.getAttribute('aria-controls')).classList.toggle('is-hidden', visible);
    item.setAttribute('aria-pressed', String(!visible));
  };
  item.addEventListener('click', toggle);
  item.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggle();
    }
  });
});

const svg = document.querySelector('svg');
const {left, right, top, bottom, start, months, ymax} = geometry;
const full = {x0:left, x1:right, y0:top, y1:bottom};
let view = {...full};
let drag = null;
let pending = false;
const xTicks = svg.querySelector('.x-ticks');
const yGrid = svg.querySelector('.y-grid');
const originalX = xTicks.innerHTML, originalY = yGrid.innerHTML;
const bands = [...svg.querySelectorAll('.partition-band')].map(el => ({
  el, x:Number(el.getAttribute('x')), width:Number(el.getAttribute('width'))
}));
const boundaries = [...svg.querySelectorAll('.split-boundary')].map(el => ({
  el, x:Number(el.getAttribute('x1'))
}));
const points = [...svg.querySelectorAll('.series-group circle')].map(el => ({
  el, x:Number(el.getAttribute('cx')), y:Number(el.getAttribute('cy'))
}));
const lines = [...svg.querySelectorAll('polyline.series')];
const sx = value => left + (value-view.x0)/(view.x1-view.x0)*(right-left);
const sy = value => top + (value-view.y0)/(view.y1-view.y0)*(bottom-top);
function position(event) {
  return new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
}
const inside = p => p.x>=left && p.x<=right && p.y>=top && p.y<=bottom;
function element(tag, attributes, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([key,value]) => el.setAttribute(key,value));
  if (text !== undefined) el.textContent = text;
  return el;
}
function render() {
  pending = false;
  const ax = (right-left)/(view.x1-view.x0), ay = (bottom-top)/(view.y1-view.y0);
  lines.forEach(el => el.setAttribute('transform',
    `matrix(${ax} 0 0 ${ay} ${left-ax*view.x0} ${top-ay*view.y0})`));
  points.forEach(({el,x,y}) => {el.setAttribute('cx',sx(x));el.setAttribute('cy',sy(y));});
  bands.forEach(({el,x,width}) => {
    const a=Math.max(left,Math.min(right,sx(x))), b=Math.max(left,Math.min(right,sx(x+width)));
    el.setAttribute('x',a);el.setAttribute('width',Math.max(0,b-a));
  });
  boundaries.forEach(({el,x}) => {
    const xx=sx(x);el.setAttribute('x1',xx);el.setAttribute('x2',xx);
    el.style.display=xx<left || xx>right ? 'none' : '';
  });
  if (view.x0===left && view.x1===right && view.y0===top && view.y1===bottom) {
    xTicks.innerHTML=originalX;yGrid.innerHTML=originalY;
    return;
  }
  xTicks.replaceChildren();yGrid.replaceChildren();
  const lo=start-.5+(view.x0-left)/(right-left)*months;
  const hi=start-.5+(view.x1-left)/(right-left)*months;
  const stride=[1,2,3,6,12,24,36,60,120].find(n=>n>=(hi-lo)/8) || 120;
  for(let m=Math.ceil(lo/stride)*stride;m<=hi;m+=stride) {
    const xx=sx(left+(m-start+.5)/months*(right-left));
    const label=String(Math.floor(m/12))+'-'+String(m%12+1).padStart(2,'0');
    xTicks.append(element('line',{x1:xx,x2:xx,y1:bottom,y2:bottom+6,stroke:'#333'}),
      element('text',{x:xx,y:bottom+27,'text-anchor':'middle',class:'tick'},label));
  }
  const min=ymax*(bottom-view.y1)/(bottom-top), max=ymax*(bottom-view.y0)/(bottom-top);
  const raw=(max-min)/5, power=10**Math.floor(Math.log10(raw));
  const step=[1,2,2.5,5,10].map(n=>n*power).find(n=>n>=raw);
  for(let v=Math.ceil(min/step)*step;v<=max+step*1e-9;v+=step) {
    const yy=sy(bottom-v/ymax*(bottom-top));
    yGrid.append(element('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:'#d9dfe4','stroke-width':'.7'}),
      element('text',{x:left-14,y:yy+5,'text-anchor':'end',class:'tick'},
        v.toLocaleString('en-US',{notation:'compact',maximumFractionDigits:2})));
  }
}
function schedule() {if(!pending){pending=true;requestAnimationFrame(render);}}
function bounded(a,b,min,max) {
  const span=b-a;
  if(span>=max-min)return [min,max];
  const lo=Math.max(min,Math.min(max-span,a));return [lo,lo+span];
}
svg.addEventListener('wheel',event=>{
  const p=position(event);if(!inside(p))return;
  event.preventDefault();
  const factor=Math.exp(Math.max(-500,Math.min(500,event.deltaY))*0.002);
  const cx=view.x0+(p.x-left)/(right-left)*(view.x1-view.x0);
  const cy=view.y0+(p.y-top)/(bottom-top)*(view.y1-view.y0);
  const wx=Math.max((right-left)/60,Math.min(right-left,(view.x1-view.x0)*factor));
  const wy=Math.max((bottom-top)/60,Math.min(bottom-top,(view.y1-view.y0)*factor));
  const x0=cx-(p.x-left)/(right-left)*wx, y0=cy-(p.y-top)/(bottom-top)*wy;
  [view.x0,view.x1]=bounded(x0,x0+wx,left,right);
  [view.y0,view.y1]=bounded(y0,y0+wy,top,bottom);
  schedule();
},{passive:false});
svg.addEventListener('pointerdown',event=>{
  const p=position(event);if(!inside(p)||event.button!==0)return;
  drag={p,view:{...view}};svg.setPointerCapture(event.pointerId);svg.style.cursor='grabbing';
  event.preventDefault();
});
svg.addEventListener('pointermove',event=>{
  if(!drag)return;
  const p=position(event), old=drag.view;
  const dx=(p.x-drag.p.x)/(right-left)*(old.x1-old.x0);
  const dy=(p.y-drag.p.y)/(bottom-top)*(old.y1-old.y0);
  [view.x0,view.x1]=bounded(old.x0-dx,old.x1-dx,left,right);
  [view.y0,view.y1]=bounded(old.y0-dy,old.y1-dy,top,bottom);
  schedule();
});
function stopDrag(){drag=null;svg.style.cursor='';}
svg.addEventListener('pointerup',stopDrag);
svg.addEventListener('pointercancel',stopDrag);
svg.addEventListener('lostpointercapture',stopDrag);
svg.addEventListener('dblclick',event=>{
  if(!inside(position(event)))return;
  event.preventDefault();view={...full};schedule();
});
})();
</script>'''
    output.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                      '<meta name="viewport" content="width=device-width, initial-scale=1">'
                      f'<title>{esc(title)}</title><style>body{{margin:0;background:white}}'
                      'main{max-width:1600px;margin:20px auto}svg{display:block;width:100%;height:auto}'
                      '@media print{main{margin:0;max-width:none}@page{size:landscape;margin:8mm}}'
                      '</style></head><body><main>' + '\n'.join(svg) + '</main>' + interaction + '</body></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=HERE / 'data/ISF-TDF2023.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'testset.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'html')
    args = parser.parse_args()
    if not args.history.exists():
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            args.history.write_bytes(response.read())
    countries, history = read_csv(args.history)
    test_countries, test = read_csv(args.test)
    if countries != test_countries:
        raise ValueError('History and test columns must match in order.')
    expected = set(range(ordinal('2023M08'), ordinal('2024M07') + 1))
    if set(test) != expected:
        raise ValueError('Test must contain exactly 2023M08–2024M07.')
    history = {m: row for m, row in history.items() if m <= ordinal('2023M07')}
    selected, excluded = [], {}
    for country in countries:
        missing = [label(m) for m in sorted(test) if test[m][country] is None]
        empty = [name for name, a, b, _ in SPLITS[:3]
                 if not any(row[country] is not None for m, row in history.items()
                            if (a is None or m >= ordinal(a)) and m <= ordinal(b))]
        if missing or empty:
            excluded[country] = {'missing_test_months': missing, 'empty_public_partitions': empty}
        else:
            selected.append(country)
    if not selected:
        raise ValueError('No eligible destinations.')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = history | test
    figure(selected, rows, args.output / 'all_destinations.html', 'Monthly tourist arrivals by destination')
    manifest = {'public_source': URL,
                'history_sha256': hashlib.sha256(args.history.read_bytes()).hexdigest(),
                'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(),
                'selected': selected, 'excluded': excluded,
                'selection_rule': 'Complete 12-month test series; observations in each public partition.',
                'missing_history': 'Not filled; lines break at missing months.',
                'splits': SPLITS, 'html_count': 1,
                'interaction': 'Legend toggles visibility; wheel zooms at cursor; drag pans; double-click resets. No added controls.'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'selected': selected, 'excluded': list(excluded), 'html_count': 1}, indent=2))


if __name__ == '__main__':
    main()
