"""Delete generated CSV/PNG/JSON/log files while preserving source datasets."""
from pathlib import Path
import hashlib
import json
import os

HERE=Path(__file__).resolve().parent
SKIP={'.deps','.git','node_modules','.agents','.codex','.aws','__pycache__'}
DATA={'data','datasets','inputs'}


def checked(path):
    resolved=path.resolve()
    if resolved==HERE or not resolved.is_relative_to(HERE):
        raise ValueError(f'Cleanup target outside experiments: {resolved}')
    return resolved


def main():
    raw={}
    targets=[]
    for folder,dirs,files in os.walk(HERE,followlinks=False):
        dirs[:]=[d for d in dirs if d not in SKIP]
        for name in files:
            path=Path(folder)/name
            relative=path.relative_to(HERE)
            parts=set(relative.parts[:-1])
            if parts & DATA:
                if path.suffix.lower() in ('.csv','.json'):
                    raw[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
                continue
            generated_record=(path.suffix.lower() in ('.csv','.json') and parts & {'outputs','metrics'})
            visual_manifest=(path.parent==HERE/'visual' and path.name.endswith('manifest.json'))
            if path.suffix.lower() in ('.png','.log') or generated_record or visual_manifest:
                targets.append(checked(path))
    for path in targets:path.unlink()
    for folder,dirs,_ in os.walk(HERE,topdown=False,followlinks=False):
        path=Path(folder)
        if any(part in SKIP for part in path.relative_to(HERE).parts):continue
        if path!=HERE and not any(path.iterdir()):checked(path).rmdir()
    for relative,expected in raw.items():
        if hashlib.sha256((HERE/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Source data changed: {relative}')
    result={'deleted_csv':sum(p.suffix.lower()=='.csv' for p in targets),
            'deleted_png':sum(p.suffix.lower()=='.png' for p in targets),
            'deleted_json':sum(p.suffix.lower()=='.json' for p in targets),
            'deleted_log':sum(p.suffix.lower()=='.log' for p in targets),
            'preserved_source_data_hashes':raw,
            'scope':'Generated project files only; dependency runtime excluded.'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
