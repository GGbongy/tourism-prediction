"""File helpers shared by retained experiments."""
import json
import pandas as pd


def csv_read(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
