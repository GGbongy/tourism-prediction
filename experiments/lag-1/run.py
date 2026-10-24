"""Run lag-1 and regenerate this method's outputs, metrics, and README."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_shared'))
from point_baselines import run

if __name__ == '__main__':
    run(lag=1, method_dir=Path(__file__).resolve().parent)
