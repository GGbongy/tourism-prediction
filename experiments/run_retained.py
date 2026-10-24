"""Regenerate local ignored results, fixed-model comparison tables and HTML figures."""
from pathlib import Path
import subprocess
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean',action='store_true',help='Remove generated CSV, PNG, JSON and logs after preserving HTML and metric reports.')
    args=parser.parse_args()
    for script in ('experiments/lag-1/run.py','experiments/lag-12/run.py',
                   'experiments/sarima/run.py','experiments/windowed-recovery/run.py',
                   'experiments/comparison/run.py','experiments/visual/generate_method_comparisons.py'):
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
    if args.clean:
        subprocess.run([sys.executable,str(ROOT/'experiments/cleanup_generated.py')],cwd=ROOT,check=True)
