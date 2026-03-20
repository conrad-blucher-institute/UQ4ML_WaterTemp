"""
Benchmark: preparingData() repeated calls

Measures elapsed time for calling `preparingData()` N times while swapping
`creatingAdditionalColumns` between original and optimized implementations.

Runs tests for N=16 and N=1440 calls and prints timings.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.helper import utils_mse_crps as utils
from esb2020_2021_experiment.benchmark_creatingAdditionalColumns import (
    creatingAdditionalColumns_ORIGINAL, creatingAdditionalColumns_OPTIMIZED
)


def single_run(num_calls: int, use_optimized: bool):
    """Call preparingData() num_calls times and return elapsed seconds."""
    # Monkeypatch the creatingAdditionalColumns implementation
    if use_optimized:
        utils.creatingAdditionalColumns = creatingAdditionalColumns_OPTIMIZED
    else:
        utils.creatingAdditionalColumns = creatingAdditionalColumns_ORIGINAL

    # Parameters matching typical use
    params = {
        'path_to_data': str(REPO_ROOT / 'data' / 'ESB_datasets'),
        'input_structure': 'descending',
        'independent_year': 'cycle',
        'input_hours_forecast': 12,
        'atp_hours_back': 24,
        'wtp_hours_back': 24,
        'pred_atp_interval': 1,
        'IPPOffset': 0.0,
        'cycle': 0,
        'model': 'MSE',
        'verbose': 0,
    }

    # Warm up 1 call (not counted)
    print(f"Warming up (1 call) - optimized={use_optimized}")
    try:
        _ = utils.preparingData(params['path_to_data'], params['input_structure'], params['independent_year'],
                                params['input_hours_forecast'], params['atp_hours_back'], params['wtp_hours_back'],
                                params['pred_atp_interval'], params['IPPOffset'], params['cycle'], params['model'], params['verbose'])
    except Exception as e:
        print('Warmup call raised exception (continuing):', e)

    start = time.time()
    for i in range(num_calls):
        if (i + 1) % 50 == 0:
            print(f"  progress: {i+1}/{num_calls} calls")
        _ = utils.preparingData(params['path_to_data'], params['input_structure'], params['independent_year'],
                                params['input_hours_forecast'], params['atp_hours_back'], params['wtp_hours_back'],
                                params['pred_atp_interval'], params['IPPOffset'], params['cycle'], params['model'], params['verbose'])
    end = time.time()
    return end - start


def main():
    counts = [16, 4]#, 1440]
    results = []

    for use_opt in (True, False):
        label = 'OPTIMIZED' if use_opt else 'ORIGINAL'
        print('\n' + '='*60)
        print(f"Running benchmark with {label} creatingAdditionalColumns")
        print('='*60)
        for c in counts:
            print(f"\nStarting {c} preparingData() calls (optimized={use_opt})...")
            elapsed = single_run(c, use_opt)
            print(f"Completed {c} calls: {elapsed:.2f}s (avg {elapsed/c:.4f}s per call)")
            results.append({'implementation': label, 'calls': c, 'elapsed': elapsed})

    print('\n' + '='*60)
    print('SUMMARY')
    for r in results:
        print(f"{r['implementation']:9s} | calls={r['calls']:4d} | elapsed={r['elapsed']:.2f}s | avg={r['elapsed']/r['calls']:.4f}s")
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
