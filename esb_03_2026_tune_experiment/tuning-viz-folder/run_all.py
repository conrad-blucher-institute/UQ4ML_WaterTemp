"""
Run the full visualization pipeline: adapt → generate predictions → visualize + serve.

Usage:
    python run_all.py                              # GUI folder picker at each step
    python run_all.py --source path/to/results     # skip GUI pickers

Steps:
    1. adapt_results.py   — convert coworker's results to tuning-viz format
    2. generate_predictions.py — run .keras models to produce prediction CSVs
    3. run_v16.py          — generate HTML visualizations + launch local server

Each step is optional — if the adapted folder already exists, step 1 is skipped.
If all predictions are up-to-date, step 2 finishes instantly.
"""

import argparse
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = str((_SCRIPT_DIR / '../../results').resolve())
_ADAPTER_SUFFIX = '_adapted-to-VE'


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select results folder (raw or already adapted)",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def main():
    parser = argparse.ArgumentParser(description="Run full visualization pipeline")
    parser.add_argument('--source', type=str, default=None,
                        help="Path to results folder (skip GUI picker)")
    parser.add_argument('--skip-adapt', action='store_true',
                        help="Skip the adapt step (folder is already in tuning-viz format)")
    parser.add_argument('--skip-generate', action='store_true',
                        help="Skip the generate predictions step")
    parser.add_argument('--no-serve', action='store_true',
                        help="Skip launching the local server after generating plots")
    args = parser.parse_args()

    if args.source:
        source = Path(args.source)
    else:
        source = pick_folder()

    print(f"Source folder: {source}")

    # ── Step 1: Adapt ──
    # Determine if this folder needs adapting or is already adapted
    adapted_folder = source.parent / f"{source.name}{_ADAPTER_SUFFIX}"
    already_adapted = source.name.endswith(_ADAPTER_SUFFIX)

    if already_adapted:
        results_folder = source
        print(f"\n{'='*60}")
        print(f"Step 1: ADAPT — Skipped (folder already adapted)")
        print(f"{'='*60}")
    elif args.skip_adapt:
        results_folder = source
        print(f"\n{'='*60}")
        print(f"Step 1: ADAPT — Skipped (--skip-adapt)")
        print(f"{'='*60}")
    elif adapted_folder.exists():
        results_folder = adapted_folder
        print(f"\n{'='*60}")
        print(f"Step 1: ADAPT — Skipped (adapted folder already exists)")
        print(f"  Using: {adapted_folder}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Step 1: ADAPT")
        print(f"{'='*60}")
        from adapt_results_v2 import adapt
        adapt(source)
        results_folder = adapted_folder

    print(f"\nResults folder for remaining steps: {results_folder}")

    # ── Step 2: Generate Predictions ──
    if args.skip_generate:
        print(f"\n{'='*60}")
        print(f"Step 2: GENERATE PREDICTIONS — Skipped (--skip-generate)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Step 2: GENERATE PREDICTIONS")
        print(f"{'='*60}")
        from generate_predictions import main as gen_main
        gen_main(results_folder)

    # ── Step 3: Visualize ──
    print(f"\n{'='*60}")
    print(f"Step 3: VISUALIZE (run_v16)")
    print(f"{'='*60}")

    import glob
    import pandas as pd
    from tuning_viz.data_loader import TuningResultsLoader

    csv_paths = sorted(glob.glob(str(results_folder / '*_progress.csv')))
    if not csv_paths:
        print(f"No *_progress.csv files found in {results_folder}")
        sys.exit(1)

    output_dir = str(results_folder.parent / f"{results_folder.name}_visuals")
    print(f"Found CSVs: {[Path(p).name for p in csv_paths]}")
    print(f"Output dir: {output_dir}")

    from run_v16 import load_data, infer_metric_column, run_v16

    for csv_path in csv_paths:
        metric_column = infer_metric_column(csv_path)
        data = load_data([csv_path], metric_column)
        run_v16(data, metric_column, output_dir, results_folder=str(results_folder))

    # ── Step 4: Serve ──
    if args.no_serve:
        print(f"\nDone. Plots saved to: {output_dir}")
        print("Run serve_timeseries_v2.py to view them in the browser.")
    else:
        print("\nLaunching local server for visualizations...")
        from serve_timeseries_v2 import main as serve_main
        serve_main(folder=output_dir)


if __name__ == '__main__':
    main()
