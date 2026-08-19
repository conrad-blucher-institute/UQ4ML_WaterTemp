"""
Run V16 visualizations for tuning results.

Usage:
    python run_v16.py                       # GUI folder picker
    python run_v16.py --test-iterations 3   # duplicate data 3x for testing iteration plot

V16 changes:
    - Time Series: training trace on validation chart, new testing chart,
      toggle buttons for train/val/test/2021 charts, hide legends button,
      metric dropdown selector, resizable left panel
"""

import argparse
import glob
import sys
import traceback
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
from tuning_viz.data_loader import TuningResultsLoader
# v16 explorer adds on-the-fly Raw/Avg/Avg+Spread aggregation.
# To revert to the previous behavior, swap this import (and the call below) back
# to unified_plot_v15 / plot_unified_v15.
from tuning_viz.plot_types.unified_plot_v17 import plot_unified_v17
from tuning_viz.plot_types.timeseries_cove import plot_timeseries_cove
from tuning_viz.plot_types.heatmap_plot_v8 import plot_heatmap_v8
from tuning_viz.plot_types.top_configs_plot_v12 import plot_top_configs_v12, plot_top10_per_cycle_v12
from tuning_viz.plot_types.parallel_coords_v8 import plot_parallel_coords_v8
from tuning_viz.plot_types.iteration_comparison_plot_v8 import plot_iteration_comparison_v8
from tuning_viz.plot_types.timeseries_compare_v16 import plot_timeseries_compare_v16
from tuning_viz.plot_types.histogram_plot_v1 import plot_histogram_v1

_VIZ_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = str((_VIZ_DIR / '../../results').resolve())


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select tuning results folder (containing *_progress.csv)",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def discover_csvs(folder):
    # Match both the plain `*_progress.csv` and the sharded campaign names
    # (e.g. mape_progress_shard10of40.csv) — the shard suffix means the file no
    # longer *ends* in _progress.csv, so anchor on 'progress' anywhere instead.
    csvs = sorted(glob.glob(str(folder / '*progress*.csv')))
    if not csvs:
        print(f"No *progress*.csv files found in {folder}")
        sys.exit(1)
    return csvs


def infer_metric_column(csv_path):
    return 'val_mae'


def load_data(csv_paths, metric_column, test_iterations=0):
    loader = TuningResultsLoader(metric_column=metric_column)
    if test_iterations > 0:
        # TEST MODE only: fake N iterations by duplicating one file, each with a
        # distinct iteration_id so the iteration-stability plot has something to
        # compare. Real shard files are NOT iterations (see below).
        paths = [csv_paths[0]] * test_iterations
        print(f"[TEST MODE] Duplicating {csv_paths[0]} x{test_iterations}")
        dfs = [loader.load(p, iteration_id=i) for i, p in enumerate(paths, start=1)]
        return pd.concat(dfs, ignore_index=True)

    # Real data: the files are disjoint config SHARDS of one campaign, not
    # repetitions. Pool them, then drive `iteration` from each row's run_id
    # (the loader's rename of run_num) so the actual repetitions — 3 reps per
    # config here — become the iteration axis the stability plot groups over.
    # (The loader stamps iteration with a constant per file; we overwrite it.)
    dfs = [loader.load(path, iteration_id=1) for path in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)
    if 'run_id' in combined.columns:
        combined['iteration'] = pd.to_numeric(combined['run_id'], errors='coerce')
    return combined


def _safe(label, failures, fn, *args, **kwargs):
    """Run one plot; if it raises, log the traceback and keep going so the rest
    of the suite still gets drawn. Records the label of any plot that failed."""
    print(f"\n--- {label} ---")
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - one bad plot shouldn't abort the run
        failures.append(label)
        print(f"[FAILED] {label}: {type(e).__name__}: {e}")
        traceback.print_exc()


def run_v16(data, metric_column, output_dir, results_folder=None):
    print(f"\n{'='*60}")
    print(f"V16 Visualizations: {output_dir}")
    print(f"{'='*60}")
    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    if 'cycle' in data.columns:
        print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    failures = []

    _safe("Unified Explorer (Send to Time Series)", failures,
          plot_unified_v17, data, metric_column, output_dir=output_dir)
    _safe("Histogram", failures,
          plot_histogram_v1, data, metric_column, output_dir=output_dir)
    _safe("Heatmaps", failures,
          plot_heatmap_v8, data, metric_column, output_dir=output_dir)
    _safe("Top Configs (cycle + leadtime dropdowns)", failures,
          plot_top_configs_v12, data, metric_column, output_dir=output_dir)
    _safe("Top 10 per cycle", failures,
          plot_top10_per_cycle_v12, data, metric_column, output_dir=output_dir)
    _safe("Parallel Coordinates", failures,
          plot_parallel_coords_v8, data, metric_column, output_dir=output_dir)
    _safe("Iteration Stability", failures,
          plot_iteration_comparison_v8, data, metric_column, output_dir=output_dir)

    if results_folder:
        # COVE time series: self-contained page the explorer's Send-to-Time-Series
        # opens. Needs the wide CSVs from `python -m esb.predict <campaign>`;
        # skipped (not failed) when they are absent, since most campaigns are
        # screened on metrics alone and never get predictions generated.
        wide_dir = Path(results_folder) / 'predictions_wide'
        # Two layouts are valid: loose *_wide.csv (one combined page), or LT<n>/
        # subfolders from `esb.predict --split-by-leadtime` (one page per lead
        # time). Checking only the first would SKIP the split layout while
        # printing "run esb.predict first" at someone who just did.
        has_loose = wide_dir.is_dir() and any(wide_dir.glob('*_wide.csv'))
        has_split = wide_dir.is_dir() and any(
            p.is_dir() and any(p.glob('*_wide.csv'))
            for p in wide_dir.glob('LT*'))
        if has_loose or has_split:
            # datasets=None: bake whatever esb.predict produced, so this never
            # drifts out of step with what was actually generated.
            # split_by_leadtime=None: let it read the layout off disk too.
            _safe("COVE Time Series (embedded predictions)", failures,
                  plot_timeseries_cove, results_folder,
                  output_dir=output_dir, datasets=None)
        else:
            print(f"\n--- COVE Time Series: SKIPPED (no {wide_dir}, "
                  f"run `python -m esb.predict {results_folder}` first) ---")

        pred_dir = str(Path(results_folder) / 'predictions')
        if Path(pred_dir).exists() and (Path(pred_dir) / '_index.csv').exists():
            _safe("Time Series Comparison (v16)", failures,
                  plot_timeseries_compare_v16, pred_dir, output_dir=output_dir)
        else:
            print("\n--- Time Series Comparison: SKIPPED (no predictions/ folder, run generate_predictions.py first) ---")

    print(f"\n{'='*60}")
    if failures:
        print(f"v16 plots saved to: {output_dir}")
        print(f"{len(failures)} plot(s) FAILED (see tracebacks above): {', '.join(failures)}")
    else:
        print(f"All v16 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V16 tuning visualizations")
    parser.add_argument(
        '--test-iterations', type=int, default=0,
        help="Duplicate data N times for testing the iteration stability plot.",
    )
    parser.add_argument(
        '--no-serve', action='store_true',
        help="Skip launching the local server after generating plots.",
    )
    parser.add_argument(
        '--folder', type=str, default=None,
        help="Results folder containing *_progress.csv (skips the tkinter picker; "
             "headless/GUI-launch friendly).",
    )
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"--folder is not a directory: {folder}")
            sys.exit(1)
    else:
        folder = pick_folder()
    csv_paths = discover_csvs(folder)
    # Put the visuals INSIDE the selected results folder (was a sibling folder).
    output_dir = str(folder / "_visuals")

    print(f"Found CSVs: {[Path(p).name for p in csv_paths]}")
    print(f"Output dir: {output_dir}")

    # Pool ALL shard CSVs into one dataset, then plot once. (Previously this
    # looped per-file and each run_v16() call overwrote the last shard's plots.)
    metric_column = infer_metric_column(csv_paths[0])
    data = load_data(csv_paths, metric_column, test_iterations=args.test_iterations)
    run_v16(data, metric_column, output_dir, results_folder=str(folder))

    if not args.no_serve:
        print("\nLaunching local server for visualizations...")
        from serve_timeseries_v2 import main as serve_main
        serve_main(folder=output_dir)
