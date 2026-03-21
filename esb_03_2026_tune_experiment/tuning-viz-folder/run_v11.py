"""
Run V11 visualizations for tuning results.

Usage:
    python run_v11.py                       # GUI folder picker
    python run_v11.py --test-iterations 3   # duplicate data 3x for testing iteration plot

A folder picker dialog opens (defaults to results/).
Select a folder containing *_progress.csv files (e.g. mape_results_run1/).
Output HTMLs go to <selected_folder>_visuals/ as a sibling directory.

V11 changes:
    - Dynamic add/remove data series with [+] / [x] buttons
    - Scatter points jittered apart per data series
    - Legend hidden by default
"""

import argparse
import glob
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.unified_plot_v11 import plot_unified_v11
from tuning_viz.plot_types.heatmap_plot_v8 import plot_heatmap_v8
from tuning_viz.plot_types.top_configs_plot_v8 import plot_top_configs_v8, plot_top10_per_cycle_v8
from tuning_viz.plot_types.parallel_coords_v8 import plot_parallel_coords_v8
from tuning_viz.plot_types.iteration_comparison_plot_v8 import plot_iteration_comparison_v8


_VIZ_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = str((_VIZ_DIR / '../../results').resolve())


def pick_folder():
    """Open a GUI folder picker, return selected Path or exit."""
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
    """Find *_progress.csv files in the selected folder."""
    csvs = sorted(glob.glob(str(folder / '*_progress.csv')))
    if not csvs:
        print(f"No *_progress.csv files found in {folder}")
        sys.exit(1)
    return csvs


def infer_metric_column(csv_path):
    """Default metric column based on filename prefix."""
    return 'val_mae'


def load_data(csv_paths, metric_column, test_iterations=0):
    """Load CSVs. If test_iterations > 0, duplicate first CSV that many times."""
    loader = TuningResultsLoader(metric_column=metric_column)
    if test_iterations > 0:
        paths = [csv_paths[0]] * test_iterations
        print(f"[TEST MODE] Duplicating {csv_paths[0]} x{test_iterations}")
    else:
        paths = csv_paths
    dfs = []
    for i, path in enumerate(paths, start=1):
        df = loader.load(path, iteration_id=i)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def run_v11(data, metric_column, output_dir):
    """Generate all v11 plots."""
    print(f"\n{'='*60}")
    print(f"V11 Visualizations: {output_dir}")
    print(f"{'='*60}")
    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    if 'cycle' in data.columns:
        print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Unified Explorer (scatter/box/violin + dynamic data series) ---")
    plot_unified_v11(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmaps ---")
    plot_heatmap_v8(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs ---")
    plot_top_configs_v8(data, metric_column, output_dir=output_dir)
    plot_top10_per_cycle_v8(data, metric_column, output_dir=output_dir)

    print("\n--- Parallel Coordinates ---")
    plot_parallel_coords_v8(data, metric_column, output_dir=output_dir)

    print("\n--- Iteration Stability ---")
    plot_iteration_comparison_v8(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v11 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V11 tuning visualizations")
    parser.add_argument(
        '--test-iterations', type=int, default=0,
        help="Duplicate data N times for testing the iteration stability plot."
    )
    args = parser.parse_args()

    folder = pick_folder()
    csv_paths = discover_csvs(folder)
    output_dir = str(folder.parent / f"{folder.name}_visuals")

    print(f"Found CSVs: {[Path(p).name for p in csv_paths]}")
    print(f"Output dir: {output_dir}")

    for csv_path in csv_paths:
        metric_column = infer_metric_column(csv_path)
        data = load_data([csv_path], metric_column, test_iterations=args.test_iterations)
        run_v11(data, metric_column, output_dir)
