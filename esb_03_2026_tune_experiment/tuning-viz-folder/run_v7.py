"""
Run V7 visualizations for MAPE and MSE results.

Usage:
    python run_v7.py                       # single iteration
    python run_v7.py --test-iterations 3   # duplicate run1 3x for testing 06 plot

When you have real multi-iteration data, add paths to OPTIMIZATION_RUNS below.

Output files:
    01_scatter.html
    02_heatmaps.html
    03_boxplot.html
    03b_violin.html
    04_top_configs.html
    04_top10_per_cycle.html
    05_parallel_coords.html
    06_iteration_stability.html  (only with 2+ iterations)
"""

import argparse
from pathlib import Path
import pandas as pd
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.scatter_plot_v7 import plot_scatter_v7
from tuning_viz.plot_types.heatmap_plot_v7 import plot_heatmap_v7
from tuning_viz.plot_types.boxplot_plot_v7 import plot_boxplot_v7
from tuning_viz.plot_types.violin_plot_v7 import plot_violin_v7
from tuning_viz.plot_types.top_configs_plot_v7 import plot_top_configs_v7, plot_top10_per_cycle_v7
from tuning_viz.plot_types.parallel_coords_v7 import plot_parallel_coords_v7
from tuning_viz.plot_types.iteration_comparison_plot_v7 import plot_iteration_comparison_v7


# ============================================================================
# CONFIGURE YOUR CSV PATHS HERE
# Each list entry = one iteration. Add more paths as you complete more runs.
# ============================================================================
OPTIMIZATION_RUNS = {
    'mape': [
        'tune_experiment_results/mape_results_run1/mape_progress.csv',
        # 'tune_experiment_results/mape_results_run2/mape_progress.csv',
        # 'tune_experiment_results/mape_results_run3/mape_progress.csv',
    ],
    'mse': [
        'tune_experiment_results/mse_results_run1/mse_progress.csv',
        # 'tune_experiment_results/mse_results_run2/mse_progress.csv',
    ],
}


def load_data(csv_paths: list, metric_column: str) -> pd.DataFrame:
    """Load one CSV per iteration. Each path = one iteration."""
    loader = TuningResultsLoader(metric_column=metric_column)
    dfs = []
    for i, path in enumerate(csv_paths, start=1):
        df = loader.load(path, iteration_id=i)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def run_v7(csv_paths: list, metric_column: str, output_dir: str):
    """Load data and generate all v7 plots."""
    print(f"\n{'='*60}")
    print(f"V7 Visualizations: {output_dir}")
    print(f"Iterations: {len(csv_paths)}")
    print(f"{'='*60}")

    data = load_data(csv_paths, metric_column)

    print(f"Loaded {len(data)} total configurations ({len(csv_paths)} iteration(s))")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Scatter (x-axis selector, color-by, filters) ---")
    plot_scatter_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmaps (param pairs × leadtimes, auto-resize) ---")
    plot_heatmap_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Boxplot (x-axis + color-by, jitter, auto-height) ---")
    plot_boxplot_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Violin (same controls as boxplot) ---")
    plot_violin_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs (cycle dropdown + top 10 grid) ---")
    plot_top_configs_v7(data, metric_column, output_dir=output_dir)
    plot_top10_per_cycle_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Parallel Coordinates (bold fonts, auto-height) ---")
    plot_parallel_coords_v7(data, metric_column, output_dir=output_dir)

    print("\n--- Iteration Stability (per-combo, needs 2+ iterations) ---")
    plot_iteration_comparison_v7(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v7 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V7 tuning visualizations")
    parser.add_argument(
        '--test-iterations', type=int, default=0,
        help="Duplicate run1 N times for testing the iteration stability plot. "
             "Not needed once you have real multi-run data."
    )
    args = parser.parse_args()

    for opt_type, csv_paths in OPTIMIZATION_RUNS.items():
        paths = csv_paths.copy()

        if args.test_iterations > 0:
            # Duplicate the first CSV for testing
            base = paths[0]
            paths = [base] * args.test_iterations
            print(f"[TEST MODE] Duplicating {base} x{args.test_iterations}")

        run_v7(
            csv_paths=paths,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v7",
        )
