"""
Run V6 visualizations for MAPE and MSE results.

V6 changes (from v5):
    - scatter: JS rebuild, x-axis selector, all hyperparams as filters,
               single dark color, filtered traces fully hidden, legend toggle
    - heatmap: 3 rows (param pairs) × N cols (leadtimes), auto-resize toggle
    - boxplot: x-axis selector + color-by dropdown, replaces layers_impact
    - top_configs: fixed headers, cycle dropdown, + top 10 per cycle subplots
    - layers_impact: REMOVED (absorbed into boxplot x-axis selector)
    - parallel_coords: height auto-fills viewport
    - iteration_comparison: per-combo stability with failure threshold
                           (only runs with 2+ iterations)

Output files per optimization type:
    01_scatter.html
    02_heatmaps.html
    03_boxplot.html
    04_top_configs.html
    04_top10_per_cycle.html
    05_parallel_coords.html
    06_iteration_stability.html  (only with 2+ iterations)
"""

from pathlib import Path
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.scatter_plot_v6 import plot_scatter_v6
from tuning_viz.plot_types.heatmap_plot_v6 import plot_heatmap_v6
from tuning_viz.plot_types.boxplot_plot_v6 import plot_boxplot_v6
from tuning_viz.plot_types.top_configs_plot_v6 import plot_top_configs_v6, plot_top10_per_cycle_v6
from tuning_viz.plot_types.parallel_coords_v6 import plot_parallel_coords_v6
from tuning_viz.plot_types.iteration_comparison_plot_v6 import plot_iteration_comparison_v6


def run_v6(csv_path: str, metric_column: str, output_dir: str):
    """Load data and generate all v6 plots."""
    print(f"\n{'='*60}")
    print(f"V6 Visualizations: {output_dir}")
    print(f"{'='*60}")

    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load(csv_path)

    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Scatter (x-axis selector, combined filters) ---")
    plot_scatter_v6(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmaps (param pairs × leadtimes, auto-resize toggle) ---")
    plot_heatmap_v6(data, metric_column, output_dir=output_dir)

    print("\n--- Boxplot (x-axis + color-by selectors) ---")
    plot_boxplot_v6(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs (cycle dropdown + top 10 per cycle) ---")
    plot_top_configs_v6(data, metric_column, output_dir=output_dir)
    plot_top10_per_cycle_v6(data, metric_column, output_dir=output_dir)

    print("\n--- Parallel Coordinates (auto-height) ---")
    plot_parallel_coords_v6(data, metric_column, output_dir=output_dir)

    print("\n--- Iteration Stability (per-combo, needs 2+ iterations) ---")
    plot_iteration_comparison_v6(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v6 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    optimization_runs = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }

    for opt_type, csv_path in optimization_runs.items():
        run_v6(
            csv_path=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v6",
        )
