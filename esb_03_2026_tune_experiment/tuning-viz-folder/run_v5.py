"""
Run V5 visualizations for MAPE and MSE results.

V5 changes (from v4):
    - ALL plots: single output file per plot type with cycle dropdown
    - scatter: combined JS filtering (activation × leadtime × cycle AND logic),
               legend toggle button, cycle in legend names, clean layout
    - heatmap: local color scale per heatmap (not global), subtitle,
               cycle dropdown
    - boxplot: cycle dropdown (stitched figures)
    - top_configs: cycle dropdown, readable light headers
    - layers_impact: cycle dropdown
    - parallel_coords: single file only, tick marks show actual values only
"""

from pathlib import Path
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.scatter_plot_v5 import plot_scatter_v5
from tuning_viz.plot_types.heatmap_plot_v5 import plot_heatmap_v5
from tuning_viz.plot_types.boxplot_plot_v5 import plot_boxplot_v5
from tuning_viz.plot_types.top_configs_plot_v5 import plot_top_configs_v5
from tuning_viz.plot_types.layers_impact_plot_v5 import plot_layers_impact_v5
from tuning_viz.plot_types.parallel_coords_v5 import plot_parallel_coords_v5


def run_v5(csv_path: str, metric_column: str, output_dir: str):
    """Load data and generate all v5 plots."""
    print(f"\n{'='*60}")
    print(f"V5 Visualizations: {output_dir}")
    print(f"{'='*60}")

    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load(csv_path)

    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Scatter (combined JS filters, legend toggle) ---")
    plot_scatter_v5(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmaps (local color scale, cycle dropdown) ---")
    plot_heatmap_v5(data, metric_column, output_dir=output_dir)

    print("\n--- Boxplots (cycle dropdown, stitched) ---")
    plot_boxplot_v5(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs (cycle dropdown, readable headers) ---")
    plot_top_configs_v5(data, metric_column, output_dir=output_dir)

    print("\n--- Layers Impact (cycle dropdown) ---")
    plot_layers_impact_v5(data, metric_column, output_dir=output_dir)

    print("\n--- Parallel Coordinates (single file, clean ticks) ---")
    plot_parallel_coords_v5(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v5 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    optimization_runs = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }

    for opt_type, csv_path in optimization_runs.items():
        run_v5(
            csv_path=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v5",
        )
