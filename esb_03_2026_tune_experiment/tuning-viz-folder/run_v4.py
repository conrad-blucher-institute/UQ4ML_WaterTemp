"""
Run V4 visualizations for MAPE and MSE results.

V4 changes (from v3):
    - scatter: dropdown menus for filtering activation & leadtime
    - heatmap: grid layout (rows=leadtimes, cols=param pairs, leadtime NOT a heatmap param)
    - parallel coords: new W&B-style parallel coordinates plot

Unchanged from v3 (still used directly):
    - boxplot: stitched HTML, independent legends, categorical x, per-cycle
    - top_configs: separate columns, overall + per-cycle tables
    - layers_impact: num_layers x-axis, colored by activation, per-cycle
    - iteration_comparison: best metric per iteration (skips if only 1)
"""

from pathlib import Path
from tuning_viz.data_loader import TuningResultsLoader

# V4 (changed)
from tuning_viz.plot_types.scatter_plot_v4 import plot_scatter_v4
from tuning_viz.plot_types.heatmap_plot_v4 import plot_heatmap_v4
from tuning_viz.plot_types.parallel_coords_v4 import plot_parallel_coords_v4

# V3 (unchanged)
from tuning_viz.plot_types.boxplot_plot_v3 import plot_boxplot_v3
from tuning_viz.plot_types.top_configs_plot_v3 import plot_top_configs_v3
from tuning_viz.plot_types.layers_impact_plot_v3 import plot_layers_impact_v3
from tuning_viz.plot_types.iteration_comparison_plot_v3 import plot_iteration_comparison_v3


def run_v4(csv_path: str, metric_column: str, output_dir: str):
    """Load data and generate all v4 plots."""
    print(f"\n{'='*60}")
    print(f"V4 Visualizations: {output_dir}")
    print(f"{'='*60}")

    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load(csv_path)

    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Scatter Plots (v4: dropdown filters) ---")
    plot_scatter_v4(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmap Grids (v4: rows=leadtimes, cols=param pairs) ---")
    plot_heatmap_v4(data, metric_column, output_dir=output_dir)

    print("\n--- Boxplots (v3: independent legends, stitched) ---")
    plot_boxplot_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs (v3: tables, overall + per-cycle) ---")
    plot_top_configs_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Layers Impact (v3) ---")
    plot_layers_impact_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Iteration Comparison (v3) ---")
    plot_iteration_comparison_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Parallel Coordinates (v4: W&B-style) ---")
    plot_parallel_coords_v4(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v4 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    optimization_runs = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }

    for opt_type, csv_path in optimization_runs.items():
        run_v4(
            csv_path=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v4",
        )
