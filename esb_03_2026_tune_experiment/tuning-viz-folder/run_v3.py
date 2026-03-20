"""
Run V3 visualizations for MAPE and MSE results.
Generates combined + per-cycle plots for everything.
"""

from pathlib import Path
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.scatter_plot_v3 import plot_scatter_v3
from tuning_viz.plot_types.heatmap_plot_v3 import plot_heatmap_v3
from tuning_viz.plot_types.boxplot_plot_v3 import plot_boxplot_v3
from tuning_viz.plot_types.top_configs_plot_v3 import plot_top_configs_v3
from tuning_viz.plot_types.layers_impact_plot_v3 import plot_layers_impact_v3
from tuning_viz.plot_types.iteration_comparison_plot_v3 import plot_iteration_comparison_v3


def run_v3(csv_path: str, metric_column: str, output_dir: str):
    """Load data and generate all v3 plots."""
    print(f"\n{'='*60}")
    print(f"V3 Visualizations: {output_dir}")
    print(f"{'='*60}")

    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load(csv_path)

    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    print(f"Cycles: {sorted(data['cycle'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Scatter Plots ---")
    plot_scatter_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Heatmaps ---")
    plot_heatmap_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Boxplots ---")
    plot_boxplot_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Top Configs ---")
    plot_top_configs_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Layers Impact ---")
    plot_layers_impact_v3(data, metric_column, output_dir=output_dir)

    print("\n--- Iteration Comparison ---")
    plot_iteration_comparison_v3(data, metric_column, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"All v3 plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    optimization_runs = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }

    for opt_type, csv_path in optimization_runs.items():
        run_v3(
            csv_path=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v3",
        )
