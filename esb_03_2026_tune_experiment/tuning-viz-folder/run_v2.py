"""
Run V2 visualizations for MAPE and MSE results.
Generates all v2 plots into separate output folders.
"""

from pathlib import Path
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.scatter_plot_v2 import plot_scatter_v2
from tuning_viz.plot_types.heatmap_plot_v2 import plot_heatmap_v2
from tuning_viz.plot_types.boxplot_plot_v2 import plot_boxplot_v2
from tuning_viz.plot_types.top_configs_plot_v2 import plot_top_configs_v2
from tuning_viz.plot_types.layers_impact_plot_v2 import plot_layers_impact_v2
from tuning_viz.plot_types.iteration_comparison_plot_v2 import plot_iteration_comparison_v2


def run_v2(csv_path: str, metric_column: str, output_dir: str):
    """Load data and generate all v2 plots."""
    print(f"\n{'='*60}")
    print(f"V2 Visualizations: {output_dir}")
    print(f"{'='*60}")

    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load(csv_path)

    print(f"Loaded {len(data)} configurations")
    print(f"Metric: {metric_column}")
    print(f"Range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_scatter_v2(data, metric_column, output_dir=output_dir)
    plot_heatmap_v2(data, metric_column, output_dir=output_dir)
    plot_boxplot_v2(data, metric_column, output_dir=output_dir)
    plot_top_configs_v2(data, metric_column, output_dir=output_dir)
    plot_layers_impact_v2(data, metric_column, output_dir=output_dir)
    plot_iteration_comparison_v2(data, metric_column, output_dir=output_dir)

    print(f"\nAll v2 plots saved to: {output_dir}")


if __name__ == "__main__":
    optimization_runs = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }

    for opt_type, csv_path in optimization_runs.items():
        run_v2(
            csv_path=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}_v2",
        )
