"""
Example usage of the modular tuning visualization suite.

Shows different ways to use the refactored code structure.
"""

from pathlib import Path
from tuning_viz import viz_driver
from tuning_viz.plot_types import plot_scatter, plot_heatmap, plot_boxplot
from tuning_viz.data_loader import TuningResultsLoader


# ============================================================================
# OPTION 1: Run complete visualization suite (recommended for most use cases)
# ============================================================================

def example_full_suite():
    """Generate all visualizations at once."""
    data, figs = viz_driver(
        csv_paths="path/to/mape_progress.csv",
        metric_column="val_mae",
        output_dir="./mape_visualizations",
        use_plotly=True
    )
    print(f"Generated {len(figs)} plots")


# ============================================================================
# OPTION 2: Run with multiple iterations (when you have 3+ runs)
# ============================================================================

def example_multiple_iterations():
    """Compare results across multiple iterations."""
    csv_paths = [
        "results/run1/mape_progress.csv",
        "results/run2/mape_progress.csv",
        "results/run3/mape_progress.csv",
    ]
    
    data, figs = viz_driver(
        csv_paths=csv_paths,
        metric_column="val_mae",
        output_dir="./multi_iteration_results",
        use_plotly=True
    )


# ============================================================================
# OPTION 3: Individual plot functions (for custom workflows)
# ============================================================================

def example_individual_plots():
    """Generate specific plots only."""
    loader = TuningResultsLoader(metric_column="val_mae")
    data = loader.load("path/to/mape_progress.csv")
    
    output_dir = "./custom_plots"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Only generate the plots you need
    fig_scatter = plot_scatter(data, "val_mae", output_dir=output_dir)
    fig_heatmap = plot_heatmap(data, "val_mae", output_dir=output_dir, interactive=True)
    fig_boxplot = plot_boxplot(data, "val_mae", output_dir=output_dir, interactive=True)
    
    # Process or modify figures before displaying
    print(f"Generated scatter, heatmap, and boxplot")


# ============================================================================
# OPTION 4: Compare MAPE vs MSE vs CRPS results side by side
# ============================================================================

def example_compare_optimization_types():
    """Compare results from different optimization types."""
    optimization_types = {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
        'crps': 'tune_experiment_results/crps_results_run1/crps_progress.csv',
    }
    
    results = {}
    for opt_type, csv_path in optimization_types.items():
        data, figs = viz_driver(
            csv_paths=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}",
            use_plotly=True
        )
        results[opt_type] = {'data': data, 'figs': figs}
    
    print(f"Generated visualizations for: {list(results.keys())}")


# ============================================================================
# OPTION 5: Custom metric column
# ============================================================================

def example_custom_metrics():
    """Use different metrics from the extracted results."""
    metrics_to_try = ['val_mae', 'val_mape', 'val_loss']
    
    for metric in metrics_to_try:
        try:
            data, figs = viz_driver(
                csv_paths="path/to/progress.csv",
                metric_column=metric,
                output_dir=f"./visualizations_{metric}",
                use_plotly=True
            )
            print(f"Success with metric: {metric}")
        except ValueError as e:
            print(f"Metric not available: {metric}")


if __name__ == "__main__":
    print("Tuning Visualization Suite - Example Usage\n")
    print("Available examples:")
    print("1. Full suite (all 6 plots)")
    print("2. Multiple iterations")
    print("3. Individual plots")
    print("4. Compare optimization types")
    print("5. Custom metrics")
    print("\nSee code comments for usage patterns.")

    for opt_type, csv_path in {
        'mape': 'tune_experiment_results/mape_results_run1/mape_progress.csv',
        'mse': 'tune_experiment_results/mse_results_run1/mse_progress.csv',
    }.items():
        data, figs = viz_driver(
            csv_paths=csv_path,
            metric_column="val_mae",
            output_dir=f"./visualizations_{opt_type}",
            use_plotly=True
        )
        print(f"Generated {len(figs)} plots for {opt_type}")