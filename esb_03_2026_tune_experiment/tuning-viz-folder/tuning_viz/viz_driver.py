"""
Orchestrator for hyperparameter tuning visualizations.

Main driver that coordinates data loading and all plot generation.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

from .data_loader import TuningResultsLoader
from .plot_types import (
    plot_scatter,
    plot_heatmap,
    plot_boxplot,
    plot_top_configs,
    plot_layers_impact,
    plot_iteration_comparison,
)


def viz_driver(
    csv_paths: List[str],
    metric_column: str = "val_mae",
    output_dir: str = "./tuning_visualizations",
    use_plotly: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main orchestrator for tuning visualizations.
    
    Args:
        csv_paths: List of CSV file paths (can be single or multiple iterations)
        metric_column: Which metric to visualize (e.g., 'val_mae', 'val_mape', 'val_loss')
        output_dir: Directory to save visualizations
        use_plotly: If True, use interactive Plotly plots. If False, use matplotlib.
        
    Returns:
        Tuple of (data DataFrame, results dictionary with figure objects)
        
    Example:
        >>> data, figs = viz_driver(
        ...     csv_paths="mape_progress.csv",
        ...     metric_column="val_mae",
        ...     output_dir="./mape_visualizations"
        ... )
    """
    # Handle single csv path
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    print(f"Loading {len(csv_paths)} CSV file(s)...")
    loader = TuningResultsLoader(metric_column=metric_column)
    data = loader.load_multiple(csv_paths)
    
    print(f"Loaded {len(data)} configurations across {data['iteration'].nunique()} iteration(s)")
    print(f"Metric being visualized: {metric_column}")
    print(f"Value range: {data[metric_column].min():.4f} - {data[metric_column].max():.4f}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print(f"\nGenerating visualizations...")
    results = {}
    
    results['scatter'] = plot_scatter(data, metric_column, output_dir=output_dir)
    results['heatmap'] = plot_heatmap(data, metric_column, output_dir=output_dir, interactive=use_plotly)
    results['boxplot'] = plot_boxplot(data, metric_column, output_dir=output_dir, interactive=use_plotly)
    results['top_configs'] = plot_top_configs(data, metric_column, top_n=15, output_dir=output_dir, interactive=use_plotly)
    results['layers_impact'] = plot_layers_impact(data, metric_column, output_dir=output_dir, interactive=use_plotly)
    results['iterations'] = plot_iteration_comparison(data, metric_column, output_dir=output_dir, interactive=use_plotly)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return data, results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        metric = sys.argv[2] if len(sys.argv) > 2 else "val_mae"
        data, figs = viz_driver(csv_path, metric_column=metric)
    else:
        print("Usage: python viz_driver.py <csv_path> [metric_column]")
        print("Example: python viz_driver.py mape_progress.csv val_mae")
