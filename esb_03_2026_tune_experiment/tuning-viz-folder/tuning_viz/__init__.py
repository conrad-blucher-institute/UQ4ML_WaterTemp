"""
Hyperparameter Tuning Visualization Suite

Modular, maintainable visualization system for ML tuning results across multiple iterations.

Structure:
    data_loader.py           - Data loading & parsing
    plot_types/              - Individual plot type modules
        scatter_plot.py      - Interactive scatter plot
        heatmap_plot.py      - Activation vs neurons heatmap
        boxplot_plot.py      - Performance distribution boxplot
        top_configs_plot.py   - Top configurations bar chart
        layers_impact_plot.py - Layers impact visualization
        iteration_comparison_plot.py - Multi-iteration comparison
    viz_driver.py            - Main orchestrator

Main entry point:
    from tuning_viz import viz_driver
    
    data, figs = viz_driver(
        csv_paths="mape_progress.csv",
        metric_column="val_mae",
        output_dir="./visualizations"
    )

For individual plots:
    from tuning_viz.plot_types import plot_scatter, plot_heatmap
    
    fig = plot_scatter(data, "val_mae", output_dir="./plots")
"""

from .viz_driver import viz_driver
from .data_loader import TuningResultsLoader

from .plot_types import *
from . import plot_types

__all__ = ['viz_driver', 'TuningResultsLoader'] + plot_types.__all__

__version__ = '1.0.0'
