"""
Compare iteration stability: best metric value from each iteration.
"""

from typing import Optional, Union
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_iteration_comparison(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None, interactive: bool = True) -> Optional[Union[go.Figure, plt.Figure]]:
    """
    Compare iteration stability: best metric value from each iteration.
    Only useful if data has multiple iterations.
    
    Args:
        data: Parsed tuning results DataFrame
        metric_column: Name of the metric column to visualize
        output_dir: Optional directory to save plot
        interactive: If True, use Plotly. If False, use matplotlib.
        
    Returns:
        Plotly figure object, matplotlib figure, or None if only 1 iteration
    """
    if 'iteration' not in data.columns or data['iteration'].nunique() < 2:
        print("Only 1 iteration in data. Skipping iteration comparison plot.")
        return None
    
    if interactive and not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib.")
        interactive = False
    
    # Best metric per iteration
    best_per_iter = data.groupby('iteration')[metric_column].min()
    
    if interactive:
        fig = go.Figure(
            data=go.Bar(
                x=best_per_iter.index,
                y=best_per_iter.values,
                text=np.round(best_per_iter.values, 4),
                textposition='auto',
                marker=dict(color='steelblue')
            )
        )
        fig.update_layout(
            title=f"Best {metric_column} per Iteration",
            xaxis_title="Iteration",
            yaxis_title=f"Best {metric_column}",
            height=400
        )
        
        if output_dir:
            fig.write_html(f"{output_dir}/06_iteration_comparison.html")
            print(f"Saved: 06_iteration_comparison.html")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(best_per_iter.index, best_per_iter.values, color='steelblue')
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Best {metric_column}")
        ax.set_title(f"Best {metric_column} per Iteration")
        
        if output_dir:
            fig.savefig(f"{output_dir}/06_iteration_comparison.png", dpi=150, bbox_inches='tight')
            print(f"Saved: 06_iteration_comparison.png")
        
        return fig
