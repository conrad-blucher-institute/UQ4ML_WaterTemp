"""
Box plot: metric distribution grouped by activation function.
"""

from typing import Optional, Union
import pandas as pd

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


def plot_boxplot(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None, interactive: bool = True) -> Optional[Union[go.Figure, plt.Figure]]:
    """
    Box plot: metric distribution grouped by activation function.
    Shows stability across iterations and configurations.
    
    Args:
        data: Parsed tuning results DataFrame
        metric_column: Name of the metric column to visualize
        output_dir: Optional directory to save plot
        interactive: If True, use Plotly. If False, use matplotlib.
        
    Returns:
        Plotly figure object or matplotlib figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib.")
        interactive = False
    
    if interactive:
        fig = go.Figure()
        
        for activation in sorted(data['activation'].unique()):
            subset = data[data['activation'] == activation]
            fig.add_trace(
                go.Box(
                    y=subset[metric_column],
                    name=activation,
                    boxmean='sd'
                )
            )
        
        fig.update_layout(
            title=f"Performance Distribution by Activation Function",
            yaxis_title=metric_column,
            xaxis_title="Activation Function",
            height=500,
            showlegend=False
        )
        
        if output_dir:
            fig.write_html(f"{output_dir}/03_boxplot_activation.html")
            print(f"Saved: 03_boxplot_activation.html")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        activations = sorted(data['activation'].unique())
        data_by_act = [data[data['activation'] == a][metric_column].values 
                      for a in activations]
        
        bp = ax.boxplot(data_by_act, labels=activations, patch_artist=True)
        ax.set_ylabel(metric_column)
        ax.set_xlabel("Activation Function")
        ax.set_title("Performance Distribution by Activation Function")
        
        if output_dir:
            fig.savefig(f"{output_dir}/03_boxplot_activation.png", dpi=150, bbox_inches='tight')
            print(f"Saved: 03_boxplot_activation.png")
        
        return fig
