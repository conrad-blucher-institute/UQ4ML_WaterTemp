"""
Visualize impact of number of layers on performance.
"""

from typing import Optional, Union
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_layers_impact(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None, interactive: bool = True) -> Optional[Union[go.Figure, plt.Figure]]:
    """
    Visualize impact of number of layers on performance.
    
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
        fig = px.box(
            data,
            x='num_layers',
            y=metric_column,
            color='activation',
            title=f"Impact of Number of Layers on {metric_column}",
            labels={'num_layers': 'Number of Layers'}
        )
        fig.update_layout(height=500, width=1000)
        
        if output_dir:
            fig.write_html(f"{output_dir}/05_layers_impact.html")
            print(f"Saved: 05_layers_impact.html")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, num_layers in enumerate(sorted(data['num_layers'].unique())):
            subset = data[data['num_layers'] == num_layers]
            ax.scatter([i] * len(subset), subset[metric_column], 
                      label=f"Layers: {num_layers}", alpha=0.6, s=50)
        
        ax.set_ylabel(metric_column)
        ax.set_xlabel("Number of Layers")
        ax.set_title(f"Impact of Number of Layers on {metric_column}")
        ax.legend()
        
        if output_dir:
            fig.savefig(f"{output_dir}/05_layers_impact.png", dpi=150, bbox_inches='tight')
            print(f"Saved: 05_layers_impact.png")
        
        return fig
