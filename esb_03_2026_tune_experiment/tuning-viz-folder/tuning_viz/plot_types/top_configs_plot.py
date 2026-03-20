"""
Bar chart: Top N best configurations ranked by metric.
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


def plot_top_configs(data: pd.DataFrame, metric_column: str, top_n: int = 15, output_dir: Optional[str] = None, interactive: bool = True) -> Optional[Union[go.Figure, plt.Figure]]:
    """
    Bar chart: Top N best configurations ranked by metric.
    
    Args:
        data: Parsed tuning results DataFrame
        metric_column: Name of the metric column to visualize
        top_n: Number of top configurations to show
        output_dir: Optional directory to save plot
        interactive: If True, use Plotly. If False, use matplotlib.
        
    Returns:
        Plotly figure object or matplotlib figure
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib.")
        interactive = False
    
    # Get top N configurations
    top_configs = data.nsmallest(top_n, metric_column)
    
    # Create readable labels
    labels = [
        f"{row['activation']}<br>L{row['num_layers']}-N{row['neurons']}<br>"
        f"LT{row['leadtime']}h-C{row['cycle']}<br>It{row['iteration']}"
        for _, row in top_configs.iterrows()
    ]
    
    if interactive:
        fig = go.Figure(
            data=go.Bar(
                y=top_configs[metric_column].values,
                x=labels,
                text=np.round(top_configs[metric_column].values, 4),
                textposition='auto',
                marker=dict(
                    color=top_configs[metric_column].values,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title=metric_column)
                )
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Best Configurations",
            yaxis_title=metric_column,
            height=500,
            width=1200,
            xaxis_tickangle=45
        )
        
        if output_dir:
            fig.write_html(f"{output_dir}/04_top_configurations.html")
            print(f"Saved: 04_top_configurations.html")
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        colors_bar = plt.cm.RdYlGn_r(
            (top_configs[metric_column] - top_configs[metric_column].min()) / 
            (top_configs[metric_column].max() - top_configs[metric_column].min())
        )
        
        ax.bar(range(len(labels)), top_configs[metric_column].values, color=colors_bar)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric_column)
        ax.set_title(f"Top {top_n} Best Configurations")
        
        if output_dir:
            fig.savefig(f"{output_dir}/04_top_configurations.png", dpi=150, bbox_inches='tight')
            print(f"Saved: 04_top_configurations.png")
        
        return fig
