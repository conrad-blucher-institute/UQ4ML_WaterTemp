"""
Interactive scatter plot: neurons vs metric, colored by activation, sized by layers.
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Interactive scatter: neurons vs metric, colored by activation, sized by layers.
    Faceted by leadtime for better exploration.
    
    Args:
        data: Parsed tuning results DataFrame
        metric_column: Name of the metric column to visualize
        output_dir: Optional directory to save HTML plot
        
    Returns:
        Plotly figure object or None if Plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None
    
    # Get unique leadtimes for faceting
    leadtimes = sorted(data['leadtime'].unique())
    
    fig = make_subplots(
        rows=1, cols=len(leadtimes),
        subplot_titles=[f"Leadtime: {lt}h" for lt in leadtimes],
        specs=[[{"type": "scatter"}] * len(leadtimes)]
    )
    
    activations = data['activation'].unique()
    colors = px.colors.qualitative.Set1
    color_map = {act: colors[i % len(colors)] for i, act in enumerate(activations)}
    
    for col, leadtime in enumerate(leadtimes, start=1):
        subset = data[data['leadtime'] == leadtime]
        
        for activation in activations:
            act_data = subset[subset['activation'] == activation]
            
            fig.add_trace(
                go.Scatter(
                    x=act_data['neurons'],
                    y=act_data[metric_column],
                    mode='markers',
                    name=activation,
                    marker=dict(
                        size=act_data['num_layers'] * 4,
                        color=color_map[activation],
                        opacity=0.7,
                        line=dict(width=1)
                    ),
                    text=[
                        f"Activation: {a}<br>Neurons: {n}<br>Layers: {l}<br>"
                        f"{metric_column}: {m:.4f}<br>Iteration: {it}"
                        for a, n, l, m, it in zip(
                            act_data['activation'],
                            act_data['neurons'],
                            act_data['num_layers'],
                            act_data[metric_column],
                            act_data['iteration']
                        )
                    ],
                    hoverinfo='text',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
        
        fig.update_xaxes(title_text="Hidden Units", row=1, col=col)
        if col == 1:
            fig.update_yaxes(title_text=metric_column, row=1, col=col)
    
    fig.update_layout(
        title=f"Hyperparameter Search: {metric_column} vs Neurons (size=layers)",
        height=500,
        hovermode='closest',
        width=1400
    )
    
    if output_dir:
        fig.write_html(f"{output_dir}/01_scatter_plot.html")
        print(f"Saved: 01_scatter_plot.html")
    
    return fig
